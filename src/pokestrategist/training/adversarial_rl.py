"""Adversarial self-play RL: Student vs Frozen Teacher on replay states.

Key differences from the previous conservative duel RL:
1. Reward = teacher's world-model evaluation of action quality (not just gold-label hit)
2. Counterfactual: evaluates ALL candidate actions via convert_score proxy
3. Adversarial: student explicitly tries to outperform teacher's own action choices
4. KL-anchored: stays close to teacher to prevent collapse
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Subset

from pokestrategist.data.hidden_move_prior import HiddenMovePrior
from pokestrategist.data.team_preview_prior import TeamPreviewPriorCatalog
from pokestrategist.models.decision_model import PokeStrategistDecisionConfig, build_model_for_checkpoint
from pokestrategist.training.dataset import DecisionTensorDataset, grouped_replay_split
from pokestrategist.training.trainer import (
    _build_loader,
    _configure_torch_runtime,
    _epoch_learning_rate,
    _masked_action_ce,
    _move_batch_to_device,
    _ranking_score,
    _resolve_num_workers,
    _resolve_pin_memory,
    _selection_score,
    _switch_selection_score,
    _write_model_artifact_manifest,
    evaluate_model,
)


@dataclass(slots=True)
class AdversarialRLConfig:
    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 2e-5
    device: str = "cpu"
    train_num_workers: int | None = None
    eval_num_workers: int | None = None
    pin_memory: bool | None = None
    prefetch_factor: int = 4
    persistent_workers: bool = True
    enable_tf32: bool = True
    split_seed: int = 0
    hidden_candidate_topk: int | None = None
    max_grad_norm: float | None = 1.0
    warmup_epochs: int = 1
    min_learning_rate_ratio: float = 0.4

    # RL-specific hyperparameters
    policy_temperature: float = 1.2
    reference_temperature: float = 1.0
    kl_weight: float = 0.08
    entropy_weight: float = 0.005
    imitation_weight: float = 0.2
    convert_score_weight: float = 0.5  # weight on convert_score axis in future
    safe_score_weight: float = 0.15    # weight on safe_score axis
    tempo_score_weight: float = 0.15   # weight on tempo_score axis
    resource_score_weight: float = 0.2 # weight on resource delta axis
    switch_sample_weight: float = 1.0

    use_team_preview_prior: bool | None = None
    use_usage_priors: bool | None = None
    usage_data_dir: str | None = None


def _checkpoint_model_config_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config_payload = dict(checkpoint["model_config"])
    if "use_static_rule_features" not in model_config_payload:
        model_config_payload["use_static_rule_features"] = False
    if "candidate_prior_feature_dim" not in model_config_payload:
        prior_weight = checkpoint["model_state"].get("prior_feature_proj.0.weight")
        model_config_payload["candidate_prior_feature_dim"] = int(prior_weight.shape[1]) if prior_weight is not None else 2
    return model_config_payload


def _load_checkpoint_bundle(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = PokeStrategistDecisionConfig(**_checkpoint_model_config_payload(checkpoint))
    model = build_model_for_checkpoint(model_config, checkpoint["model_state"])
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "model": model,
        "model_config": model_config,
        "trainer_config": dict(checkpoint.get("trainer_config", {})),
        "hidden_move_prior": HiddenMovePrior.from_dict(checkpoint.get("hidden_move_prior")),
        "team_preview_prior": TeamPreviewPriorCatalog.from_dict(checkpoint["team_preview_prior"])
        if checkpoint.get("team_preview_prior")
        else None,
    }


def _gold_policy_score(metrics: dict[str, float]) -> float:
    return 0.65 * metrics.get("gold_top1", 0.0) + 0.35 * metrics.get("gold_in_top3", 0.0)


def _world_model_action_value(
    outputs: dict[str, Tensor],
    config: AdversarialRLConfig,
) -> Tensor:
    """Compute action-conditioned expected value using the model's world-model heads.

    Uses candidate_future_mean (8-dim future summary) weighted by configurable
    axis weights to produce a scalar value per candidate action.
    """
    future_mean = outputs["candidate_future_mean"]  # [B, N, 8]
    # Axis mapping: 0=delta_plan, 1=delta_belief, 2=delta_resource,
    #   3=delta_phase, 4=delta_unlock, 5=safe_score, 6=tempo_score, 7=convert_score
    weights = future_mean.new_tensor([
        0.0,  # delta_plan - not directly value-relevant
        0.0,  # delta_belief - indirect
        config.resource_score_weight,  # delta_resource
        0.0,  # delta_phase - indirect
        0.0,  # delta_unlock - redundant with convert
        config.safe_score_weight,      # safe_score
        config.tempo_score_weight,     # tempo_score
        config.convert_score_weight,   # convert_score
    ])
    # Also blend in typed_consequence scores for richer world-model eval
    if "typed_consequence_scores" in outputs:
        typed = outputs["typed_consequence_scores"]  # [B, N]
    else:
        typed = future_mean.new_zeros(future_mean.shape[:2])
    wm_value = (future_mean * weights).sum(dim=-1) + 0.15 * typed  # [B, N]
    return wm_value


def _masked_policy(scores: Tensor, mask: Tensor, *, temperature: float) -> tuple[Tensor, Tensor]:
    scaled_scores = scores.masked_fill(~mask, -1e9) / max(float(temperature), 1e-6)
    log_probs = scaled_scores.log_softmax(dim=-1)
    return log_probs.exp(), log_probs


def _adversarial_loss(
    student_outputs: dict[str, Tensor],
    teacher_outputs: dict[str, Tensor],
    batch: dict[str, Any],
    config: AdversarialRLConfig,
) -> tuple[Tensor, dict[str, float]]:
    mask = batch["legal_action_mask"].bool()

    # --- World-model action values (computed from TEACHER's predictions) ---
    # Use teacher's world model as the unbiased evaluator
    with torch.no_grad():
        teacher_wm_value = _world_model_action_value(teacher_outputs, config)  # [B, N]

    # --- Action distributions ---
    student_probs, student_log_probs = _masked_policy(
        student_outputs["legal_action_scores"], mask, temperature=config.policy_temperature,
    )
    teacher_probs, teacher_log_probs = _masked_policy(
        teacher_outputs["legal_action_scores"], mask, temperature=config.reference_temperature,
    )

    # --- Expected value under each policy (using teacher's world model) ---
    student_expected_value = (student_probs * teacher_wm_value).sum(dim=1)  # [B]
    teacher_expected_value = (teacher_probs * teacher_wm_value).sum(dim=1)  # [B]

    # Adversarial advantage: how much better is student's expected value vs teacher's
    advantage = student_expected_value - teacher_expected_value  # [B]

    # --- Sample weighting for switch states ---
    sample_weight = torch.where(batch["gold_is_switch"], config.switch_sample_weight, 1.0).to(advantage.dtype)
    normalized_weight = sample_weight / sample_weight.sum().clamp_min(1e-6)

    # --- Policy loss: maximize advantage ---
    policy_advantage_loss = -(advantage * normalized_weight).sum()

    # --- KL constraint to teacher ---
    kl_per_row = (student_probs * (student_log_probs - teacher_log_probs.detach())).sum(dim=1)
    kl_loss = (kl_per_row * normalized_weight).sum()

    # --- Entropy bonus for exploration ---
    entropy_per_row = -(student_probs * student_log_probs).sum(dim=1)
    entropy = (entropy_per_row * normalized_weight).sum()

    # --- Imitation anchor (keep gold action in mind) ---
    imitation_loss = _masked_action_ce(
        student_outputs["legal_action_scores"],
        batch["legal_action_mask"],
        batch["legal_target"],
        sample_weight=sample_weight,
    )

    # --- Total loss ---
    total = (
        policy_advantage_loss
        + config.kl_weight * kl_loss
        + config.imitation_weight * imitation_loss
        - config.entropy_weight * entropy
    )

    # --- Diagnostics ---
    # Top-1 action quality comparison
    student_top1_idx = student_outputs["legal_action_scores"].masked_fill(~mask, -1e9).argmax(dim=1)
    teacher_top1_idx = teacher_outputs["legal_action_scores"].masked_fill(~mask, -1e9).argmax(dim=1)

    student_top1_value = teacher_wm_value.gather(1, student_top1_idx.unsqueeze(1)).squeeze(1)
    teacher_top1_value = teacher_wm_value.gather(1, teacher_top1_idx.unsqueeze(1)).squeeze(1)

    # Gold action value
    valid = batch["legal_target"] >= 0
    gold_value = teacher_wm_value.gather(1, batch["legal_target"].unsqueeze(1).clamp_min(0)).squeeze(1)

    return total, {
        "adv_loss": float(total.detach().cpu()),
        "adv_policy_advantage": float(advantage.mean().detach().cpu()),
        "adv_student_expected_value": float(student_expected_value.mean().detach().cpu()),
        "adv_teacher_expected_value": float(teacher_expected_value.mean().detach().cpu()),
        "adv_kl": float(kl_loss.detach().cpu()),
        "adv_entropy": float(entropy.detach().cpu()),
        "adv_imitation_loss": float(imitation_loss.detach().cpu()),
        "adv_student_top1_value": float(student_top1_value.mean().detach().cpu()),
        "adv_teacher_top1_value": float(teacher_top1_value.mean().detach().cpu()),
        "adv_top1_value_gap": float((student_top1_value - teacher_top1_value).mean().detach().cpu()),
        "adv_top1_beat_rate": float(student_top1_value.gt(teacher_top1_value).float().mean().detach().cpu()),
        "adv_top1_tie_rate": float(student_top1_value.eq(teacher_top1_value).float().mean().detach().cpu()),
        "adv_gold_value": float(gold_value[valid].mean().detach().cpu()) if valid.any() else 0.0,
    }


def train_adversarial_policy(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    reference_checkpoint_path: str | Path | None = None,
    rl_config: AdversarialRLConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    rl_config = rl_config or AdversarialRLConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_bundle = _load_checkpoint_bundle(checkpoint_path)
    reference_bundle = _load_checkpoint_bundle(reference_checkpoint_path or checkpoint_path)
    actor_trainer_config = dict(actor_bundle["trainer_config"])

    resolved_hidden_candidate_topk = int(
        rl_config.hidden_candidate_topk
        if rl_config.hidden_candidate_topk is not None
        else actor_trainer_config.get("hidden_candidate_topk", 4)
    )
    resolved_use_usage_priors = (
        bool(rl_config.use_usage_priors)
        if rl_config.use_usage_priors is not None
        else bool(actor_trainer_config.get("use_usage_priors", True))
    )
    resolved_use_team_preview_prior = (
        bool(rl_config.use_team_preview_prior)
        if rl_config.use_team_preview_prior is not None
        else actor_bundle["team_preview_prior"] is not None
    )
    resolved_usage_data_dir = (
        rl_config.usage_data_dir if rl_config.usage_data_dir is not None
        else actor_trainer_config.get("usage_data_dir")
    )
    hidden_move_prior = actor_bundle["hidden_move_prior"]
    team_preview_prior = actor_bundle["team_preview_prior"] if resolved_use_team_preview_prior else None

    print(f"Indexing dataset from {dataset_path}...", flush=True)
    dataset = DecisionTensorDataset(
        dataset_path,
        max_samples=max_samples,
        hidden_move_prior=hidden_move_prior,
        team_preview_prior=team_preview_prior,
        hidden_candidate_topk=resolved_hidden_candidate_topk,
        use_usage_priors=resolved_use_usage_priors,
        usage_data_dir=resolved_usage_data_dir,
    )
    train_indices, val_indices = grouped_replay_split(dataset.replay_ids, seed=rl_config.split_seed)
    if not train_indices and val_indices:
        train_indices = [val_indices[0]]
        val_indices = val_indices[1:] or list(train_indices)
    if not val_indices and train_indices:
        val_indices = [train_indices[0]]
        train_indices = train_indices[1:] or list(val_indices)
    print(f"Loaded dataset with {len(dataset)} samples; train={len(train_indices)} val={len(val_indices)}.", flush=True)

    device = torch.device(rl_config.device)
    _configure_torch_runtime(device, enable_tf32=rl_config.enable_tf32)
    train_num_workers = _resolve_num_workers(rl_config.train_num_workers, device=device)
    eval_num_workers = _resolve_num_workers(rl_config.eval_num_workers, device=device, fallback=0)
    pin_memory = _resolve_pin_memory(rl_config.pin_memory, device=device)
    print(f"Using device={device}, train_workers={train_num_workers}, eval_workers={eval_num_workers}, pin_memory={pin_memory}.", flush=True)

    train_loader = _build_loader(
        Subset(dataset, train_indices),
        batch_size=rl_config.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=rl_config.prefetch_factor,
        persistent_workers=rl_config.persistent_workers,
    )
    val_loader = _build_loader(
        Subset(dataset, val_indices),
        batch_size=rl_config.batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=rl_config.prefetch_factor,
        persistent_workers=rl_config.persistent_workers,
    )

    model = actor_bundle["model"]
    reference_model = reference_bundle["model"]
    model.to(device)
    reference_model.to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=rl_config.learning_rate)
    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_ranking_score = float("-inf")
    best_ranking_metrics: dict[str, float] | None = None
    best_ranking_epoch = 0
    best_switch_score = float("-inf")
    best_switch_metrics: dict[str, float] | None = None
    best_switch_epoch = 0
    non_blocking = device.type == "cuda" and pin_memory

    checkpoint_trainer_config = dict(actor_trainer_config)
    checkpoint_trainer_config.update({
        "hidden_candidate_topk": resolved_hidden_candidate_topk,
        "use_usage_priors": resolved_use_usage_priors,
        "use_team_preview_prior": resolved_use_team_preview_prior,
        "usage_data_dir": resolved_usage_data_dir,
    })

    def _checkpoint_payload(metrics: dict[str, float], *, epoch: int) -> dict[str, Any]:
        payload_metrics = dict(metrics)
        payload_metrics["ranking_score"] = _ranking_score(metrics)
        payload_metrics["switch_selection_score"] = _switch_selection_score(metrics)
        payload_metrics["selection_score"] = _selection_score(metrics)
        payload_metrics["gold_policy_score"] = _gold_policy_score(metrics)
        payload_metrics["best_epoch"] = float(epoch)
        return {
            "model_state": model.state_dict(),
            "model_config": asdict(actor_bundle["model_config"]),
            "trainer_config": checkpoint_trainer_config,
            "rl_config": asdict(rl_config),
            "metrics": payload_metrics,
            "hidden_move_prior": hidden_move_prior.to_dict() if hidden_move_prior is not None else None,
            "team_preview_prior": team_preview_prior.to_dict() if team_preview_prior is not None else None,
            "initial_checkpoint": str(actor_bundle["checkpoint_path"]),
            "reference_checkpoint": str(reference_bundle["checkpoint_path"]),
        }

    for epoch in range(rl_config.epochs):
        model.train()
        epoch_stats: dict[str, list[float]] = {
            "adv_loss": [],
            "adv_policy_advantage": [],
            "adv_student_expected_value": [],
            "adv_teacher_expected_value": [],
            "adv_kl": [],
            "adv_entropy": [],
            "adv_imitation_loss": [],
            "adv_student_top1_value": [],
            "adv_teacher_top1_value": [],
            "adv_top1_value_gap": [],
            "adv_top1_beat_rate": [],
            "adv_top1_tie_rate": [],
            "adv_gold_value": [],
        }
        current_learning_rate = _epoch_learning_rate(
            base_learning_rate=rl_config.learning_rate,
            epoch_index=epoch,
            total_epochs=rl_config.epochs,
            warmup_epochs=rl_config.warmup_epochs,
            min_learning_rate_ratio=rl_config.min_learning_rate_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate
        print(f"Starting adversarial RL epoch {epoch + 1}/{rl_config.epochs}...", flush=True)

        for batch in train_loader:
            tensor_batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            with torch.inference_mode():
                teacher_outputs = reference_model(tensor_batch)
            student_outputs = model(tensor_batch)
            loss, stats = _adversarial_loss(student_outputs, teacher_outputs, tensor_batch, rl_config)
            optimizer.zero_grad()
            loss.backward()
            if rl_config.max_grad_norm is not None and rl_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rl_config.max_grad_norm)
            optimizer.step()
            for key, value in stats.items():
                epoch_stats[key].append(value)

        metrics = evaluate_model(model, val_loader, device)
        metrics.update({
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in epoch_stats.items()
        })
        metrics["epoch"] = float(epoch + 1)
        metrics["learning_rate"] = current_learning_rate
        metrics["ranking_score"] = _ranking_score(metrics)
        metrics["switch_selection_score"] = _switch_selection_score(metrics)
        metrics["selection_score"] = _selection_score(metrics)
        metrics["gold_policy_score"] = _gold_policy_score(metrics)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

        if metrics["gold_policy_score"] > best_score:
            best_score = metrics["gold_policy_score"]
            best_metrics = dict(metrics)
            best_epoch = epoch + 1
            checkpoint = _checkpoint_payload(best_metrics, epoch=best_epoch)
            torch.save(checkpoint, output_dir / "best_model.pt")
            torch.save(checkpoint, output_dir / "model.pt")
            (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

        if metrics["ranking_score"] > best_ranking_score:
            best_ranking_score = metrics["ranking_score"]
            best_ranking_metrics = dict(metrics)
            best_ranking_epoch = epoch + 1
            torch.save(_checkpoint_payload(best_ranking_metrics, epoch=best_ranking_epoch),
                       output_dir / "best_ranking_model.pt")
            (output_dir / "ranking_metrics.json").write_text(
                json.dumps(best_ranking_metrics, indent=2), encoding="utf-8")

        if metrics["switch_selection_score"] > best_switch_score:
            best_switch_score = metrics["switch_selection_score"]
            best_switch_metrics = dict(metrics)
            best_switch_epoch = epoch + 1
            torch.save(_checkpoint_payload(best_switch_metrics, epoch=best_switch_epoch),
                       output_dir / "best_switch_model.pt")
            (output_dir / "switch_metrics.json").write_text(
                json.dumps(best_switch_metrics, indent=2), encoding="utf-8")

    final_metrics = history[-1] if history else evaluate_model(model, val_loader, device)
    final_metrics["ranking_score"] = _ranking_score(final_metrics)
    final_metrics["switch_selection_score"] = _switch_selection_score(final_metrics)
    final_metrics["selection_score"] = _selection_score(final_metrics)
    final_metrics["gold_policy_score"] = _gold_policy_score(final_metrics)
    best_metrics = best_metrics or final_metrics

    final_checkpoint = _checkpoint_payload(final_metrics, epoch=rl_config.epochs)
    torch.save(final_checkpoint, output_dir / "final_model.pt")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if hidden_move_prior is not None:
        (output_dir / "hidden_move_prior.json").write_text(
            json.dumps(hidden_move_prior.to_dict(), indent=2), encoding="utf-8")
    if team_preview_prior is not None:
        (output_dir / "team_preview_prior.json").write_text(
            json.dumps(team_preview_prior.to_dict(), indent=2), encoding="utf-8")

    if best_epoch:
        best_metrics = dict(best_metrics)
        best_metrics["best_epoch"] = float(best_epoch)
        best_metrics["ranking_score"] = _ranking_score(best_metrics)
        best_metrics["switch_selection_score"] = _switch_selection_score(best_metrics)
        best_metrics["selection_score"] = _selection_score(best_metrics)
        best_metrics["gold_policy_score"] = _gold_policy_score(best_metrics)
        (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    if best_ranking_metrics is not None:
        best_ranking_metrics = dict(best_ranking_metrics)
        best_ranking_metrics["best_epoch"] = float(best_ranking_epoch)
        best_ranking_metrics["ranking_score"] = _ranking_score(best_ranking_metrics)
        best_ranking_metrics["switch_selection_score"] = _switch_selection_score(best_ranking_metrics)
        best_ranking_metrics["selection_score"] = _selection_score(best_ranking_metrics)
        best_ranking_metrics["gold_policy_score"] = _gold_policy_score(best_ranking_metrics)
        (output_dir / "ranking_metrics.json").write_text(json.dumps(best_ranking_metrics, indent=2), encoding="utf-8")
    if best_switch_metrics is not None:
        best_switch_metrics = dict(best_switch_metrics)
        best_switch_metrics["best_epoch"] = float(best_switch_epoch)
        best_switch_metrics["ranking_score"] = _ranking_score(best_switch_metrics)
        best_switch_metrics["switch_selection_score"] = _switch_selection_score(best_switch_metrics)
        best_switch_metrics["selection_score"] = _selection_score(best_switch_metrics)
        best_switch_metrics["gold_policy_score"] = _gold_policy_score(best_switch_metrics)
        (output_dir / "switch_metrics.json").write_text(json.dumps(best_switch_metrics, indent=2), encoding="utf-8")

    _write_model_artifact_manifest(
        output_dir,
        best_metrics=best_metrics,
        final_metrics=final_metrics,
        best_ranking_metrics=best_ranking_metrics,
        best_switch_metrics=best_switch_metrics,
    )
    return best_metrics
