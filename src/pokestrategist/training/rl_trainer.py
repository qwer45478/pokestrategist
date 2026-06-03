"""Monte-Carlo Q-learning trainer with Win Head and unified Q-value function.

This module replaces the 12-term linear scoring with a single Q(s,a) head
trained on replay final outcomes. All existing evaluation metrics are preserved
as diagnostics only — they are never used as training targets.
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
class MCTrainerConfig:
    """Configuration for Monte-Carlo Q-learning training."""

    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-4
    device: str = "cpu"
    train_num_workers: int | None = None
    eval_num_workers: int | None = None
    pin_memory: bool | None = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    enable_tf32: bool = True
    split_seed: int = 0
    hidden_candidate_topk: int | None = None
    max_grad_norm: float | None = 1.0
    warmup_epochs: int = 0
    min_learning_rate_ratio: float = 0.2
    # MC-Q specific
    q_weight: float = 0.2
    v_weight: float = 0.1
    aux_weight: float = 0.8
    gold_q_weight: float = 3.0  # weight for gold action Q target vs non-gold
    use_team_preview_prior: bool | None = None
    use_usage_priors: bool | None = None
    usage_data_dir: str | None = None


def _checkpoint_model_config_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config_payload = dict(checkpoint["model_config"])
    if "use_static_rule_features" not in model_config_payload:
        model_config_payload["use_static_rule_features"] = False
    if "candidate_prior_feature_dim" not in model_config_payload:
        prior_weight = checkpoint["model_state"].get("prior_feature_proj.0.weight")
        model_config_payload["candidate_prior_feature_dim"] = (
            int(prior_weight.shape[1]) if prior_weight is not None else 2
        )
    if "preview_prior_feature_dim" not in model_config_payload:
        preview_weight = checkpoint["model_state"].get("preview_prior_mlp.0.weight")
        from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM
        model_config_payload["preview_prior_feature_dim"] = (
            int(preview_weight.shape[1]) if preview_weight is not None else TEAM_PREVIEW_FEATURE_DIM
        )
    return model_config_payload


def _load_bundle(path: str | Path) -> dict[str, Any]:
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


def _mc_q_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    config: MCTrainerConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Monte-Carlo Q-learning with pairwise ranking for action discrimination.

    Gold Q target = final_outcome (MSE anchor).
    Non-gold Q constraint = softplus(q_non_gold - q_gold) to push
    gold action above alternatives (pairwise ranking loss).
    """
    mask = batch["legal_action_mask"].bool()
    q_values = outputs["q_values"]  # [B, N], sigmoid-bounded to (0,1)
    v_state = outputs["win_logit"]  # [B, 1]
    target = batch["legal_target"]
    valid = target >= 0

    if not valid.any():
        return q_values.new_zeros(()), {"mc_q_loss": 0.0, "mc_v_loss": 0.0, "mc_q_top1": 0.0}

    # Outcome target
    if "final_outcome" in batch:
        outcome = batch["final_outcome"].float()
    else:
        outcome = outputs.get("state_consequence_mean", q_values.new_zeros(q_values.size(0)))
        if outcome.dim() > 1:
            outcome = outcome[:, -1]

    q_valid = q_values[valid]  # [V, N]
    m_valid = mask[valid]  # [V, N]
    t_valid = target[valid]  # [V]
    o_valid = outcome[valid]  # [V]

    # Gold Q → anchored to replay outcome via MSE
    gold_q = q_valid.gather(1, t_valid.unsqueeze(1)).squeeze(1)  # [V]
    gold_loss = F.mse_loss(gold_q, o_valid)

    # Pairwise ranking: non-gold Q should be below gold Q
    non_gold_mask = m_valid.clone()
    non_gold_mask.scatter_(1, t_valid.unsqueeze(1), False)
    if non_gold_mask.any():
        gold_q_expanded = gold_q.unsqueeze(1).expand_as(q_valid)
        # For each non-gold action: penalty = softplus(q_non_gold - q_gold)
        ranking_losses = F.softplus(q_valid - gold_q_expanded)  # [V, N]
        ranking_loss = (ranking_losses * non_gold_mask.float()).sum() / non_gold_mask.float().sum().clamp_min(1.0)
    else:
        ranking_loss = q_valid.new_zeros(())

    q_loss = gold_loss + ranking_loss

    # V loss: predict outcome from state
    v_loss = F.mse_loss(v_state[valid].squeeze(-1), o_valid)

    total = config.q_weight * q_loss + config.v_weight * v_loss

    # Diagnostics
    with torch.no_grad():
        # Use only legal actions for argmax
        q_masked = q_valid.masked_fill(~m_valid, -1e9)
        q_pred_top1 = q_masked.argmax(dim=1)
        q_top1_acc = q_pred_top1.eq(t_valid).float().mean().item()
        v_sign_acc = (v_state[valid].squeeze(-1).sign().eq(o_valid.sign())).float().mean().item()
        # Average Q over legal actions only
        legal_q = q_valid[m_valid]
        avg_q = legal_q.mean().item() if legal_q.numel() > 0 else 0.0

    return total, {
        "mc_q_loss": float(q_loss.detach().cpu()),
        "mc_v_loss": float(v_loss.detach().cpu()),
        "mc_gold_loss": float(gold_loss.detach().cpu()),
        "mc_ranking_loss": float(ranking_loss.detach().cpu()),
        "mc_q_top1": float(q_top1_acc),
        "mc_v_sign_acc": float(v_sign_acc),
        "mc_avg_q": float(avg_q),
        "mc_avg_v": float(v_state[valid].mean().detach().cpu()),
    }


def train_mc_q_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    mc_config: MCTrainerConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    mc_config = mc_config or MCTrainerConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_bundle(checkpoint_path)
    actor_trainer_config = dict(bundle["trainer_config"])

    resolved_hidden_candidate_topk = int(
        mc_config.hidden_candidate_topk
        if mc_config.hidden_candidate_topk is not None
        else actor_trainer_config.get("hidden_candidate_topk", 4)
    )
    resolved_use_usage_priors = (
        bool(mc_config.use_usage_priors)
        if mc_config.use_usage_priors is not None
        else bool(actor_trainer_config.get("use_usage_priors", True))
    )
    resolved_use_team_preview_prior = (
        bool(mc_config.use_team_preview_prior)
        if mc_config.use_team_preview_prior is not None
        else bundle["team_preview_prior"] is not None
    )
    resolved_usage_data_dir = (
        mc_config.usage_data_dir if mc_config.usage_data_dir is not None else actor_trainer_config.get("usage_data_dir")
    )
    hidden_move_prior = bundle["hidden_move_prior"]
    team_preview_prior = bundle["team_preview_prior"] if resolved_use_team_preview_prior else None

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
    train_indices, val_indices = grouped_replay_split(dataset.replay_ids, seed=mc_config.split_seed)
    if not train_indices and val_indices:
        train_indices = [val_indices[0]]
        val_indices = val_indices[1:] or list(train_indices)
    if not val_indices and train_indices:
        val_indices = [train_indices[0]]
        train_indices = train_indices[1:] or list(val_indices)
    print(f"Loaded dataset with {len(dataset)} samples; train={len(train_indices)} val={len(val_indices)}.", flush=True)

    device = torch.device(mc_config.device)
    _configure_torch_runtime(device, enable_tf32=mc_config.enable_tf32)
    train_num_workers = _resolve_num_workers(mc_config.train_num_workers, device=device)
    eval_num_workers = _resolve_num_workers(mc_config.eval_num_workers, device=device, fallback=0)
    pin_memory = _resolve_pin_memory(mc_config.pin_memory, device=device)
    print(f"Using device={device}, train_workers={train_num_workers}, eval_workers={eval_num_workers}, pin_memory={pin_memory}.", flush=True)

    train_loader = _build_loader(
        Subset(dataset, train_indices),
        batch_size=mc_config.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=mc_config.prefetch_factor,
        persistent_workers=mc_config.persistent_workers,
    )
    val_loader = _build_loader(
        Subset(dataset, val_indices),
        batch_size=mc_config.batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=mc_config.prefetch_factor,
        persistent_workers=mc_config.persistent_workers,
    )

    model = bundle["model"]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mc_config.learning_rate)
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
    checkpoint_trainer_config.update(
        {
            "hidden_candidate_topk": resolved_hidden_candidate_topk,
            "use_usage_priors": resolved_use_usage_priors,
            "use_team_preview_prior": resolved_use_team_preview_prior,
            "usage_data_dir": resolved_usage_data_dir,
        }
    )

    def _checkpoint_payload(metrics: dict[str, float], *, epoch: int) -> dict[str, Any]:
        payload_metrics = dict(metrics)
        payload_metrics["ranking_score"] = _ranking_score(metrics)
        payload_metrics["switch_selection_score"] = _switch_selection_score(metrics)
        payload_metrics["selection_score"] = _selection_score(metrics)
        payload_metrics["best_epoch"] = float(epoch)
        return {
            "model_state": model.state_dict(),
            "model_config": asdict(bundle["model_config"]),
            "trainer_config": checkpoint_trainer_config,
            "mc_config": asdict(mc_config),
            "metrics": payload_metrics,
            "hidden_move_prior": hidden_move_prior.to_dict() if hidden_move_prior is not None else None,
            "team_preview_prior": team_preview_prior.to_dict() if team_preview_prior is not None else None,
            "initial_checkpoint": str(bundle["checkpoint_path"]),
        }

    for epoch in range(mc_config.epochs):
        model.train()
        epoch_stats: dict[str, list[float]] = {
            "mc_q_loss": [],
            "mc_v_loss": [],
            "mc_gold_loss": [],
            "mc_ranking_loss": [],
            "mc_q_top1": [],
            "mc_v_sign_acc": [],
            "mc_avg_q": [],
            "mc_avg_v": [],
        }
        current_lr = _epoch_learning_rate(
            base_learning_rate=mc_config.learning_rate,
            epoch_index=epoch,
            total_epochs=mc_config.epochs,
            warmup_epochs=mc_config.warmup_epochs,
            min_learning_rate_ratio=mc_config.min_learning_rate_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f"Starting MC-Q epoch {epoch + 1}/{mc_config.epochs} (lr={current_lr:.2g})...", flush=True)

        for batch in train_loader:
            tensor_batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            outputs = model(tensor_batch)
            q_loss_total, q_stats = _mc_q_loss(outputs, tensor_batch, mc_config)

            # Light auxiliary loss for regularization of the shared encoder
            aux_loss = _light_aux_loss(outputs, tensor_batch)
            total_loss = q_loss_total + mc_config.aux_weight * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            if mc_config.max_grad_norm is not None and mc_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), mc_config.max_grad_norm)
            optimizer.step()

            for key, value in q_stats.items():
                epoch_stats[key].append(value)
            epoch_stats.setdefault("mc_aux_loss", []).append(float(aux_loss.detach().cpu()))

        metrics = evaluate_model(model, val_loader, device)
        metrics.update(
            {key: (sum(values) / len(values) if values else 0.0) for key, values in epoch_stats.items()}
        )
        metrics["epoch"] = float(epoch + 1)
        metrics["learning_rate"] = current_lr
        metrics["ranking_score"] = _ranking_score(metrics)
        metrics["switch_selection_score"] = _switch_selection_score(metrics)
        metrics["selection_score"] = _selection_score(metrics)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
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
            torch.save(_checkpoint_payload(best_ranking_metrics, epoch=best_ranking_epoch), output_dir / "best_ranking_model.pt")
            (output_dir / "ranking_metrics.json").write_text(json.dumps(best_ranking_metrics, indent=2), encoding="utf-8")

        if metrics["switch_selection_score"] > best_switch_score:
            best_switch_score = metrics["switch_selection_score"]
            best_switch_metrics = dict(metrics)
            best_switch_epoch = epoch + 1
            torch.save(_checkpoint_payload(best_switch_metrics, epoch=best_switch_epoch), output_dir / "best_switch_model.pt")
            (output_dir / "switch_metrics.json").write_text(json.dumps(best_switch_metrics, indent=2), encoding="utf-8")

    final_metrics = history[-1] if history else evaluate_model(model, val_loader, device)
    final_metrics["ranking_score"] = _ranking_score(final_metrics)
    final_metrics["switch_selection_score"] = _switch_selection_score(final_metrics)
    final_metrics["selection_score"] = _selection_score(final_metrics)
    best_metrics_final = best_metrics or final_metrics

    final_checkpoint = _checkpoint_payload(final_metrics, epoch=mc_config.epochs)
    torch.save(final_checkpoint, output_dir / "final_model.pt")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "hidden_move_prior.json").write_text(
        json.dumps(hidden_move_prior.to_dict() if hidden_move_prior is not None else None, indent=2),
        encoding="utf-8",
    )
    if team_preview_prior is not None:
        (output_dir / "team_preview_prior.json").write_text(
            json.dumps(team_preview_prior.to_dict(), indent=2), encoding="utf-8",
        )

    if best_epoch:
        best_metrics = dict(best_metrics)
        best_metrics["best_epoch"] = float(best_epoch)
        best_metrics["ranking_score"] = _ranking_score(best_metrics)
        best_metrics["switch_selection_score"] = _switch_selection_score(best_metrics)
        best_metrics["selection_score"] = _selection_score(best_metrics)
        (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    if best_ranking_metrics is not None:
        best_ranking_metrics = dict(best_ranking_metrics)
        best_ranking_metrics["best_epoch"] = float(best_ranking_epoch)
        best_ranking_metrics["ranking_score"] = _ranking_score(best_ranking_metrics)
        best_ranking_metrics["switch_selection_score"] = _switch_selection_score(best_ranking_metrics)
        best_ranking_metrics["selection_score"] = _selection_score(best_ranking_metrics)
        (output_dir / "ranking_metrics.json").write_text(json.dumps(best_ranking_metrics, indent=2), encoding="utf-8")
    if best_switch_metrics is not None:
        best_switch_metrics = dict(best_switch_metrics)
        best_switch_metrics["best_epoch"] = float(best_switch_epoch)
        best_switch_metrics["ranking_score"] = _ranking_score(best_switch_metrics)
        best_switch_metrics["switch_selection_score"] = _switch_selection_score(best_switch_metrics)
        best_switch_metrics["selection_score"] = _selection_score(best_switch_metrics)
        (output_dir / "switch_metrics.json").write_text(json.dumps(best_switch_metrics, indent=2), encoding="utf-8")

    _write_model_artifact_manifest(
        output_dir,
        best_metrics=best_metrics_final,
        final_metrics=final_metrics,
        best_ranking_metrics=best_ranking_metrics,
        best_switch_metrics=best_switch_metrics,
    )
    return best_metrics_final


def _light_aux_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    """Minimal auxiliary losses to keep the encoder from drifting.

    Only the most stable supervision heads are kept at very low weight.
    """
    loss = outputs.get("plan_logits", outputs.get("state", None))
    if loss is None:
        return torch.tensor(0.0)

    # Only keep plan, phase, and line — the three most stable aux heads
    total = torch.tensor(0.0)
    count = 0

    if "plan_logits" in outputs and "plan_target" in batch:
        plan_valid = batch["plan_target"] >= 0
        if plan_valid.any():
            total = total + F.cross_entropy(outputs["plan_logits"][plan_valid], batch["plan_target"][plan_valid])
            count += 1

    if "line_logits" in outputs and "line_target" in batch:
        line_valid = batch["line_target"] >= 0
        if line_valid.any():
            total = total + F.cross_entropy(outputs["line_logits"][line_valid], batch["line_target"][line_valid])
            count += 1

    if "phase_logits" in outputs and "phase_target" in batch:
        phase_valid = batch["phase_target"] >= 0
        if phase_valid.any():
            total = total + F.cross_entropy(outputs["phase_logits"][phase_valid], batch["phase_target"][phase_valid])
            count += 1

    if count == 0:
        return torch.tensor(0.0)
    return total / count
