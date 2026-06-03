"""Training loop, losses, and evaluation for the decision-assist v3 model."""

from __future__ import annotations

import json
import math
import os
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from pokestrategist.data.hidden_move_prior import (
    HiddenMovePrior,
    _context_key,
    _counter_to_candidates,
    _counter_to_family_defaults,
    _item_context_key,
)
from pokestrategist.data.schema import ActionHead, DecisionSample, SideSummary
from pokestrategist.data.static_rules import normalize_name
from pokestrategist.data.team_preview_prior import (
    PreviewSetObservation,
    TeamPreviewExample,
    TeamPreviewPriorCatalog,
    _AggregatedPreviewState,
    _ensure_species,
    _merge_opponent_action,
    _unique_preserve_order,
    build_team_preview_prior,
)
from pokestrategist.models.decision_model import PokeStrategistDecisionConfig, PokeStrategistDecisionModel
from pokestrategist.training.dataset import DecisionTensorDataset, collate_decision_batch, grouped_replay_split

RESOURCE_VALUE_WEIGHTS = (0.25, 0.15, 0.15, 0.20, 0.10, 0.15)
TRAIN_PRIOR_CACHE_VERSION = 2


def _write_model_artifact_manifest(
    output_dir: Path,
    *,
    best_metrics: dict[str, float] | None,
    final_metrics: dict[str, float],
    best_ranking_metrics: dict[str, float] | None,
    best_switch_metrics: dict[str, float] | None,
) -> None:
    manifest = {
        "best_model": {
            "path": "best_model.pt",
            "alias_path": "model.pt",
            "metrics_path": "metrics.json",
            "metrics": best_metrics,
        },
        "best_ranking_model": {
            "path": "best_ranking_model.pt",
            "metrics_path": "ranking_metrics.json",
            "metrics": best_ranking_metrics,
        },
        "best_switch_model": {
            "path": "best_switch_model.pt",
            "metrics_path": "switch_metrics.json",
            "metrics": best_switch_metrics,
        },
        "final_model": {
            "path": "final_model.pt",
            "metrics_path": "final_metrics.json",
            "metrics": final_metrics,
        },
    }
    (output_dir / "model_artifacts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str = "cpu"
    train_num_workers: int | None = None
    eval_num_workers: int | None = None
    pin_memory: bool | None = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    enable_tf32: bool = True
    split_seed: int = 0
    hidden_candidate_topk: int = 4
    preference_weight: float = 0.5
    recoverable_bc_weight: float = 0.25
    opponent_weight: float = 0.4
    plan_weight: float = 0.15
    belief_weight: float = 0.15
    resource_weight: float = 0.15
    phase_weight: float = 0.15
    line_weight: float = 0.2
    unlock_weight: float = 0.15
    state_future_weight: float = 0.5
    local_future_weight: float = 0.2
    particle_weight: float = 0.15
    reveal_weight: float = 0.1
    typed_consequence_weight: float = 0.15
    frontier_weight: float = 0.1
    candidate_response_weight: float = 0.2
    switch_follow_weight: float = 0.15
    head_weight: float = 0.35
    switch_ranking_weight: float = 0.5
    switch_sample_weight: float = 2.0
    action_score_ce_weight: float = 0.35
    route_consistency_weight: float = 0.0
    consequence_value_weight: float = 0.0
    consequence_value_margin_weight: float = 0.0
    consequence_value_margin: float = 0.05
    consequence_value_negative_topk: int = 4
    max_grad_norm: float | None = 1.0
    warmup_epochs: int = 1
    min_learning_rate_ratio: float = 0.2
    use_team_preview_prior: bool = True
    use_usage_priors: bool = True
    usage_data_dir: str | None = None


def _clean_name(value: str | None) -> str:
    return (value or "").strip()


def _resolve_num_workers(requested: int | None, *, device: torch.device, fallback: int | None = None) -> int:
    if requested is not None:
        return max(0, int(requested))
    if fallback is not None:
        return max(0, int(fallback))
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, max(1, cpu_count - 2)))


def _resolve_pin_memory(requested: bool | None, *, device: torch.device) -> bool:
    if requested is not None:
        return bool(requested) and device.type == "cuda"
    return device.type == "cuda"


def _configure_torch_runtime(device: torch.device, *, enable_tf32: bool) -> None:
    if device.type != "cuda":
        return
    if enable_tf32:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def _epoch_learning_rate(
    *,
    base_learning_rate: float,
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int,
    min_learning_rate_ratio: float,
) -> float:
    if total_epochs <= 1:
        return base_learning_rate
    effective_ratio = min(max(float(min_learning_rate_ratio), 0.0), 1.0)
    effective_warmup = max(0, int(warmup_epochs))
    if effective_warmup > 0 and epoch_index < effective_warmup:
        warmup_scale = float(epoch_index + 1) / float(effective_warmup)
        return base_learning_rate * warmup_scale
    if total_epochs <= effective_warmup + 1:
        return base_learning_rate
    cosine_progress = float(epoch_index - effective_warmup) / float(max(1, total_epochs - effective_warmup - 1))
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * min(max(cosine_progress, 0.0), 1.0)))
    return base_learning_rate * (effective_ratio + (1.0 - effective_ratio) * cosine_scale)


def _build_loader(
    dataset: Subset[DecisionTensorDataset] | DecisionTensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_decision_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **loader_kwargs)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device, *, non_blocking: bool) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _merge_preview_side_summary(state: _AggregatedPreviewState, side_summary: SideSummary) -> None:
    for species in side_summary.team_order:
        _ensure_species(state, species)
    for species, moves in side_summary.revealed_moves.items():
        record = _ensure_species(state, species)
        if record is not None:
            record.moves.update(_clean_name(move) for move in moves if _clean_name(move))
    for species, item_name in side_summary.revealed_items.items():
        record = _ensure_species(state, species)
        if record is not None and _clean_name(item_name):
            record.item = _clean_name(item_name)
    for species, ability_name in side_summary.revealed_abilities.items():
        record = _ensure_species(state, species)
        if record is not None and _clean_name(ability_name):
            record.ability = _clean_name(ability_name)
    for species, tera_type in side_summary.revealed_tera_types.items():
        record = _ensure_species(state, species)
        if record is not None and _clean_name(tera_type):
            record.tera_type = _clean_name(tera_type)


def _iter_team_preview_examples_from_states(
    grouped_states: dict[tuple[str, str], _AggregatedPreviewState],
    *,
    min_team_size: int = 2,
    min_evidence_fields: int = 1,
) -> Iterator[TeamPreviewExample]:
    for state in grouped_states.values():
        team_species = _unique_preserve_order(state.team_species)
        if len(team_species) < min_team_size:
            continue
        set_observations: list[PreviewSetObservation] = []
        for species in team_species:
            record = state.observations.get(normalize_name(species))
            if record is None:
                continue
            observation = PreviewSetObservation(
                species=record.species,
                moves=_unique_preserve_order(sorted(record.moves, key=normalize_name)),
                item=record.item,
                ability=record.ability,
                tera_type=record.tera_type,
                spread_bucket=record.spread_bucket,
            )
            if observation.evidence_count() >= min_evidence_fields:
                set_observations.append(observation)
        if set_observations:
            yield TeamPreviewExample(team_species=team_species, set_observations=tuple(set_observations))


def _build_train_priors(
    samples: Iterable[DecisionSample],
    *,
    default_topk: int,
    use_team_preview_prior: bool,
) -> tuple[HiddenMovePrior, TeamPreviewPriorCatalog | None]:
    exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    item_counts: dict[str, Counter[str]] = defaultdict(Counter)
    item_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    species_counts: dict[str, Counter[str]] = defaultdict(Counter)
    species_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    species_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_family_counts: Counter[str] = Counter()
    grouped_preview_states: dict[tuple[str, str], _AggregatedPreviewState] | None = {} if use_team_preview_prior else None

    for sample in samples:
        if grouped_preview_states is not None:
            key = (sample.replay_id, sample.perspective.value)
            state = grouped_preview_states.setdefault(key, _AggregatedPreviewState())
            _merge_preview_side_summary(state, sample.observation.opp_side)
            _merge_opponent_action(state, sample)

        active = sample.observation.self_side.active_species
        action = sample.our_action
        if action.head not in {ActionHead.MOVE, ActionHead.TERA_MOVE}:
            continue
        if not active or not action.move_token:
            continue
        revealed_moves = tuple(sorted(sample.observation.self_side.revealed_moves.get(active, [])))
        if action.move_token in revealed_moves:
            continue
        family = action.move_family or "utility"
        context_key = _context_key(active, revealed_moves)
        exact_counts[context_key][action.move_token] += 1
        exact_move_families[context_key][action.move_token] = family
        item = sample.observation.self_side.revealed_items.get(active)
        if item:
            item_key = _item_context_key(active, item)
            item_counts[item_key][action.move_token] += 1
            item_move_families[item_key][action.move_token] = family
        species_counts[active][action.move_token] += 1
        species_move_families[active][action.move_token] = family
        species_family_counts[active][family] += 1
        global_family_counts[family] += 1

    hidden_move_prior = HiddenMovePrior(
        exact_contexts={key: _counter_to_candidates(counter, exact_move_families[key]) for key, counter in exact_counts.items()},
        item_contexts={key: _counter_to_candidates(counter, item_move_families[key]) for key, counter in item_counts.items()},
        species_defaults={key: _counter_to_candidates(counter, species_move_families[key]) for key, counter in species_counts.items()},
        species_family_defaults={key: _counter_to_family_defaults(counter) for key, counter in species_family_counts.items()},
        global_family_defaults=_counter_to_family_defaults(global_family_counts),
        default_topk=default_topk,
    )
    team_preview_prior = None
    if grouped_preview_states is not None:
        team_preview_prior = build_team_preview_prior(_iter_team_preview_examples_from_states(grouped_preview_states))
    return hidden_move_prior, team_preview_prior


def _file_content_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _train_prior_logic_fingerprint() -> str:
    trainer_file = Path(__file__).resolve()
    team_preview_file = Path(build_team_preview_prior.__code__.co_filename).resolve()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_file_content_digest(trainer_file).encode("utf-8"))
    digest.update(_file_content_digest(team_preview_file).encode("utf-8"))
    return digest.hexdigest()


def _train_prior_cache_metadata(
    dataset: DecisionTensorDataset,
    *,
    trainer_config: TrainerConfig,
    max_samples: int | None,
) -> dict[str, object]:
    metadata = {
        "cache_version": TRAIN_PRIOR_CACHE_VERSION,
        "dataset_path": str(dataset.path.resolve()),
        "dataset_content_fingerprint": dataset.content_fingerprint,
        "sample_count": int(len(dataset)),
        "split_seed": int(trainer_config.split_seed),
        "hidden_candidate_topk": int(trainer_config.hidden_candidate_topk),
        "use_team_preview_prior": bool(trainer_config.use_team_preview_prior),
        "max_samples": None if max_samples is None else int(max_samples),
        "prior_logic_fingerprint": _train_prior_logic_fingerprint(),
    }
    digest = hashlib.blake2b(digest_size=16)
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    metadata["cache_key"] = digest.hexdigest()
    return metadata


def _train_prior_cache_paths(dataset_path: Path, cache_key: str) -> dict[str, Path]:
    cache_dir = dataset_path.parent / ".pokestrategist_train_prior_cache" / cache_key
    return {
        "dir": cache_dir,
        "meta": cache_dir / "metadata.json",
        "hidden": cache_dir / "hidden_move_prior.json",
        "team_preview": cache_dir / "team_preview_prior.json",
    }


def _load_train_prior_cache(
    dataset_path: Path,
    metadata: dict[str, object],
) -> tuple[HiddenMovePrior, TeamPreviewPriorCatalog | None] | None:
    paths = _train_prior_cache_paths(dataset_path, str(metadata["cache_key"]))
    if not paths["meta"].exists() or not paths["hidden"].exists():
        return None
    try:
        cached_metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if cached_metadata != metadata:
            return None
        hidden_move_prior = HiddenMovePrior.from_dict(json.loads(paths["hidden"].read_text(encoding="utf-8")))
        if hidden_move_prior is None:
            return None
        team_preview_prior = None
        if metadata["use_team_preview_prior"]:
            if not paths["team_preview"].exists():
                return None
            team_preview_prior = TeamPreviewPriorCatalog.from_dict(
                json.loads(paths["team_preview"].read_text(encoding="utf-8"))
            )
        return hidden_move_prior, team_preview_prior
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _save_train_prior_cache(
    dataset_path: Path,
    metadata: dict[str, object],
    *,
    hidden_move_prior: HiddenMovePrior,
    team_preview_prior: TeamPreviewPriorCatalog | None,
) -> Path:
    paths = _train_prior_cache_paths(dataset_path, str(metadata["cache_key"]))
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["hidden"].write_text(json.dumps(hidden_move_prior.to_dict(), indent=2), encoding="utf-8")
    if team_preview_prior is not None:
        paths["team_preview"].write_text(json.dumps(team_preview_prior.to_dict(), indent=2), encoding="utf-8")
    elif paths["team_preview"].exists():
        paths["team_preview"].unlink()
    paths["meta"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths["dir"]


def _zero_loss(outputs: dict[str, Tensor]) -> Tensor:
    for value in outputs.values():
        if isinstance(value, Tensor):
            return value.new_zeros(())
    return torch.tensor(0.0)


def _masked_action_ce(logits: Tensor, mask: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
    valid = target >= 0
    if not valid.any():
        return logits.new_zeros(())
    masked_logits = logits.masked_fill(~mask.bool(), -1e9)
    losses = F.cross_entropy(masked_logits[valid], target[valid], reduction="none")
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight[valid]
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def _recoverable_bc_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        return outputs["recoverable_support_logits"].new_zeros(())
    valid_target = target[valid]
    valid_recoverable_flags = batch["legal_action_recoverable_flags"][valid]
    gold_recoverable = valid_recoverable_flags.gather(1, valid_target.unsqueeze(1)).squeeze(1)
    recoverable_rows = gold_recoverable.bool()
    if not recoverable_rows.any():
        return outputs["recoverable_support_logits"].new_zeros(())
    logits = outputs["recoverable_support_logits"][valid][recoverable_rows]
    mask = valid_recoverable_flags[recoverable_rows]
    return _masked_action_ce(logits, mask, valid_target[recoverable_rows])


def _preference_loss(scores: Tensor, mask: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
    valid = target >= 0
    if not valid.any():
        return scores.new_zeros(())
    masked_scores = scores[valid].masked_fill(~mask[valid].bool(), -1e9)
    valid_target = target[valid]
    gold_scores = masked_scores.gather(1, valid_target.unsqueeze(1)).squeeze(1)
    negative_mask = mask[valid].bool().clone()
    negative_mask.scatter_(1, valid_target.unsqueeze(1), False)
    if not negative_mask.any():
        return scores.new_zeros(())
    pairwise = F.softplus(masked_scores - gold_scores.unsqueeze(1))
    sample_losses: list[Tensor] = []
    sample_indices: list[int] = []
    for row_index in range(pairwise.size(0)):
        row_mask = negative_mask[row_index]
        if not row_mask.any():
            continue
        sample_losses.append(pairwise[row_index][row_mask].mean())
        sample_indices.append(row_index)
    if not sample_losses:
        return scores.new_zeros(())
    loss_tensor = torch.stack(sample_losses)
    if sample_weight is None:
        return loss_tensor.mean()
    weight_index = torch.tensor(sample_indices, device=scores.device)
    weights = sample_weight[valid][weight_index]
    return (loss_tensor * weights).sum() / weights.sum().clamp_min(1e-6)


def _soft_cross_entropy(logits: Tensor, target_probs: Tensor) -> Tensor:
    return -(target_probs * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def _resource_value(resource_ledger: Tensor) -> Tensor:
    weights = resource_ledger.new_tensor(RESOURCE_VALUE_WEIGHTS)
    return (resource_ledger * weights).sum(dim=-1)


def _switch_ranking_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    switch_rows = batch["gold_is_switch"].bool() & batch["supervised_target"].bool()
    if not switch_rows.any():
        return outputs["legal_action_scores"].new_zeros(())
    switch_scores = outputs["legal_action_scores"][switch_rows].masked_fill(~batch["switch_candidate_mask"][switch_rows].bool(), -1e9)
    return F.cross_entropy(switch_scores, batch["legal_target"][switch_rows])


def _action_score_ce_loss(outputs: dict[str, Tensor], batch: dict[str, Any], sample_weight: Tensor | None = None) -> Tensor:
    return _masked_action_ce(
        outputs["legal_action_scores"],
        batch["legal_action_mask"],
        batch["legal_target"],
        sample_weight=sample_weight,
    )


def _consequence_value_losses(outputs: dict[str, Tensor], batch: dict[str, Any], config: TrainerConfig) -> dict[str, Tensor]:
    """Calibrate consequence value and separate gold from hard non-gold candidates.

    Target = state_future_mean[:, convert_score] (index 7) — the model's own supervised
    estimate of how convertible the current position is to a win.
    """
    if "consequence_value_scores" not in outputs:
        zero = _zero_loss(outputs)
        return {
            "consequence_value_loss": zero,
            "consequence_value_mse_loss": zero,
            "consequence_value_margin_loss": zero,
            "consequence_value_gold_mean": zero,
            "consequence_value_hard_negative_mean": zero,
            "consequence_value_margin_mean": zero,
        }
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        zero = outputs["consequence_value_scores"].new_zeros(())
        return {
            "consequence_value_loss": zero,
            "consequence_value_mse_loss": zero,
            "consequence_value_margin_loss": zero,
            "consequence_value_gold_mean": zero,
            "consequence_value_hard_negative_mean": zero,
            "consequence_value_margin_mean": zero,
        }

    valid_target = target[valid]
    valid_scores = outputs["consequence_value_scores"][valid]
    gold_value_scores = valid_scores.gather(1, valid_target.unsqueeze(1)).squeeze(1)
    state_value_target = outputs["state_future_mean"][valid, -1]
    mse_loss = F.mse_loss(gold_value_scores, state_value_target.detach())

    negative_mask = batch["legal_action_mask"][valid].bool().clone()
    negative_mask.scatter_(1, valid_target.unsqueeze(1), False)
    rows_with_negative = negative_mask.any(dim=1)
    margin_loss = mse_loss.new_zeros(())
    hard_negative_mean = mse_loss.new_zeros(())
    margin_mean = mse_loss.new_zeros(())
    if rows_with_negative.any():
        row_scores = valid_scores[rows_with_negative]
        row_negative_mask = negative_mask[rows_with_negative]
        negative_scores = row_scores.masked_fill(~row_negative_mask, -1e9)
        topk = min(max(1, int(config.consequence_value_negative_topk)), negative_scores.size(1))
        top_negative_scores = negative_scores.topk(k=topk, dim=1).values
        finite_negative_mask = top_negative_scores > -1e8
        if finite_negative_mask.any():
            row_gold_scores = gold_value_scores[rows_with_negative]
            expanded_gold = row_gold_scores.unsqueeze(1).expand_as(top_negative_scores)
            margin_terms = F.relu(top_negative_scores - expanded_gold + float(config.consequence_value_margin))
            margin_loss = margin_terms[finite_negative_mask].mean()
            hard_negative_mean = top_negative_scores[finite_negative_mask].mean()
            hardest_negative_scores = negative_scores.max(dim=1).values
            margin_mean = (row_gold_scores - hardest_negative_scores).mean()

    total_loss = mse_loss + float(config.consequence_value_margin_weight) * margin_loss
    return {
        "consequence_value_loss": total_loss,
        "consequence_value_mse_loss": mse_loss,
        "consequence_value_margin_loss": margin_loss,
        "consequence_value_gold_mean": gold_value_scores.mean().detach(),
        "consequence_value_hard_negative_mean": hard_negative_mean.detach(),
        "consequence_value_margin_mean": margin_mean.detach(),
    }


def _route_consistency_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    legal_action_mask = batch["legal_action_mask"].bool()
    switch_mask = legal_action_mask & batch["legal_action_head_ids"].eq(1)
    move_mask = legal_action_mask & batch["legal_action_head_ids"].ne(1)
    target = batch["head_target"]

    valid = torch.where(target.eq(1), switch_mask.any(dim=1), move_mask.any(dim=1))
    if not valid.any():
        return outputs["legal_action_scores"].new_zeros(())

    move_logits = outputs["legal_action_scores"].masked_fill(~move_mask, -1e9).logsumexp(dim=1)
    switch_logits = outputs["legal_action_scores"].masked_fill(~switch_mask, -1e9).logsumexp(dim=1)
    route_logits = torch.stack([move_logits, switch_logits], dim=1)
    return F.cross_entropy(route_logits[valid], target[valid])


def _candidate_future_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        return outputs["candidate_future_mean"].new_zeros(())
    gather_index = target[valid].view(-1, 1, 1).expand(-1, 1, outputs["candidate_future_mean"].size(-1))
    pred_mean = outputs["candidate_future_mean"][valid].gather(1, gather_index).squeeze(1)
    pred_tail = outputs["candidate_future_tail"][valid].gather(1, gather_index).squeeze(1)
    return F.mse_loss(pred_mean, batch["future_mean_target"][valid]) + F.mse_loss(pred_tail, batch["future_tail_target"][valid])


def _candidate_response_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        return outputs["candidate_response_logits"].new_zeros(())
    gather_index = target[valid].view(-1, 1, 1).expand(-1, 1, outputs["candidate_response_logits"].size(-1))
    logits = outputs["candidate_response_logits"][valid].gather(1, gather_index).squeeze(1)
    return F.cross_entropy(logits, batch["response_target"][valid])


def _switch_follow_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    if "switch_follow_mean" not in outputs or "switch_follow_tail" not in outputs or "switch_entry_value" not in outputs:
        return _zero_loss(outputs)
    switch_rows = batch["gold_is_switch"].bool() & batch["supervised_target"].bool()
    if not switch_rows.any():
        return outputs["switch_follow_mean"].new_zeros(())
    target = batch["legal_target"][switch_rows]
    gather_index = target.view(-1, 1, 1).expand(-1, 1, outputs["switch_follow_mean"].size(-1))
    pred_mean = outputs["switch_follow_mean"][switch_rows].gather(1, gather_index).squeeze(1)
    pred_tail = outputs["switch_follow_tail"][switch_rows].gather(1, gather_index).squeeze(1)
    entry_pred = outputs["switch_entry_value"][switch_rows].gather(1, target.unsqueeze(1)).squeeze(1)
    hazard_cost = batch["legal_action_switch_hazard_costs"][switch_rows].gather(1, target.unsqueeze(1)).squeeze(1)
    entry_target = batch["future_mean_target"][switch_rows][:, 5] - hazard_cost
    return (
        F.mse_loss(pred_mean, batch["future_mean_target"][switch_rows])
        + F.mse_loss(pred_tail, batch["future_tail_target"][switch_rows])
        + F.mse_loss(entry_pred, entry_target)
    )


def _balanced_head_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    counts = torch.bincount(batch["head_target"], minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return F.cross_entropy(outputs["head_logits"], batch["head_target"], weight=weights)


def _particle_alignment_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    if "particle_posterior" not in outputs:
        return _zero_loss(outputs)
    loss = F.mse_loss(outputs["particle_posterior"], batch["particle_posterior_target"])
    if "particle_uncertainty" in outputs:
        loss = loss + F.mse_loss(outputs["particle_uncertainty"], batch["particle_uncertainty_target"])
    return loss


def _reveal_likelihood_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    if "reveal_likelihood_logits" not in outputs:
        return _zero_loss(outputs)
    return F.binary_cross_entropy_with_logits(outputs["reveal_likelihood_logits"], batch["reveal_likelihood_target"])


def _typed_consequence_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    if "typed_consequence_logits" not in outputs:
        return _zero_loss(outputs)
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        return outputs["typed_consequence_logits"].new_zeros(())
    valid_target = target[valid]
    gather_index = valid_target.view(-1, 1, 1, 1).expand(
        -1,
        1,
        outputs["typed_consequence_logits"].size(2),
        outputs["typed_consequence_logits"].size(3),
    )
    logits = outputs["typed_consequence_logits"][valid].gather(1, gather_index).squeeze(1)
    bin_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch["typed_consequence_bin_target"][valid].reshape(-1),
    )
    if "typed_consequence_interactions" not in outputs:
        return bin_loss
    interaction_index = valid_target.view(-1, 1, 1).expand(-1, 1, outputs["typed_consequence_interactions"].size(-1))
    interaction_pred = outputs["typed_consequence_interactions"][valid].gather(1, interaction_index).squeeze(1)
    interaction_loss = F.mse_loss(interaction_pred, batch["typed_consequence_interaction_target"][valid])
    return bin_loss + interaction_loss


def _frontier_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> Tensor:
    required = {"frontier_policy_logits", "frontier_mask", "adaptive_epsilon", "legal_action_scores"}
    if not required.issubset(outputs):
        return _zero_loss(outputs)
    target = batch["legal_target"]
    valid = target >= 0
    if not valid.any():
        return outputs["legal_action_scores"].new_zeros(())
    valid_target = target[valid]
    valid_scores = outputs["legal_action_scores"][valid]
    gold_scores = valid_scores.gather(1, valid_target.unsqueeze(1)).squeeze(1)
    best_scores = valid_scores.max(dim=1).values
    keep_loss = F.relu(best_scores - gold_scores - outputs["adaptive_epsilon"][valid]).mean()
    valid_frontier_mask = outputs["frontier_mask"][valid]
    gold_in_frontier = valid_frontier_mask.gather(1, valid_target.unsqueeze(1)).squeeze(1)
    if gold_in_frontier.any():
        anchor_loss = F.cross_entropy(outputs["frontier_policy_logits"][valid][gold_in_frontier], valid_target[gold_in_frontier])
    else:
        anchor_loss = keep_loss.new_zeros(())
    return keep_loss + anchor_loss


def _ranking_score(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics.get("covered_top1", 0.0)
        + 0.25 * metrics.get("covered_top3", 0.0)
        + 0.10 * metrics.get("top1_family_accuracy", 0.0)
        + 0.15 * metrics.get("gold_candidate_coverage", 0.0)
        + 0.05 * metrics.get("frontier_gold_recall", 0.0)
        - 0.05 * metrics.get("ece", 0.0)
    )


def _switch_selection_score(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics.get("switch_target_species_accuracy", 0.0)
        + 0.25 * metrics.get("switch_target_accuracy", 0.0)
        + 0.20 * metrics.get("switch_head_recall", 0.0)
        + 0.10 * metrics.get("head_routing_accuracy", 0.0)
        - 0.05 * metrics.get("ece", 0.0)
    )


def _selection_score(metrics: dict[str, float]) -> float:
    return 0.75 * _ranking_score(metrics) + 0.25 * _switch_selection_score(metrics)


def _compute_losses(batch: dict[str, Any], outputs: dict[str, Tensor], config: TrainerConfig) -> tuple[Tensor, dict[str, float]]:
    switch_weights = torch.where(batch["gold_is_switch"], config.switch_sample_weight, 1.0).to(outputs["imitation_logits"].dtype)
    bc_loss = _masked_action_ce(outputs["imitation_logits"], batch["legal_action_mask"], batch["legal_target"], sample_weight=switch_weights)
    recoverable_bc_loss = _recoverable_bc_loss(outputs, batch)
    preference_loss = _preference_loss(outputs["legal_action_scores"], batch["legal_action_mask"], batch["legal_target"], sample_weight=switch_weights)
    action_score_ce_loss = _action_score_ce_loss(outputs, batch, sample_weight=switch_weights)
    head_loss = _balanced_head_loss(outputs, batch)
    route_consistency_loss = _route_consistency_loss(outputs, batch)
    switch_ranking_loss = _switch_ranking_loss(outputs, batch)
    line_loss = F.cross_entropy(outputs["line_logits"], batch["line_target"])

    plan_loss = F.cross_entropy(outputs["plan_logits"], batch["plan_target"]) + _soft_cross_entropy(
        outputs["plan_logits"], batch["plan_posterior_target"]
    )
    phase_loss = F.cross_entropy(outputs["phase_logits"], batch["phase_target"]) + _soft_cross_entropy(
        outputs["phase_logits"], batch["phase_posterior_target"]
    )
    unlock_loss = F.binary_cross_entropy_with_logits(outputs["unlock_logit"], batch["unlock_target"])
    belief_loss = F.mse_loss(outputs["belief_summary"], batch["belief_summary_target"]) + F.mse_loss(
        outputs["belief_uncertainty"],
        batch["belief_uncertainty_target"],
    )
    resource_loss = F.mse_loss(outputs["resource_ledger"], batch["resource_ledger_target"]) + F.mse_loss(
        _resource_value(outputs["resource_ledger"]),
        batch["resource_value_target"],
    )
    opponent_loss = F.cross_entropy(outputs["response_logits"], batch["response_target"])
    candidate_response_loss = _candidate_response_loss(outputs, batch)
    state_future_loss = F.mse_loss(outputs["state_future_mean"], batch["future_mean_target"]) + F.mse_loss(
        outputs["state_future_tail"],
        batch["future_tail_target"],
    )
    candidate_future_loss = _candidate_future_loss(outputs, batch)
    switch_follow_loss = _switch_follow_loss(outputs, batch)
    particle_loss = _particle_alignment_loss(outputs, batch)
    reveal_loss = _reveal_likelihood_loss(outputs, batch)
    typed_consequence_loss = _typed_consequence_loss(outputs, batch)
    consequence_value_losses = _consequence_value_losses(outputs, batch, config)
    consequence_value_loss = consequence_value_losses["consequence_value_loss"]
    frontier_loss = _frontier_loss(outputs, batch)

    total = (
        bc_loss
        + config.recoverable_bc_weight * recoverable_bc_loss
        + config.preference_weight * preference_loss
        + config.action_score_ce_weight * action_score_ce_loss
        + config.head_weight * head_loss
        + config.route_consistency_weight * route_consistency_loss
        + config.switch_ranking_weight * switch_ranking_loss
        + config.line_weight * line_loss
        + config.opponent_weight * opponent_loss
        + config.candidate_response_weight * candidate_response_loss
        + config.plan_weight * plan_loss
        + config.belief_weight * belief_loss
        + config.resource_weight * resource_loss
        + config.phase_weight * phase_loss
        + config.unlock_weight * unlock_loss
        + config.state_future_weight * state_future_loss
        + config.local_future_weight * candidate_future_loss
        + config.switch_follow_weight * switch_follow_loss
        + config.particle_weight * particle_loss
        + config.reveal_weight * reveal_loss
        + config.typed_consequence_weight * typed_consequence_loss
        + config.consequence_value_weight * consequence_value_loss
        + config.frontier_weight * frontier_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "recoverable_bc_loss": float(recoverable_bc_loss.detach().cpu()),
        "preference_loss": float(preference_loss.detach().cpu()),
        "action_score_ce_loss": float(action_score_ce_loss.detach().cpu()),
        "head_loss": float(head_loss.detach().cpu()),
        "route_consistency_loss": float(route_consistency_loss.detach().cpu()),
        "switch_ranking_loss": float(switch_ranking_loss.detach().cpu()),
        "line_loss": float(line_loss.detach().cpu()),
        "plan_loss": float(plan_loss.detach().cpu()),
        "phase_loss": float(phase_loss.detach().cpu()),
        "belief_loss": float(belief_loss.detach().cpu()),
        "resource_loss": float(resource_loss.detach().cpu()),
        "opponent_loss": float(opponent_loss.detach().cpu()),
        "candidate_response_loss": float(candidate_response_loss.detach().cpu()),
        "state_future_loss": float(state_future_loss.detach().cpu()),
        "candidate_future_loss": float(candidate_future_loss.detach().cpu()),
        "switch_follow_loss": float(switch_follow_loss.detach().cpu()),
        "particle_loss": float(particle_loss.detach().cpu()),
        "reveal_loss": float(reveal_loss.detach().cpu()),
        "typed_consequence_loss": float(typed_consequence_loss.detach().cpu()),
        "consequence_value_loss": float(consequence_value_loss.detach().cpu()),
        "consequence_value_mse_loss": float(consequence_value_losses["consequence_value_mse_loss"].detach().cpu()),
        "consequence_value_margin_loss": float(consequence_value_losses["consequence_value_margin_loss"].detach().cpu()),
        "consequence_value_gold_mean": float(consequence_value_losses["consequence_value_gold_mean"].detach().cpu()),
        "consequence_value_hard_negative_mean": float(
            consequence_value_losses["consequence_value_hard_negative_mean"].detach().cpu()
        ),
        "consequence_value_margin_mean": float(consequence_value_losses["consequence_value_margin_mean"].detach().cpu()),
        "frontier_loss": float(frontier_loss.detach().cpu()),
    }


def _expected_calibration_error(probs: Tensor, targets: Tensor, bins: int = 10) -> float:
    confidences, predictions = probs.max(dim=1)
    correctness = predictions.eq(targets)
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        acc = correctness[mask].float().mean().item()
        conf = confidences[mask].mean().item()
        ece += mask.float().mean().item() * abs(acc - conf)
    return float(ece)


def evaluate_model(model: PokeStrategistDecisionModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    non_blocking = device.type == "cuda" and bool(getattr(loader, "pin_memory", False))
    gold_in_top3 = 0
    gold_top1 = 0
    covered_gold_in_top3 = 0
    covered_gold_top1 = 0
    family_top1 = 0
    switch_correct = 0
    switch_species_correct = 0
    switch_total = 0
    switch_head_hits = 0
    head_routing_hits = 0
    line_hits = 0
    candidate_coverage = 0
    outcome_total = 0.0
    endgame_total = 0
    endgame_hits = 0
    false_switch_on_move = 0
    move_row_total = 0
    frontier_gold_hits = 0
    frontier_width_total = 0.0
    particle_mae_total = 0.0
    particle_mae_count = 0
    reveal_bce_total = 0.0
    reveal_bce_count = 0
    typed_axis_hits = 0
    typed_axis_total = 0
    consequence_value_gold_total = 0.0
    consequence_value_hard_negative_total = 0.0
    consequence_value_margin_total = 0.0
    consequence_value_diag_count = 0
    all_probs: list[Tensor] = []
    all_targets: list[Tensor] = []
    total = 0
    covered_total = 0

    with torch.no_grad():
        for batch in loader:
            tensor_batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            outputs = model(tensor_batch)
            scores = outputs["legal_action_scores"]
            top3 = scores.topk(k=min(3, scores.size(1)), dim=1).indices
            gold = tensor_batch["legal_target"]
            covered_mask = tensor_batch["gold_candidate_covered"].bool()

            gold_in_top3 += top3.eq(gold.unsqueeze(1)).any(dim=1)[covered_mask].sum().item()
            predicted_top1 = scores.argmax(dim=1)
            gold_top1 += predicted_top1.eq(gold)[covered_mask].sum().item()
            total += gold.size(0)
            covered_total += covered_mask.sum().item()

            predicted_family = tensor_batch["legal_action_family_ids"].gather(1, predicted_top1.unsqueeze(1)).squeeze(1)
            family_top1 += predicted_family.eq(tensor_batch["family_target"]).sum().item()
            head_routing_hits += outputs["head_logits"].argmax(dim=1).eq(tensor_batch["head_target"]).sum().item()
            if "line_logits" in outputs:
                line_hits += outputs["line_logits"].argmax(dim=1).eq(tensor_batch["line_target"]).sum().item()
            candidate_coverage += tensor_batch["gold_candidate_covered"].sum().item()
            covered_gold_in_top3 += top3.eq(gold.unsqueeze(1)).any(dim=1)[covered_mask].sum().item()
            covered_gold_top1 += predicted_top1.eq(gold)[covered_mask].sum().item()

            predicted_head = tensor_batch["legal_action_head_ids"].gather(1, predicted_top1.unsqueeze(1)).squeeze(1)
            predicted_switch_slot = tensor_batch["legal_action_switch_slots"].gather(1, predicted_top1.unsqueeze(1)).squeeze(1)
            predicted_switch_species = tensor_batch["legal_action_species_ids"].gather(1, predicted_top1.unsqueeze(1)).squeeze(1)
            switch_mask = tensor_batch["gold_is_switch"]
            switch_total += switch_mask.sum().item()
            if switch_mask.any():
                switch_head_hits += (predicted_head[switch_mask] == 1).sum().item()
                switch_hits = (predicted_head == 1) & predicted_switch_slot.eq(tensor_batch["gold_switch_slot"])
                species_hits = switch_hits & predicted_switch_species.eq(tensor_batch["gold_switch_species_id"])
                switch_correct += switch_hits[switch_mask].sum().item()
                switch_species_correct += species_hits[switch_mask].sum().item()

            move_mask = ~switch_mask
            move_row_total += move_mask.sum().item()
            false_switch_on_move += ((predicted_head == 1) & move_mask).sum().item()

            outcome_proxy = outputs["state_future_mean"][:, -1]
            outcome_total += outcome_proxy.sum().item()
            end_mask = tensor_batch["unlock_target"] >= 0.75
            endgame_total += end_mask.sum().item()
            if end_mask.any():
                endgame_hits += (outcome_proxy[end_mask] > 0).sum().item()

            if covered_mask.any():
                all_probs.append(scores.softmax(dim=1)[covered_mask].cpu())
                all_targets.append(gold[covered_mask].cpu())

            if "frontier_mask" in outputs:
                valid_gold = gold >= 0
                if valid_gold.any():
                    frontier_gold_hits += (
                        outputs["frontier_mask"][valid_gold]
                        .gather(1, gold[valid_gold].unsqueeze(1))
                        .squeeze(1)
                        .float()
                        .sum()
                        .item()
                    )
                frontier_width_total += outputs["frontier_mask"].float().sum(dim=1).sum().item()
            if "particle_posterior" in outputs:
                particle_mae_total += torch.abs(outputs["particle_posterior"] - tensor_batch["particle_posterior_target"]).sum().item()
                particle_mae_count += outputs["particle_posterior"].numel()
            if "reveal_likelihood_logits" in outputs:
                reveal_bce_total += F.binary_cross_entropy_with_logits(
                    outputs["reveal_likelihood_logits"],
                    tensor_batch["reveal_likelihood_target"],
                    reduction="sum",
                ).item()
                reveal_bce_count += outputs["reveal_likelihood_logits"].numel()
            if "typed_consequence_logits" in outputs and (gold >= 0).any():
                valid = gold >= 0
                valid_gold = gold[valid]
                gather_index = valid_gold.view(-1, 1, 1, 1).expand(
                    -1,
                    1,
                    outputs["typed_consequence_logits"].size(2),
                    outputs["typed_consequence_logits"].size(3),
                )
                gold_logits = outputs["typed_consequence_logits"][valid].gather(1, gather_index).squeeze(1)
                typed_axis_hits += gold_logits.argmax(dim=-1).eq(tensor_batch["typed_consequence_bin_target"][valid]).sum().item()
                typed_axis_total += tensor_batch["typed_consequence_bin_target"][valid].numel()
            if "consequence_value_scores" in outputs and (gold >= 0).any():
                valid = gold >= 0
                valid_gold = gold[valid]
                valid_scores = outputs["consequence_value_scores"][valid]
                gold_scores = valid_scores.gather(1, valid_gold.unsqueeze(1)).squeeze(1)
                negative_mask = tensor_batch["legal_action_mask"][valid].bool().clone()
                negative_mask.scatter_(1, valid_gold.unsqueeze(1), False)
                rows_with_negative = negative_mask.any(dim=1)
                if rows_with_negative.any():
                    negative_scores = valid_scores[rows_with_negative].masked_fill(~negative_mask[rows_with_negative], -1e9)
                    hard_negative_scores = negative_scores.max(dim=1).values
                    row_gold_scores = gold_scores[rows_with_negative]
                    consequence_value_gold_total += row_gold_scores.sum().item()
                    consequence_value_hard_negative_total += hard_negative_scores.sum().item()
                    consequence_value_margin_total += (row_gold_scores - hard_negative_scores).sum().item()
                    consequence_value_diag_count += row_gold_scores.numel()

    action_probs = torch.cat(all_probs, dim=0) if all_probs else torch.zeros(0, model.config.max_legal_actions)
    action_targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0, dtype=torch.long)
    ece = _expected_calibration_error(action_probs, action_targets) if covered_total else 0.0
    return {
        "gold_in_top3": gold_in_top3 / total if total else 0.0,
        "gold_top1": gold_top1 / total if total else 0.0,
        "covered_top3": covered_gold_in_top3 / covered_total if covered_total else 0.0,
        "covered_top1": covered_gold_top1 / covered_total if covered_total else 0.0,
        "top1_family_accuracy": family_top1 / total if total else 0.0,
        "gold_candidate_coverage": candidate_coverage / total if total else 0.0,
        "head_routing_accuracy": head_routing_hits / total if total else 0.0,
        "line_accuracy": line_hits / total if total else 0.0,
        "switch_head_recall": switch_head_hits / switch_total if switch_total else 0.0,
        "switch_target_accuracy": switch_correct / switch_total if switch_total else 0.0,
        "switch_target_species_accuracy": switch_species_correct / switch_total if switch_total else 0.0,
        "false_switch_rate_on_move": false_switch_on_move / move_row_total if move_row_total else 0.0,
        "frontier_gold_recall": frontier_gold_hits / total if total else 0.0,
        "frontier_mean_width": frontier_width_total / total if total else 0.0,
        "particle_mae": particle_mae_total / particle_mae_count if particle_mae_count else 0.0,
        "reveal_bce": reveal_bce_total / reveal_bce_count if reveal_bce_count else 0.0,
        "typed_consequence_axis_accuracy": typed_axis_hits / typed_axis_total if typed_axis_total else 0.0,
        "consequence_value_gold_mean": consequence_value_gold_total / consequence_value_diag_count if consequence_value_diag_count else 0.0,
        "consequence_value_hard_negative_mean": consequence_value_hard_negative_total / consequence_value_diag_count if consequence_value_diag_count else 0.0,
        "consequence_value_margin_mean": consequence_value_margin_total / consequence_value_diag_count if consequence_value_diag_count else 0.0,
        "ece": ece,
        "outcome_proxy": outcome_total / total if total else 0.0,
        "endgame_stability": endgame_hits / endgame_total if endgame_total else 0.0,
    }


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    trainer_config: TrainerConfig | None = None,
    model_config: PokeStrategistDecisionConfig | None = None,
    max_samples: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    trainer_config = trainer_config or TrainerConfig()
    model_config_input = model_config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing dataset from {dataset_path}...", flush=True)
    dataset = DecisionTensorDataset(
        dataset_path,
        max_samples=max_samples,
        hidden_candidate_topk=trainer_config.hidden_candidate_topk,
        use_usage_priors=trainer_config.use_usage_priors,
        usage_data_dir=trainer_config.usage_data_dir,
    )
    train_indices, val_indices = grouped_replay_split(dataset.replay_ids, seed=trainer_config.split_seed)
    if not train_indices and val_indices:
        train_indices = [val_indices[0]]
        val_indices = val_indices[1:] or list(train_indices)
    if not val_indices and train_indices:
        val_indices = [train_indices[0]]
        train_indices = train_indices[1:] or list(val_indices)
    print(
        f"Loaded dataset with {len(dataset)} samples; train={len(train_indices)} val={len(val_indices)}.",
        flush=True,
    )
    prior_cache_metadata = _train_prior_cache_metadata(dataset, trainer_config=trainer_config, max_samples=max_samples)
    cached_priors = _load_train_prior_cache(dataset.path, prior_cache_metadata)
    if cached_priors is not None:
        print(
            f"Loading train priors from cache { _train_prior_cache_paths(dataset.path, str(prior_cache_metadata['cache_key']))['dir'] }...",
            flush=True,
        )
        hidden_move_prior, team_preview_prior = cached_priors
    else:
        print("Building train priors...", flush=True)
        hidden_move_prior, team_preview_prior = _build_train_priors(
            dataset.iter_raw_samples(train_indices),
            default_topk=trainer_config.hidden_candidate_topk,
            use_team_preview_prior=trainer_config.use_team_preview_prior,
        )
        cache_dir = _save_train_prior_cache(
            dataset.path,
            prior_cache_metadata,
            hidden_move_prior=hidden_move_prior,
            team_preview_prior=team_preview_prior,
        )
        print(f"Saved train priors cache to {cache_dir}.", flush=True)
    dataset.set_hidden_move_prior(hidden_move_prior)
    dataset.set_team_preview_prior(team_preview_prior)

    device = torch.device(trainer_config.device)
    _configure_torch_runtime(device, enable_tf32=trainer_config.enable_tf32)
    train_num_workers = _resolve_num_workers(trainer_config.train_num_workers, device=device)
    eval_num_workers = _resolve_num_workers(
        trainer_config.eval_num_workers,
        device=device,
        fallback=0,
    )
    pin_memory = _resolve_pin_memory(trainer_config.pin_memory, device=device)
    print(
        f"Using device={device}, train_workers={train_num_workers}, eval_workers={eval_num_workers}, pin_memory={pin_memory}.",
        flush=True,
    )
    train_loader = _build_loader(
        Subset(dataset, train_indices),
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=trainer_config.prefetch_factor,
        persistent_workers=trainer_config.persistent_workers and train_num_workers > 0,
    )
    val_loader = _build_loader(
        Subset(dataset, val_indices),
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=trainer_config.prefetch_factor,
        persistent_workers=trainer_config.persistent_workers and eval_num_workers > 0,
    )
    model_config = model_config_input or PokeStrategistDecisionConfig()
    model = PokeStrategistDecisionModel(model_config).to(device)

    if checkpoint_path is not None:
        print(f"Initializing from checkpoint: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Merge config: use checkpoint dimensions for compatibility
        ckpt_config = dict(checkpoint.get("model_config", {}))
        if "candidate_prior_feature_dim" not in ckpt_config:
            prior_weight = checkpoint["model_state"].get("prior_feature_proj.0.weight")
            ckpt_config["candidate_prior_feature_dim"] = int(prior_weight.shape[1]) if prior_weight is not None else model_config.candidate_prior_feature_dim
        if "preview_prior_feature_dim" not in ckpt_config:
            preview_weight = checkpoint["model_state"].get("preview_prior_mlp.0.weight")
            from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM
            ckpt_config["preview_prior_feature_dim"] = int(preview_weight.shape[1]) if preview_weight is not None else TEAM_PREVIEW_FEATURE_DIM
        preserved_config_keys = {
            "consequence_value_weight",
            "consequence_value_latent_weight",
            "consequence_value_move_scale",
            "consequence_value_switch_scale",
        }
        merged = {k: v for k, v in ckpt_config.items() if k not in preserved_config_keys}
        for key in preserved_config_keys:
            merged[key] = getattr(model_config, key)
        ckpt_model_config = PokeStrategistDecisionConfig(**merged)
        from pokestrategist.models.decision_model import build_model_for_checkpoint
        model = build_model_for_checkpoint(ckpt_model_config, checkpoint["model_state"]).to(device)

        # Expand preview_prior_mlp weights if old checkpoint uses smaller feature dim
        from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM
        old_dim = ckpt_model_config.preview_prior_feature_dim
        if old_dim < TEAM_PREVIEW_FEATURE_DIM:
            print(f"Expanding preview_prior_mlp from {old_dim} → {TEAM_PREVIEW_FEATURE_DIM} dims", flush=True)
            old_weight = model.preview_prior_mlp[0].weight  # [hidden, old_dim]
            old_bias = model.preview_prior_mlp[0].bias      # [hidden]
            new_weight = torch.zeros(old_weight.size(0), TEAM_PREVIEW_FEATURE_DIM, device=old_weight.device, dtype=old_weight.dtype)
            new_weight[:, :old_dim] = old_weight
            # Rebuild first layer with expanded input
            new_first = torch.nn.Linear(TEAM_PREVIEW_FEATURE_DIM, old_weight.size(0))
            new_first.weight.data.copy_(new_weight)
            new_first.bias.data.copy_(old_bias)
            model.preview_prior_mlp[0] = new_first.to(device)
            model.config.preview_prior_feature_dim = TEAM_PREVIEW_FEATURE_DIM

        model_config = model.config
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as error:
                print(f"Skipping optimizer state restore: {error}", flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
    non_blocking = device.type == "cuda" and pin_memory

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_ranking_score = float("-inf")
    best_switch_score = float("-inf")
    best_metrics: dict[str, float] | None = None
    best_ranking_metrics: dict[str, float] | None = None
    best_switch_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_ranking_epoch = 0
    best_switch_epoch = 0

    def _checkpoint_payload(metrics: dict[str, float], *, epoch: int) -> dict[str, Any]:
        payload_metrics = dict(metrics)
        payload_metrics["ranking_score"] = _ranking_score(metrics)
        payload_metrics["switch_selection_score"] = _switch_selection_score(metrics)
        payload_metrics["selection_score"] = _selection_score(metrics)
        payload_metrics["best_epoch"] = float(epoch)
        return {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer_config),
            "metrics": payload_metrics,
            "hidden_move_prior": hidden_move_prior.to_dict(),
            "team_preview_prior": team_preview_prior.to_dict() if team_preview_prior is not None else None,
        }

    for epoch in range(trainer_config.epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_loss_stats: defaultdict[str, list[float]] = defaultdict(list)
        current_learning_rate = _epoch_learning_rate(
            base_learning_rate=trainer_config.learning_rate,
            epoch_index=epoch,
            total_epochs=trainer_config.epochs,
            warmup_epochs=trainer_config.warmup_epochs,
            min_learning_rate_ratio=trainer_config.min_learning_rate_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate
        print(f"Starting epoch {epoch + 1}/{trainer_config.epochs}...", flush=True)
        for batch in train_loader:
            tensor_batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            outputs = model(tensor_batch)
            loss, loss_stats = _compute_losses(tensor_batch, outputs, trainer_config)
            optimizer.zero_grad()
            loss.backward()
            if trainer_config.max_grad_norm is not None and trainer_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            for key, value in loss_stats.items():
                epoch_loss_stats[key].append(float(value))

        metrics = evaluate_model(model, val_loader, device)
        metrics["epoch"] = float(epoch + 1)
        metrics["train_loss"] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        # Per-epoch loss component averages (only include if non-zero weight)
        for key, values in epoch_loss_stats.items():
            if values:
                metrics[key] = sum(values) / len(values)
        metrics["learning_rate"] = current_learning_rate
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
    best_metrics = best_metrics or final_metrics

    final_checkpoint = _checkpoint_payload(final_metrics, epoch=trainer_config.epochs)
    torch.save(final_checkpoint, output_dir / "final_model.pt")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "hidden_move_prior.json").write_text(json.dumps(hidden_move_prior.to_dict(), indent=2), encoding="utf-8")
    if team_preview_prior is not None:
        (output_dir / "team_preview_prior.json").write_text(
            json.dumps(team_preview_prior.to_dict(), indent=2),
            encoding="utf-8",
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
        best_metrics=best_metrics,
        final_metrics=final_metrics,
        best_ranking_metrics=best_ranking_metrics,
        best_switch_metrics=best_switch_metrics,
    )
    return best_metrics


__all__ = ["TrainerConfig", "evaluate_model", "train_model", "HiddenMovePrior"]
