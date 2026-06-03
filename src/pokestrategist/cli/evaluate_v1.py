"""Evaluate the clean v1 decision model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pokestrategist.data.hidden_move_prior import HiddenMovePrior
from pokestrategist.data.team_preview_prior import TeamPreviewPriorCatalog
from pokestrategist.models.decision_model import PokeStrategistDecisionConfig, build_model_for_checkpoint
from pokestrategist.training.dataset import DecisionTensorDataset, collate_decision_batch
from pokestrategist.training.trainer import evaluate_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the clean v1 decision model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-team-preview-prior", action="store_true")
    parser.add_argument("--disable-usage-priors", action="store_true")
    parser.add_argument("--usage-data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not args.disable_tf32:
            torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    trainer_config = checkpoint.get("trainer_config", {})
    hidden_move_prior = HiddenMovePrior.from_dict(checkpoint.get("hidden_move_prior"))
    team_preview_prior = TeamPreviewPriorCatalog.from_dict(checkpoint["team_preview_prior"]) if checkpoint.get("team_preview_prior") and not args.disable_team_preview_prior else None
    use_usage_priors = bool(trainer_config.get("use_usage_priors", True)) and not args.disable_usage_priors
    usage_data_dir = args.usage_data_dir if args.usage_data_dir is not None else trainer_config.get("usage_data_dir")
    dataset = DecisionTensorDataset(
        args.data,
        max_samples=args.max_samples,
        hidden_move_prior=hidden_move_prior,
        team_preview_prior=team_preview_prior,
        hidden_candidate_topk=int(trainer_config.get("hidden_candidate_topk", 4)),
        use_usage_priors=use_usage_priors,
        usage_data_dir=usage_data_dir,
    )
    requested_num_workers = args.num_workers
    if requested_num_workers is None:
        requested_num_workers = trainer_config.get("eval_num_workers", trainer_config.get("train_num_workers"))
    if requested_num_workers is None:
        requested_num_workers = max(1, min(8, max(1, (os.cpu_count() or 1) - 2))) if device.type == "cuda" else 0
    num_workers = max(0, int(requested_num_workers))
    pin_memory = device.type == "cuda" and not args.disable_pin_memory
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_decision_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = not args.disable_persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)
    model_config_payload = dict(checkpoint["model_config"])
    if "use_static_rule_features" not in model_config_payload:
        model_config_payload["use_static_rule_features"] = False
    if "candidate_prior_feature_dim" not in model_config_payload:
        prior_weight = checkpoint["model_state"].get("prior_feature_proj.0.weight")
        model_config_payload["candidate_prior_feature_dim"] = int(prior_weight.shape[1]) if prior_weight is not None else 2
    if "preview_prior_feature_dim" not in model_config_payload:
        preview_weight = checkpoint["model_state"].get("preview_prior_mlp.0.weight")
        from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM
        model_config_payload["preview_prior_feature_dim"] = int(preview_weight.shape[1]) if preview_weight is not None else TEAM_PREVIEW_FEATURE_DIM
    model = build_model_for_checkpoint(PokeStrategistDecisionConfig(**model_config_payload), checkpoint["model_state"])
    model.to(device)
    metrics = evaluate_model(model, loader, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()