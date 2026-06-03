"""Average two pokestrategist checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average two pokestrategist model checkpoints.")
    parser.add_argument("--checkpoint-a", required=True, help="Base checkpoint; metadata and non-float tensors are copied from this file.")
    parser.add_argument("--checkpoint-b", required=True, help="Second checkpoint; floating tensors are blended with checkpoint A.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for checkpoint B. 0 keeps A, 1 keeps B.")
    return parser.parse_args()


def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    return value


def average_model_states(
    state_a: dict[str, Tensor],
    state_b: dict[str, Tensor],
    *,
    alpha: float,
) -> tuple[dict[str, Tensor], dict[str, int]]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    averaged: dict[str, Tensor] = {}
    stats = {
        "averaged_tensor_count": 0,
        "copied_tensor_count": 0,
        "missing_or_mismatched_tensor_count": 0,
    }
    for key, tensor_a in state_a.items():
        tensor_b = state_b.get(key)
        if (
            isinstance(tensor_a, Tensor)
            and isinstance(tensor_b, Tensor)
            and tensor_a.shape == tensor_b.shape
            and tensor_a.dtype == tensor_b.dtype
            and tensor_a.is_floating_point()
        ):
            averaged[key] = tensor_a.detach().cpu().mul(1.0 - alpha).add(tensor_b.detach().cpu(), alpha=alpha)
            stats["averaged_tensor_count"] += 1
        else:
            averaged[key] = tensor_a.detach().cpu().clone() if isinstance(tensor_a, Tensor) else _clone_value(tensor_a)
            stats["copied_tensor_count"] += 1
            if not (isinstance(tensor_b, Tensor) and isinstance(tensor_a, Tensor) and tensor_a.shape == tensor_b.shape):
                stats["missing_or_mismatched_tensor_count"] += 1
    return averaged, stats


def average_checkpoints(
    checkpoint_a_path: str | Path,
    checkpoint_b_path: str | Path,
    output_path: str | Path,
    *,
    alpha: float = 0.5,
) -> dict[str, Any]:
    checkpoint_a_path = Path(checkpoint_a_path)
    checkpoint_b_path = Path(checkpoint_b_path)
    output_path = Path(output_path)
    checkpoint_a = torch.load(checkpoint_a_path, map_location="cpu")
    checkpoint_b = torch.load(checkpoint_b_path, map_location="cpu")
    averaged_state, stats = average_model_states(
        checkpoint_a["model_state"],
        checkpoint_b["model_state"],
        alpha=alpha,
    )
    payload = {key: value for key, value in checkpoint_a.items() if key not in {"model_state", "optimizer"}}
    payload["model_state"] = averaged_state
    payload["averaging"] = {
        "checkpoint_a": str(checkpoint_a_path),
        "checkpoint_b": str(checkpoint_b_path),
        "alpha": min(max(float(alpha), 0.0), 1.0),
        **stats,
    }
    metrics = dict(payload.get("metrics", {}))
    metrics["averaged_checkpoint_alpha"] = payload["averaging"]["alpha"]
    payload["metrics"] = metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload["averaging"]


def main() -> None:
    args = _parse_args()
    summary = average_checkpoints(args.checkpoint_a, args.checkpoint_b, args.output, alpha=args.alpha)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
