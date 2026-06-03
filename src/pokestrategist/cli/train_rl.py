"""Train the v1 decision model with Monte-Carlo Q-learning on replay outcomes."""

from __future__ import annotations

import argparse
import json

from pokestrategist.training.rl_trainer import MCTrainerConfig, train_mc_q_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC-Q train the v1 decision model on replay outcomes.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Initialization checkpoint (stage3 supervised).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-candidate-topk", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.2)
    parser.add_argument("--q-weight", type=float, default=0.2)
    parser.add_argument("--v-weight", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.8)
    parser.add_argument("--gold-q-weight", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-team-preview-prior", action="store_true")
    parser.add_argument("--disable-usage-priors", action="store_true")
    parser.add_argument("--usage-data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_mc_q_model(
        args.data,
        args.output_dir,
        checkpoint_path=args.checkpoint,
        mc_config=MCTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            train_num_workers=args.num_workers,
            eval_num_workers=args.eval_num_workers,
            pin_memory=not args.disable_pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=not args.disable_persistent_workers,
            enable_tf32=not args.disable_tf32,
            split_seed=args.split_seed,
            hidden_candidate_topk=args.hidden_candidate_topk,
            max_grad_norm=args.max_grad_norm,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            q_weight=args.q_weight,
            v_weight=args.v_weight,
            aux_weight=args.aux_weight,
            gold_q_weight=args.gold_q_weight,
            use_team_preview_prior=not args.disable_team_preview_prior,
            use_usage_priors=not args.disable_usage_priors,
            usage_data_dir=args.usage_data_dir,
        ),
        max_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
