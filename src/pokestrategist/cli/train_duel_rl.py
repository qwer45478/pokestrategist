"""Train a duel-regularized offline RL refinement on processed decision states."""

from __future__ import annotations

import argparse
import json

from pokestrategist.training.duel_rl import DuelRLConfig, train_duel_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline duel RL refinement from a v1 checkpoint.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Student initialization checkpoint.")
    parser.add_argument("--reference-checkpoint", default=None, help="Frozen reference checkpoint. Defaults to --checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
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
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--reference-temperature", type=float, default=1.0)
    parser.add_argument("--imitation-weight", type=float, default=0.15)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--gold-logprob-weight", type=float, default=1.0)
    parser.add_argument("--top1-margin-weight", type=float, default=0.35)
    parser.add_argument("--top3-margin-weight", type=float, default=0.2)
    parser.add_argument("--top1-margin", type=float, default=0.1)
    parser.add_argument("--top3-margin", type=float, default=0.05)
    parser.add_argument("--switch-sample-weight", type=float, default=2.0)
    parser.add_argument("--gold-reward", type=float, default=1.0)
    parser.add_argument("--family-reward", type=float, default=0.2)
    parser.add_argument("--head-reward", type=float, default=0.15)
    parser.add_argument("--switch-reward", type=float, default=0.3)
    parser.add_argument("--switch-species-reward", type=float, default=0.2)
    parser.add_argument("--false-switch-penalty", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-team-preview-prior", action="store_true")
    parser.add_argument("--disable-usage-priors", action="store_true")
    parser.add_argument("--usage-data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_duel_policy(
        args.data,
        args.output_dir,
        checkpoint_path=args.checkpoint,
        reference_checkpoint_path=args.reference_checkpoint,
        rl_config=DuelRLConfig(
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
            policy_temperature=args.policy_temperature,
            reference_temperature=args.reference_temperature,
            imitation_weight=args.imitation_weight,
            kl_weight=args.kl_weight,
            entropy_weight=args.entropy_weight,
            gold_logprob_weight=args.gold_logprob_weight,
            top1_margin_weight=args.top1_margin_weight,
            top3_margin_weight=args.top3_margin_weight,
            top1_margin=args.top1_margin,
            top3_margin=args.top3_margin,
            switch_sample_weight=args.switch_sample_weight,
            gold_reward=args.gold_reward,
            family_reward=args.family_reward,
            head_reward=args.head_reward,
            switch_reward=args.switch_reward,
            switch_species_reward=args.switch_species_reward,
            false_switch_penalty=args.false_switch_penalty,
            use_team_preview_prior=not args.disable_team_preview_prior,
            use_usage_priors=not args.disable_usage_priors,
            usage_data_dir=args.usage_data_dir,
        ),
        max_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()