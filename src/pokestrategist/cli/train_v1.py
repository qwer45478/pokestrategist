"""Train the clean v1 decision model."""

from __future__ import annotations

import argparse
import json

from pokestrategist.models.decision_model import PokeStrategistDecisionConfig
from pokestrategist.training.trainer import TrainerConfig, train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean v1 decision model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default=None, help="Initialize from an existing checkpoint (stage3).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--response-topk", type=int, default=3)
    parser.add_argument("--recoverable-support-weight", type=float, default=0.25)
    parser.add_argument("--local-score-weight", type=float, default=0.25)
    parser.add_argument("--local-tail-weight", type=float, default=0.10)
    parser.add_argument("--rule-score-weight", type=float, default=0.20)
    parser.add_argument("--prior-score-weight", type=float, default=0.10)
    parser.add_argument("--switch-subgame-weight", type=float, default=0.20)
    parser.add_argument("--switch-follow-discount", type=float, default=0.50)
    parser.add_argument("--disable-static-rule-features", action="store_true")
    parser.add_argument("--irreversibility-weight", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-candidate-topk", type=int, default=4)
    parser.add_argument("--head-route-weight", type=float, default=0.25)
    parser.add_argument("--preference-weight", type=float, default=0.5)
    parser.add_argument("--recoverable-bc-weight", type=float, default=0.25)
    parser.add_argument("--opponent-weight", type=float, default=0.5)
    parser.add_argument("--plan-weight", type=float, default=0.15)
    parser.add_argument("--belief-weight", type=float, default=0.15)
    parser.add_argument("--resource-weight", type=float, default=0.15)
    parser.add_argument("--phase-weight", type=float, default=0.15)
    parser.add_argument("--line-weight", type=float, default=0.2)
    parser.add_argument("--unlock-weight", type=float, default=0.15)
    parser.add_argument("--state-future-weight", dest="state_future_weight", type=float, default=0.5)
    parser.add_argument("--local-future-weight", dest="local_future_weight", type=float, default=0.2)
    parser.add_argument("--candidate-response-weight", type=float, default=0.2)
    parser.add_argument("--switch-follow-weight", type=float, default=0.15)
    parser.add_argument("--response-score-weight", type=float, default=0.15)
    parser.add_argument("--head-weight", type=float, default=0.3)
    parser.add_argument("--switch-ranking-weight", type=float, default=0.5)
    parser.add_argument("--switch-sample-weight", type=float, default=2.0)
    parser.add_argument("--action-score-ce-weight", type=float, default=0.35)
    parser.add_argument("--route-consistency-weight", type=float, default=0.0)
    parser.add_argument("--consequence-value-weight", type=float, default=0.0)
    parser.add_argument("--consequence-value-latent-weight", type=float, default=1.0)
    parser.add_argument("--consequence-value-move-scale", type=float, default=1.0)
    parser.add_argument("--consequence-value-switch-scale", type=float, default=1.0)
    parser.add_argument("--consequence-value-margin-weight", type=float, default=0.0)
    parser.add_argument("--consequence-value-margin", type=float, default=0.05)
    parser.add_argument("--consequence-value-negative-topk", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-team-preview-prior", action="store_true")
    parser.add_argument("--disable-usage-priors", action="store_true")
    parser.add_argument("--usage-data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_model(
        args.data,
        args.output_dir,
        trainer_config=TrainerConfig(
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
            preference_weight=args.preference_weight,
            recoverable_bc_weight=args.recoverable_bc_weight,
            opponent_weight=args.opponent_weight,
            plan_weight=args.plan_weight,
            belief_weight=args.belief_weight,
            resource_weight=args.resource_weight,
            phase_weight=args.phase_weight,
            line_weight=args.line_weight,
            unlock_weight=args.unlock_weight,
            state_future_weight=args.state_future_weight,
            local_future_weight=args.local_future_weight,
            candidate_response_weight=args.candidate_response_weight,
            switch_follow_weight=args.switch_follow_weight,
            head_weight=args.head_weight,
            switch_ranking_weight=args.switch_ranking_weight,
            switch_sample_weight=args.switch_sample_weight,
            action_score_ce_weight=args.action_score_ce_weight,
            route_consistency_weight=args.route_consistency_weight,
            consequence_value_weight=args.consequence_value_weight,
            consequence_value_margin_weight=args.consequence_value_margin_weight,
            consequence_value_margin=args.consequence_value_margin,
            consequence_value_negative_topk=args.consequence_value_negative_topk,
            max_grad_norm=args.max_grad_norm,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            use_team_preview_prior=not args.disable_team_preview_prior,
            use_usage_priors=not args.disable_usage_priors,
            usage_data_dir=args.usage_data_dir,
        ),
        model_config=PokeStrategistDecisionConfig(
            hidden_size=args.hidden_size,
            response_topk=args.response_topk,
            recoverable_support_weight=args.recoverable_support_weight,
            local_score_weight=args.local_score_weight,
            local_tail_weight=args.local_tail_weight,
            rule_score_weight=args.rule_score_weight,
            prior_score_weight=args.prior_score_weight,
            switch_subgame_weight=args.switch_subgame_weight,
            switch_follow_discount=args.switch_follow_discount,
            response_score_weight=args.response_score_weight,
            consequence_value_weight=args.consequence_value_weight,
            consequence_value_latent_weight=args.consequence_value_latent_weight,
            consequence_value_move_scale=args.consequence_value_move_scale,
            consequence_value_switch_scale=args.consequence_value_switch_scale,
            use_static_rule_features=not args.disable_static_rule_features,
            irreversibility_weight=args.irreversibility_weight,
            head_route_weight=args.head_route_weight,
        ),
        max_samples=args.max_samples,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()