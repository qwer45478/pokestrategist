from __future__ import annotations

from torch.utils.data import DataLoader

from pokestrategist.data.static_rules import (
    ABILITY_RULE_FEATURE_DIM,
    ITEM_RULE_FEATURE_DIM,
    MATCHUP_RULE_FEATURE_DIM,
    MOVE_RULE_FEATURE_DIM,
    SPECIES_RULE_FEATURE_DIM,
)
from pokestrategist.data.static_rules import load_static_rule_catalog
from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM, build_team_preview_prior, iter_team_preview_examples
from pokestrategist.models import PokeStrategistDecisionModel
from pokestrategist.training.dataset import DecisionTensorDataset, collate_decision_batch

from tests.v1_fixtures import write_dataset


def test_static_rule_catalog_covers_type_matchup():
    catalog = load_static_rule_catalog()

    assert catalog.type_multiplier("ground", catalog.species_types("Gholdengo")) == 2.0
    assert catalog.type_multiplier("normal", catalog.species_types("Gholdengo")) == 0.0
    assert catalog.base_speed("Dragapult") > catalog.base_speed("Gholdengo")


def test_decision_model_forward_shapes(tmp_path):
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("replay-a",))
    dataset = DecisionTensorDataset(dataset_path)
    dataset.set_team_preview_prior(build_team_preview_prior(iter_team_preview_examples(dataset.iter_raw_samples(range(len(dataset))))))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_decision_batch)
    batch = next(iter(loader))
    model = PokeStrategistDecisionModel()

    outputs = model(batch)

    assert batch["self_species_rule_features"].shape == (2, SPECIES_RULE_FEATURE_DIM)
    assert batch["opp_species_rule_features"].shape == (2, SPECIES_RULE_FEATURE_DIM)
    assert batch["self_item_rule_features"].shape == (2, ITEM_RULE_FEATURE_DIM)
    assert batch["self_ability_rule_features"].shape == (2, ABILITY_RULE_FEATURE_DIM)
    assert batch["opp_item_rule_features"].shape == (2, ITEM_RULE_FEATURE_DIM)
    assert batch["opp_ability_rule_features"].shape == (2, ABILITY_RULE_FEATURE_DIM)
    assert batch["opp_preview_prior_features"].shape == (2, TEAM_PREVIEW_FEATURE_DIM)
    assert batch["legal_action_species_rule_features"].shape == (2, 32, SPECIES_RULE_FEATURE_DIM)
    assert batch["legal_action_move_rule_features"].shape == (2, 32, MOVE_RULE_FEATURE_DIM)
    assert batch["legal_action_matchup_rule_features"].shape == (2, 32, MATCHUP_RULE_FEATURE_DIM)
    assert batch["legal_action_external_prior_features"].shape == (2, 32, 4)
    assert batch["legal_action_usage_scores"].shape == (2, 32)
    assert batch["particle_posterior_target"].shape == (2, model.config.particle_posterior_dim)
    assert batch["reveal_likelihood_target"].shape == (2, model.config.reveal_dim)
    assert batch["typed_consequence_target"].shape == (2, model.config.typed_consequence_dim)
    assert batch["typed_consequence_bin_target"].shape == (2, model.config.typed_consequence_dim)
    assert batch["typed_consequence_interaction_target"].shape == (2, model.config.typed_consequence_interaction_dim)
    assert float(batch["legal_action_external_prior_features"].max()) > 0.0
    assert float(batch["legal_action_usage_scores"].max()) > 0.0
    assert float(batch["opp_preview_prior_features"].max()) > 0.0
    assert batch["line_target"].shape == (2,)
    assert outputs["plan_posterior"].shape == (2, 4)
    assert outputs["phase_posterior"].shape == (2, 4)
    assert outputs["line_posterior"].shape == (2, model.config.line_count)
    assert outputs["particle_posterior"].shape == (2, model.config.particle_posterior_dim)
    assert outputs["particle_uncertainty"].shape == (2,)
    assert outputs["reveal_likelihood_logits"].shape == (2, model.config.reveal_dim)
    assert outputs["belief_summary"].shape == (2, 6)
    assert outputs["resource_ledger"].shape == (2, 6)
    assert outputs["head_logits"].shape == (2, 2)
    assert outputs["response_logits"].shape == (2, 6)
    assert outputs["state_future_mean"].shape == (2, 8)
    assert outputs["state_future_tail"].shape == (2, 8)
    assert outputs["candidate_future_mean"].shape == (2, 32, 8)
    assert outputs["candidate_future_tail"].shape == (2, 32, 8)
    assert outputs["typed_consequence_logits"].shape == (
        2,
        32,
        model.config.typed_consequence_dim,
        model.config.typed_consequence_bin_count,
    )
    assert outputs["typed_consequence_values"].shape == (2, 32, model.config.typed_consequence_dim)
    assert outputs["typed_consequence_interactions"].shape == (2, 32, model.config.typed_consequence_interaction_dim)
    assert outputs["typed_consequence_scores"].shape == (2, 32)
    assert outputs["typed_interaction_scores"].shape == (2, 32)
    assert outputs["consequence_value_scores"].shape == (2, 32)
    assert outputs["consequence_value_base_scores"].shape == (2, 32)
    assert outputs["consequence_value_latent_scores"].shape == (2, 32)
    assert outputs["consequence_value_action_scales"].shape == (2, 32)
    assert outputs["consequence_value_residual_scores"].shape == (2, 32)
    assert outputs["local_future_scores"].shape == (2, 32)
    assert outputs["local_tail_penalty"].shape == (2, 32)
    assert outputs["local_future_gate"].shape == (2, 32)
    assert outputs["candidate_structure_latents"].shape == (2, 32, model.config.hidden_size)
    assert outputs["candidate_identity_latents"].shape == (2, 32, model.config.hidden_size)
    assert outputs["candidate_prior_latents"].shape == (2, 32, model.config.hidden_size)
    assert outputs["candidate_meta_latents"].shape == (2, 32, model.config.hidden_size)
    assert outputs["candidate_matchup_latents"].shape == (2, 32, model.config.hidden_size)
    assert outputs["state_self_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_opp_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_history_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_numeric_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_preview_prior_latents"].shape == (2, model.config.hidden_size // 2)
    assert outputs["state_plan_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_belief_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_resource_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_phase_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_unlock_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_line_latents"].shape == (2, model.config.hidden_size)
    assert outputs["state_decision_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_line_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_route_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_response_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_future_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_branch_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_support_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_rule_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_particle_context"].shape == (2, model.config.hidden_size)
    assert outputs["state_consequence_mean"].shape == (2, 8)
    assert outputs["state_consequence_tail"].shape == (2, 8)
    assert outputs["candidate_consequence_mean"].shape == (2, 32, 8)
    assert outputs["candidate_consequence_tail"].shape == (2, 32, 8)
    assert outputs["imitation_logits"].shape == (2, 32)
    assert outputs["rule_value_scores"].shape == (2, 32)
    assert outputs["rule_gate"].shape == (2, 32)
    assert outputs["candidate_response_logits"].shape == (2, 32, 6)
    assert outputs["candidate_response_probs"].shape == (2, 32, 6)
    assert outputs["response_value_scores"].shape == (2, 32)
    assert outputs["prior_source_gate"].shape == (2, 32)
    assert outputs["prior_mix_scores"].shape == (2, 32)
    assert outputs["recoverable_support_logits"].shape == (2, 32)
    assert outputs["move_branch_logits"].shape == (2, 32)
    assert outputs["switch_branch_logits"].shape == (2, 32)
    assert outputs["switch_entry_value"].shape == (2, 32)
    assert outputs["switch_entry_scores"].shape == (2, 32)
    assert outputs["switch_follow_mean"].shape == (2, 32, 8)
    assert outputs["switch_follow_tail"].shape == (2, 32, 8)
    assert outputs["switch_follow_scores"].shape == (2, 32)
    assert outputs["switch_subgame_scores"].shape == (2, 32)
    assert outputs["branch_logits"].shape == (2, 32)
    assert outputs["legal_action_scores"].shape == (2, 32)
    assert outputs["route_bias"].shape == (2, 32)
    assert outputs["adaptive_epsilon"].shape == (2,)
    assert outputs["frontier_threshold"].shape == (2,)
    assert outputs["frontier_mask"].shape == (2, 32)
    assert outputs["frontier_policy_logits"].shape == (2, 32)
    assert outputs["frontier_policy_probs"].shape == (2, 32)
    assert outputs["frontier_mask"].any(dim=1).all()
    assert outputs["switch_subgame_scores"][~batch["switch_candidate_mask"]].abs().sum().item() == 0.0


def test_tensor_dataset_leaves_uncovered_gold_unsupervised(tmp_path):
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("replay-a",))
    dataset = DecisionTensorDataset(dataset_path, use_usage_priors=False)

    uncovered_rows = [dataset[index] for index in range(len(dataset)) if not dataset[index]["gold_candidate_covered"]]

    assert uncovered_rows
    assert all(row["legal_target"] == -1 for row in uncovered_rows)
    assert all(not row["supervised_target"] for row in uncovered_rows)
