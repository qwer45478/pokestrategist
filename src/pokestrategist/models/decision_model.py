"""Decision-assist v3 model with honest candidates and branch experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pokestrategist.data.static_rules import (
    ABILITY_RULE_FEATURE_DIM,
    ITEM_RULE_FEATURE_DIM,
    MATCHUP_RULE_FEATURE_DIM,
    MOVE_RULE_FEATURE_DIM,
    SPECIES_RULE_FEATURE_DIM,
)
from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM
from pokestrategist.data.usage_priors import USAGE_PRIOR_FEATURE_DIM


TensorMap = dict[str, Tensor]
REVEALED_SOURCE_ID = 0
SWITCH_SOURCE_ID = 1
PRIOR_HIDDEN_SOURCE_ID = 2
ABSTRACT_HIDDEN_SOURCE_ID = 3


def _masked_standardize(values: Tensor, mask: Tensor) -> Tensor:
    valid = mask.float()
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * valid).sum(dim=1, keepdim=True) / count
    centered = values - mean
    variance = (centered.square() * valid).sum(dim=1, keepdim=True) / count
    normalized = centered * torch.rsqrt(variance + 1e-4)
    return normalized.masked_fill(~mask.bool(), 0.0)


def _make_mlp_tower(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


@dataclass(slots=True)
class PokeStrategistDecisionConfig:
    state_dim: int = 20
    history_length: int = 16
    token_vocab_size: int = 4096
    species_vocab_size: int = 1024
    move_vocab_size: int = 4096
    hidden_size: int = 128
    plan_count: int = 4
    belief_dim: int = 6
    resource_dim: int = 6
    phase_count: int = 4
    line_count: int = 6
    action_family_count: int = 10
    response_count: int = 6
    future_dim: int = 8
    particle_posterior_dim: int = 10
    reveal_dim: int = 5
    typed_consequence_dim: int = 10
    typed_consequence_bin_count: int = 5
    typed_consequence_interaction_dim: int = 5
    max_legal_actions: int = 32
    candidate_source_count: int = 5
    recoverable_support_weight: float = 0.25
    head_route_weight: float = 0.35
    response_topk: int = 3
    local_score_weight: float = 0.25
    local_tail_weight: float = 0.10
    response_score_weight: float = 0.15
    prior_score_weight: float = 0.10
    rule_score_weight: float = 0.20
    typed_consequence_score_weight: float = 0.15
    typed_interaction_score_weight: float = 0.05
    consequence_value_weight: float = 0.0
    consequence_value_latent_weight: float = 1.0
    consequence_value_move_scale: float = 1.0
    consequence_value_switch_scale: float = 1.0
    switch_subgame_weight: float = 0.20
    switch_follow_discount: float = 0.50
    frontier_base_epsilon: float = 0.15
    frontier_uncertainty_weight: float = 0.60
    frontier_phase_weight: float = 0.25
    frontier_temperature_base: float = 0.75
    use_static_rule_features: bool = True
    species_rule_feature_dim: int = SPECIES_RULE_FEATURE_DIM
    move_rule_feature_dim: int = MOVE_RULE_FEATURE_DIM
    matchup_rule_feature_dim: int = MATCHUP_RULE_FEATURE_DIM
    ability_rule_feature_dim: int = ABILITY_RULE_FEATURE_DIM
    item_rule_feature_dim: int = ITEM_RULE_FEATURE_DIM
    irreversibility_weight: float = 0.0
    candidate_prior_feature_dim: int = 2 + USAGE_PRIOR_FEATURE_DIM
    preview_prior_feature_dim: int = TEAM_PREVIEW_FEATURE_DIM


class _DecoupledStateEncoder(nn.Module):
    def __init__(self, config: PokeStrategistDecisionConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.self_tower = _make_mlp_tower(hidden // 4, hidden)
        self.opp_tower = _make_mlp_tower(hidden // 4, hidden)
        self.history_tower = _make_mlp_tower(hidden // 4, hidden)
        self.numeric_tower = _make_mlp_tower(hidden // 2, hidden)
        self.observation_fuse = _make_mlp_tower(hidden * 4, hidden)

        self.plan_tower = _make_mlp_tower(hidden, hidden)
        self.belief_tower = _make_mlp_tower(hidden, hidden)
        self.resource_tower = _make_mlp_tower(hidden, hidden)
        self.phase_tower = _make_mlp_tower(hidden, hidden)
        self.unlock_tower = _make_mlp_tower(hidden, hidden)
        self.line_tower = _make_mlp_tower(hidden, hidden)
        self.response_tower = _make_mlp_tower(hidden, hidden)
        self.future_tower = _make_mlp_tower(hidden, hidden)
        self.route_tower = _make_mlp_tower(hidden, hidden)
        self.branch_tower = _make_mlp_tower(hidden, hidden)
        self.support_tower = _make_mlp_tower(hidden, hidden)
        self.rule_tower = _make_mlp_tower(hidden, hidden)

        self.decision_context_proj = _make_mlp_tower(hidden * 5, hidden)
        self.line_context_proj = _make_mlp_tower(hidden * 6, hidden)
        self.response_context_proj = _make_mlp_tower(hidden * 3, hidden)
        self.future_context_proj = _make_mlp_tower(hidden * 4, hidden)
        self.route_context_proj = _make_mlp_tower(hidden * 3, hidden)
        self.branch_context_proj = _make_mlp_tower(hidden * 3, hidden)
        self.support_context_proj = _make_mlp_tower(hidden * 3, hidden)
        self.rule_context_proj = _make_mlp_tower(hidden * 2, hidden)

    def forward(
        self,
        *,
        self_species: Tensor,
        opp_species: Tensor,
        history: Tensor,
        numeric: Tensor,
    ) -> TensorMap:
        self_state = self.self_tower(self_species)
        opp_state = self.opp_tower(opp_species)
        history_state = self.history_tower(history)
        numeric_state = self.numeric_tower(numeric)
        observation_state = self.observation_fuse(torch.cat([self_state, opp_state, history_state, numeric_state], dim=-1))

        plan_state = self.plan_tower(observation_state)
        belief_state = self.belief_tower(observation_state)
        resource_state = self.resource_tower(observation_state)
        phase_state = self.phase_tower(observation_state)
        unlock_state = self.unlock_tower(observation_state)
        line_state = self.line_tower(observation_state)
        response_state = self.response_tower(observation_state)
        future_state = self.future_tower(observation_state)
        route_state = self.route_tower(observation_state)
        branch_state = self.branch_tower(observation_state)
        support_state = self.support_tower(observation_state)
        rule_state = self.rule_tower(observation_state)

        return {
            "self_state": self_state,
            "opp_state": opp_state,
            "history_state": history_state,
            "numeric_state": numeric_state,
            "observation_state": observation_state,
            "plan_state": plan_state,
            "belief_state": belief_state,
            "resource_state": resource_state,
            "phase_state": phase_state,
            "unlock_state": unlock_state,
            "line_state": line_state,
            "response_state": response_state,
            "future_state": future_state,
            "route_state": route_state,
            "branch_state": branch_state,
            "support_state": support_state,
            "rule_state": rule_state,
            "decision_context": self.decision_context_proj(
                torch.cat([plan_state, belief_state, resource_state, phase_state, unlock_state], dim=-1)
            ),
            "line_context": self.line_context_proj(
                torch.cat([plan_state, belief_state, resource_state, phase_state, unlock_state, line_state], dim=-1)
            ),
            "response_context": self.response_context_proj(
                torch.cat([plan_state, belief_state, response_state], dim=-1)
            ),
            "future_context": self.future_context_proj(
                torch.cat([plan_state, resource_state, phase_state, future_state], dim=-1)
            ),
            "route_context": self.route_context_proj(
                torch.cat([phase_state, unlock_state, route_state], dim=-1)
            ),
            "branch_context": self.branch_context_proj(
                torch.cat([plan_state, phase_state, branch_state], dim=-1)
            ),
            "support_context": self.support_context_proj(
                torch.cat([belief_state, resource_state, support_state], dim=-1)
            ),
            "rule_context": self.rule_context_proj(
                torch.cat([resource_state, rule_state], dim=-1)
            ),
        }


class _LegacyStateContextProjector(nn.Module):
    def __init__(self, config: PokeStrategistDecisionConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.plan_embedding = nn.Embedding(config.plan_count, hidden)
        self.phase_embedding = nn.Embedding(config.phase_count, hidden)
        self.belief_proj = nn.Linear(config.belief_dim, hidden)
        self.resource_proj = nn.Linear(config.resource_dim, hidden)
        self.context_proj = nn.Sequential(
            nn.Linear(hidden * 5 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, state: TensorMap) -> Tensor:
        plan_context = state["plan_posterior"] @ self.plan_embedding.weight
        phase_context = state["phase_posterior"] @ self.phase_embedding.weight
        belief_context = self.belief_proj(state["belief_summary"])
        resource_context = self.resource_proj(state["resource_ledger"])
        return self.context_proj(
            torch.cat(
                [
                    state["history"],
                    plan_context,
                    belief_context,
                    resource_context,
                    phase_context,
                    state["unlock_score"],
                ],
                dim=-1,
            )
        )


class PokeStrategistDecisionModel(nn.Module):
    def __init__(self, config: PokeStrategistDecisionConfig | None = None) -> None:
        super().__init__()
        self.config = config or PokeStrategistDecisionConfig()
        hidden = self.config.hidden_size

        self.self_species_embedding = nn.Embedding(self.config.species_vocab_size, hidden // 4)
        self.opp_species_embedding = nn.Embedding(self.config.species_vocab_size, hidden // 4)
        if self.config.use_static_rule_features:
            self.self_species_rule_proj = nn.Linear(self.config.species_rule_feature_dim, hidden // 4)
            self.opp_species_rule_proj = nn.Linear(self.config.species_rule_feature_dim, hidden // 4)
            self.self_item_rule_proj = nn.Linear(self.config.item_rule_feature_dim, hidden // 4)
            self.self_ability_rule_proj = nn.Linear(self.config.ability_rule_feature_dim, hidden // 4)
            self.opp_item_rule_proj = nn.Linear(self.config.item_rule_feature_dim, hidden // 4)
            self.opp_ability_rule_proj = nn.Linear(self.config.ability_rule_feature_dim, hidden // 4)
        self.history_embedding = nn.Embedding(self.config.token_vocab_size, hidden // 4)
        self.history_gru = nn.GRU(hidden // 4, hidden // 4, batch_first=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(self.config.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.preview_prior_mlp = nn.Sequential(
            nn.Linear(self.config.preview_prior_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.state_encoder = _DecoupledStateEncoder(self.config)

        self.plan_head = nn.Linear(hidden, self.config.plan_count)
        self.belief_head = nn.Linear(hidden, self.config.belief_dim)
        self.resource_head = nn.Linear(hidden, self.config.resource_dim)
        self.phase_head = nn.Linear(hidden, self.config.phase_count)
        self.line_head = nn.Linear(hidden, self.config.line_count)
        self.unlock_head = nn.Linear(hidden, 1)
        self.response_head = nn.Linear(hidden, self.config.response_count)
        self.win_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.state_future_mean_head = nn.Linear(hidden, self.config.future_dim)
        self.state_future_tail_head = nn.Linear(hidden, self.config.future_dim)
        self.particle_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.particle_posterior_dim),
        )
        self.particle_uncertainty_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.particle_context_proj = nn.Sequential(
            nn.Linear(hidden * 2 + self.config.particle_posterior_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.reveal_likelihood_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.reveal_dim),
        )
        self.plan_value_embedding = nn.Embedding(self.config.plan_count, self.config.future_dim)
        self.plan_tail_embedding = nn.Embedding(self.config.plan_count, self.config.future_dim)
        self.line_value_embedding = nn.Embedding(self.config.line_count, self.config.future_dim)
        self.line_tail_embedding = nn.Embedding(self.config.line_count, self.config.future_dim)
        self.line_head_prior_embedding = nn.Embedding(self.config.line_count, 2)
        self.line_response_value_embedding = nn.Embedding(self.config.line_count, self.config.response_count)
        self.line_consequence_axis_embedding = nn.Embedding(self.config.line_count, self.config.typed_consequence_dim)
        self.line_consequence_interaction_embedding = nn.Embedding(
            self.config.line_count,
            self.config.typed_consequence_interaction_dim,
        )
        nn.init.zeros_(self.line_head_prior_embedding.weight)
        nn.init.zeros_(self.line_response_value_embedding.weight)

        self.action_head_embedding = nn.Embedding(4, hidden // 8)
        self.move_embedding = nn.Embedding(self.config.move_vocab_size, hidden // 4)
        if self.config.use_static_rule_features:
            self.move_rule_proj = nn.Linear(self.config.move_rule_feature_dim, hidden // 4)
        self.family_embedding = nn.Embedding(self.config.action_family_count, hidden // 8)
        self.switch_embedding = nn.Embedding(8, hidden // 8)
        self.tera_embedding = nn.Embedding(2, hidden // 8)
        self.recoverable_embedding = nn.Embedding(2, hidden // 8)
        self.source_embedding = nn.Embedding(self.config.candidate_source_count, hidden // 8)
        self.candidate_species_embedding = nn.Embedding(self.config.species_vocab_size, hidden // 4)
        if self.config.use_static_rule_features:
            self.candidate_species_rule_proj = nn.Linear(self.config.species_rule_feature_dim, hidden // 4)
            self.matchup_rule_proj = nn.Sequential(
                nn.Linear(self.config.matchup_rule_feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        structure_feature_dim = hidden // 8 * 6
        identity_feature_dim = hidden // 4 * 2
        self.structure_tower = nn.Sequential(
            nn.Linear(structure_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.identity_tower = nn.Sequential(
            nn.Linear(identity_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.prior_feature_proj = nn.Sequential(
            nn.Linear(self.config.candidate_prior_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.candidate_meta_proj = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_fusion_norm = nn.LayerNorm(hidden)
        self.imitation_head = nn.Linear(hidden * 2, 1)
        self.q_value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q_temperature = nn.Parameter(torch.tensor(5.0))
        nn.init.constant_(self.q_value_head[-1].bias, -1.0)
        if self.config.use_static_rule_features:
            self.rule_pair_norm = nn.LayerNorm(hidden * 2)
            self.move_rule_value_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.switch_rule_value_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.rule_gate_head = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.response_pair_norm = nn.LayerNorm(hidden * 2)
        self.candidate_response_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.response_count),
        )
        self.prior_gate_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.recoverable_support_head = nn.Linear(hidden * 2, 1)
        self.revealed_move_expert_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.hidden_move_expert_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.abstract_move_expert_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.switch_expert_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.switch_entry_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.future_pair_norm = nn.LayerNorm(hidden * 2)
        self.candidate_future_mean_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, self.config.future_dim))
        self.candidate_future_tail_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, self.config.future_dim))
        self.typed_consequence_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.typed_consequence_dim * self.config.typed_consequence_bin_count),
        )
        self.consequence_value_head = nn.Sequential(
            nn.Linear(self.config.typed_consequence_dim + hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        self.consequence_value_latent_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        nn.init.zeros_(self.consequence_value_head[-1].weight)
        nn.init.zeros_(self.consequence_value_head[-1].bias)
        nn.init.zeros_(self.consequence_value_latent_head[-1].weight)
        nn.init.zeros_(self.consequence_value_latent_head[-1].bias)
        self.switch_follow_mean_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, self.config.future_dim))
        self.switch_follow_tail_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, self.config.future_dim))
        self.local_future_gate_head = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        nn.init.constant_(self.local_future_gate_head[-1].bias, -2.0)
        self.head_router = nn.Linear(hidden, 2)

    def _encode_observation_inputs(self, batch: dict[str, Tensor]) -> TensorMap:
        self_species = self.self_species_embedding(batch["self_species_id"])
        opp_species = self.opp_species_embedding(batch["opp_species_id"])
        if self.config.use_static_rule_features:
            self_species = (
                self_species
                + self.self_species_rule_proj(batch["self_species_rule_features"])
                + self.self_item_rule_proj(batch["self_item_rule_features"])
                + self.self_ability_rule_proj(batch["self_ability_rule_features"])
            )
            opp_species = (
                opp_species
                + self.opp_species_rule_proj(batch["opp_species_rule_features"])
                + self.opp_item_rule_proj(batch["opp_item_rule_features"])
                + self.opp_ability_rule_proj(batch["opp_ability_rule_features"])
            )
        history_tokens = self.history_embedding(batch["history_token_ids"])
        _, history_hidden = self.history_gru(history_tokens)
        numeric_hidden = self.state_mlp(batch["state_numeric"])
        preview_prior_features = batch.get("opp_preview_prior_features")
        if preview_prior_features is None:
            preview_hidden = torch.zeros_like(numeric_hidden)
        else:
            preview_hidden = self.preview_prior_mlp(preview_prior_features)
        numeric_hidden = numeric_hidden + preview_hidden
        return {
            "self_species": self_species,
            "opp_species": opp_species,
            "history": history_hidden.squeeze(0),
            "numeric": numeric_hidden,
            "preview_prior": preview_hidden,
        }

    def _encode_legal_actions(self, batch: dict[str, Tensor]) -> TensorMap:
        head = self.action_head_embedding(batch["legal_action_head_ids"])
        move = self.move_embedding(batch["legal_action_move_ids"])
        if self.config.use_static_rule_features:
            move = move + self.move_rule_proj(batch["legal_action_move_rule_features"])
        family = self.family_embedding(batch["legal_action_family_ids"])
        switch_slot = self.switch_embedding(batch["legal_action_switch_slots"].clamp_min(0).clamp_max(7))
        tera = self.tera_embedding(batch["legal_action_tera_flags"])
        recoverable = self.recoverable_embedding(batch["legal_action_recoverable_flags"].long())
        source = self.source_embedding(batch["legal_action_candidate_source_ids"])
        species = self.candidate_species_embedding(batch["legal_action_species_ids"])
        if self.config.use_static_rule_features:
            species = species + self.candidate_species_rule_proj(batch["legal_action_species_rule_features"])
            matchup = self.matchup_rule_proj(batch["legal_action_matchup_rule_features"])
        else:
            matchup = torch.zeros_like(species.new_zeros(*species.shape[:2], self.config.hidden_size))

        structure_inputs = torch.cat([head, family, switch_slot, tera, recoverable, source], dim=-1)
        structure = self.structure_tower(structure_inputs)
        identity_inputs = torch.cat([move, species], dim=-1)
        identity = self.identity_tower(identity_inputs)
        prior_inputs = torch.cat(
            [
                torch.stack([batch["legal_action_prior_probs"], batch["legal_action_support_scores"]], dim=-1),
                batch["legal_action_external_prior_features"],
            ],
            dim=-1,
        )
        prior = self.prior_feature_proj(prior_inputs)
        meta = self.candidate_meta_proj(
            torch.stack(
                [
                    batch["legal_action_known_move_counts"],
                    batch["legal_action_species_hps"],
                    batch["legal_action_species_status_flags"],
                    batch["legal_action_item_known_flags"],
                    batch["legal_action_ability_known_flags"],
                    batch["legal_action_tera_known_flags"],
                    batch["legal_action_switch_hazard_costs"],
                ],
                dim=-1,
            )
        )
        fused = structure + identity + prior + meta + matchup
        return {
            "structure": structure,
            "identity": identity,
            "prior": prior,
            "meta": meta,
            "matchup": matchup,
            "fused": self.action_fusion_norm(fused),
        }

    def _rule_outputs(self, rule_pair: Tensor, batch: dict[str, Tensor], mask: Tensor) -> TensorMap:
        if not self.config.use_static_rule_features:
            zeros = rule_pair.new_zeros(rule_pair.shape[:2])
            return {"rule_value_scores": zeros, "rule_gate": zeros}

        normalized_rule_pair = self.rule_pair_norm(rule_pair)
        move_rule_scores = self.move_rule_value_head(normalized_rule_pair).squeeze(-1)
        switch_rule_scores = self.switch_rule_value_head(normalized_rule_pair).squeeze(-1)
        raw_scores = torch.where(batch["legal_action_head_ids"].eq(1), switch_rule_scores, move_rule_scores)
        rule_gate = torch.sigmoid(self.rule_gate_head(normalized_rule_pair)).squeeze(-1)
        rule_value_scores = torch.tanh(_masked_standardize(raw_scores, mask)) * rule_gate
        return {
            "rule_value_scores": rule_value_scores,
            "rule_gate": rule_gate.masked_fill(~mask.bool(), 0.0),
        }

    def _candidate_response_outputs(self, response_pair: Tensor, mask: Tensor) -> Tensor:
        normalized_response_pair = self.response_pair_norm(response_pair)
        logits = self.candidate_response_head(normalized_response_pair)
        return logits.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _initial_state(self, state_components: TensorMap) -> TensorMap:
        plan_logits = self.plan_head(state_components["plan_state"])
        phase_logits = self.phase_head(state_components["phase_state"])
        line_logits = self.line_head(state_components["line_context"])
        unlock_logit = self.unlock_head(state_components["unlock_state"])
        belief_summary = self.belief_head(state_components["belief_state"]).sigmoid()
        resource_ledger = self.resource_head(state_components["resource_state"])
        return {
            "history": state_components["observation_state"],
            "plan_logits": plan_logits,
            "plan_posterior": plan_logits.softmax(dim=-1),
            "line_logits": line_logits,
            "line_posterior": line_logits.softmax(dim=-1),
            "belief_summary": belief_summary,
            "resource_ledger": resource_ledger,
            "phase_logits": phase_logits,
            "phase_posterior": phase_logits.softmax(dim=-1),
            "unlock_logit": unlock_logit,
            "unlock_score": unlock_logit.sigmoid(),
            "decision_context": state_components["decision_context"],
            "line_context": state_components["line_context"],
            "response_context": state_components["response_context"],
            "future_context": state_components["future_context"],
            "route_context": state_components["route_context"],
            "branch_context": state_components["branch_context"],
            "support_context": state_components["support_context"],
            "rule_context": state_components["rule_context"],
            "self_state": state_components["self_state"],
            "opp_state": state_components["opp_state"],
            "history_component": state_components["history_state"],
            "numeric_state": state_components["numeric_state"],
            "plan_state": state_components["plan_state"],
            "belief_state": state_components["belief_state"],
            "resource_state": state_components["resource_state"],
            "phase_state": state_components["phase_state"],
            "unlock_state": state_components["unlock_state"],
            "line_state": state_components["line_state"],
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        observation_inputs = self._encode_observation_inputs(batch)
        state_components = self.state_encoder(
            self_species=observation_inputs["self_species"],
            opp_species=observation_inputs["opp_species"],
            history=observation_inputs["history"],
            numeric=observation_inputs["numeric"],
        )
        initial_state = self._initial_state(state_components)
        particle_inputs = torch.cat([initial_state["belief_state"], initial_state["support_context"]], dim=-1)
        particle_posterior = torch.sigmoid(self.particle_head(particle_inputs))
        particle_uncertainty = torch.sigmoid(self.particle_uncertainty_head(particle_inputs))
        particle_context = self.particle_context_proj(
            torch.cat([initial_state["belief_state"], initial_state["support_context"], particle_posterior, particle_uncertainty], dim=-1)
        )
        reveal_likelihood_logits = self.reveal_likelihood_head(particle_context)

        state_context = initial_state["decision_context"] + particle_context
        line_context = initial_state["line_context"] + particle_context
        route_context = initial_state["route_context"] + particle_context
        response_context = initial_state["response_context"] + particle_context
        future_context = initial_state["future_context"] + particle_context
        branch_context = initial_state["branch_context"] + particle_context
        support_context = initial_state["support_context"] + particle_context
        rule_context = initial_state["rule_context"] + particle_context

        legal_action_components = self._encode_legal_actions(batch)
        legal_action_latents = legal_action_components["fused"]
        action_state = state_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        action_pair = torch.cat([legal_action_latents, action_state], dim=-1)
        branch_state = branch_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        support_state = support_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        future_state = future_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        rule_state = rule_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        response_state = response_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        structure_pair = torch.cat([legal_action_components["structure"] + legal_action_components["identity"], branch_state], dim=-1)
        prior_pair = torch.cat([legal_action_components["prior"] + legal_action_components["identity"], support_state], dim=-1)
        meta_pair = torch.cat([legal_action_components["meta"] + legal_action_components["identity"], future_state], dim=-1)
        rule_pair = torch.cat([legal_action_components["matchup"] + legal_action_components["identity"], rule_state], dim=-1)
        response_pair = torch.cat([legal_action_components["structure"] + legal_action_components["matchup"], response_state], dim=-1)
        legal_action_mask = batch["legal_action_mask"].bool()

        head_logits = self.head_router(route_context)
        response_logits = self.response_head(response_context)
        win_logit = self.win_head(state_context).squeeze(-1)
        state_future_mean = self.state_future_mean_head(future_context)
        state_future_tail = -F.softplus(self.state_future_tail_head(future_context))
        future_action_pair = self.future_pair_norm(meta_pair.detach())
        candidate_future_mean = self.candidate_future_mean_head(future_action_pair)
        candidate_future_tail = -F.softplus(self.candidate_future_tail_head(future_action_pair))
        typed_consequence_logits = self.typed_consequence_head(future_action_pair).view(
            future_action_pair.size(0),
            future_action_pair.size(1),
            self.config.typed_consequence_dim,
            self.config.typed_consequence_bin_count,
        )
        plan_future_value = initial_state["plan_posterior"] @ self.plan_value_embedding.weight
        plan_future_tail = F.softplus(initial_state["plan_posterior"] @ self.plan_tail_embedding.weight)
        line_future_value = initial_state["line_posterior"] @ self.line_value_embedding.weight
        line_future_tail = F.softplus(initial_state["line_posterior"] @ self.line_tail_embedding.weight)
        blended_future_value = 0.75 * plan_future_value + 0.25 * line_future_value
        blended_future_tail = 0.75 * plan_future_tail + 0.25 * line_future_tail
        typed_consequence_probs = typed_consequence_logits.softmax(dim=-1)
        consequence_bin_values = typed_consequence_logits.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        typed_consequence_values = (typed_consequence_probs * consequence_bin_values).sum(dim=-1)
        typed_consequence_interactions = torch.stack(
            [
                typed_consequence_values[..., 1] * typed_consequence_values[..., 2],
                typed_consequence_values[..., 3] * typed_consequence_values[..., 8],
                typed_consequence_values[..., 6] * typed_consequence_values[..., 9],
                typed_consequence_values[..., 4] * typed_consequence_values[..., 5],
                typed_consequence_values[..., 7] * typed_consequence_values[..., 8],
            ],
            dim=-1,
        )
        line_axis_values = initial_state["line_posterior"] @ self.line_consequence_axis_embedding.weight
        line_interaction_values = initial_state["line_posterior"] @ self.line_consequence_interaction_embedding.weight
        typed_consequence_scores = (typed_consequence_values * line_axis_values.unsqueeze(1)).sum(dim=-1)
        typed_consequence_scores = torch.tanh(_masked_standardize(typed_consequence_scores, legal_action_mask))
        typed_interaction_scores = (typed_consequence_interactions * line_interaction_values.unsqueeze(1)).sum(dim=-1)
        typed_interaction_scores = torch.tanh(_masked_standardize(typed_interaction_scores, legal_action_mask))
        # Consequence value calibration: state-conditioned consequence → scalar value
        conseq_value_inputs = torch.cat(
            [typed_consequence_values.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0),
             state_context.unsqueeze(1).expand(-1, typed_consequence_values.size(1), -1)],
            dim=-1,
        )
        consequence_value_base_scores = self.consequence_value_head(conseq_value_inputs).squeeze(-1)
        consequence_value_latent_scores = self.consequence_value_latent_head(future_action_pair).squeeze(-1)
        consequence_value_scores = torch.tanh(
            consequence_value_base_scores + self.config.consequence_value_latent_weight * consequence_value_latent_scores
        ).masked_fill(~legal_action_mask, 0.0)
        consequence_value_action_scales = torch.where(
            batch["legal_action_head_ids"].eq(1),
            consequence_value_scores.new_tensor(float(self.config.consequence_value_switch_scale)),
            consequence_value_scores.new_tensor(float(self.config.consequence_value_move_scale)),
        ).masked_fill(~legal_action_mask, 0.0)
        consequence_value_residual_scores = consequence_value_scores * consequence_value_action_scales
        local_future_scores = (candidate_future_mean * blended_future_value.unsqueeze(1)).sum(dim=-1)
        local_tail_penalty = (candidate_future_tail.abs() * blended_future_tail.unsqueeze(1)).sum(dim=-1)
        local_future_gate = torch.sigmoid(self.local_future_gate_head(future_action_pair)).squeeze(-1)
        local_future_scores = torch.tanh(_masked_standardize(local_future_scores, legal_action_mask)) * local_future_gate
        local_tail_penalty = torch.tanh(_masked_standardize(local_tail_penalty, legal_action_mask)) * local_future_gate

        imitation_logits = self.imitation_head(action_pair).squeeze(-1)
        q_values = self.q_value_head(action_pair).squeeze(-1)
        rule_outputs = self._rule_outputs(rule_pair, batch, legal_action_mask)
        candidate_response_logits = self._candidate_response_outputs(response_pair, legal_action_mask)
        candidate_response_probs = candidate_response_logits.softmax(dim=-1)
        line_response_values = initial_state["line_posterior"] @ self.line_response_value_embedding.weight
        response_value_scores = (candidate_response_probs * line_response_values.unsqueeze(1)).sum(dim=-1)
        response_value_scores = torch.tanh(_masked_standardize(response_value_scores, legal_action_mask))
        source_ids = batch["legal_action_candidate_source_ids"]
        hidden_prior_mask = legal_action_mask & (
            source_ids.eq(PRIOR_HIDDEN_SOURCE_ID) | source_ids.eq(ABSTRACT_HIDDEN_SOURCE_ID)
        )
        prior_source_gate = torch.sigmoid(self.prior_gate_head(prior_pair)).squeeze(-1)
        prior_mix_raw = (
            prior_source_gate * batch["legal_action_prior_probs"]
            + (1.0 - prior_source_gate) * batch["legal_action_usage_scores"]
        )
        prior_mix_scores = torch.tanh(_masked_standardize(prior_mix_raw, hidden_prior_mask))
        raw_recoverable_support_logits = self.recoverable_support_head(prior_pair).squeeze(-1)
        recoverable_support = raw_recoverable_support_logits * batch["legal_action_recoverable_flags"].float()

        revealed_move_branch_logits = self.revealed_move_expert_head(structure_pair).squeeze(-1)
        hidden_move_branch_logits = self.hidden_move_expert_head(structure_pair).squeeze(-1)
        abstract_move_branch_logits = self.abstract_move_expert_head(structure_pair).squeeze(-1)
        switch_branch_logits = self.switch_expert_head(structure_pair).squeeze(-1)
        move_branch_logits = torch.where(
            source_ids.eq(PRIOR_HIDDEN_SOURCE_ID),
            hidden_move_branch_logits,
            revealed_move_branch_logits,
        )
        move_branch_logits = torch.where(
            source_ids.eq(ABSTRACT_HIDDEN_SOURCE_ID),
            abstract_move_branch_logits,
            move_branch_logits,
        )
        branch_logits = torch.where(batch["legal_action_head_ids"].eq(1), switch_branch_logits, move_branch_logits)

        head_log_probs = head_logits.log_softmax(dim=-1)
        move_route_bias = head_log_probs[:, 0].unsqueeze(1)
        switch_route_bias = head_log_probs[:, 1].unsqueeze(1)
        line_head_prior = initial_state["line_posterior"] @ self.line_head_prior_embedding.weight
        line_move_bias = line_head_prior[:, 0].unsqueeze(1)
        line_switch_bias = line_head_prior[:, 1].unsqueeze(1)
        route_bias = torch.where(batch["legal_action_head_ids"].eq(1), switch_route_bias, move_route_bias)
        route_bias = route_bias + torch.where(batch["legal_action_head_ids"].eq(1), line_switch_bias, line_move_bias)

        switch_action_mask = legal_action_mask & batch["legal_action_head_ids"].eq(1)
        switch_future_pair = self.future_pair_norm(
            torch.cat([legal_action_components["meta"] + legal_action_components["matchup"], future_state], dim=-1).detach()
        )
        switch_entry_value = self.switch_entry_head(switch_future_pair).squeeze(-1) - batch["legal_action_switch_hazard_costs"]
        switch_entry_scores = torch.tanh(_masked_standardize(switch_entry_value, switch_action_mask))
        switch_follow_mean = self.switch_follow_mean_head(switch_future_pair)
        switch_follow_tail = -F.softplus(self.switch_follow_tail_head(switch_future_pair))
        switch_follow_value = (switch_follow_mean * blended_future_value.unsqueeze(1)).sum(dim=-1)
        switch_follow_risk = (switch_follow_tail.abs() * blended_future_tail.unsqueeze(1)).sum(dim=-1)
        switch_follow_scores = torch.tanh(
            _masked_standardize(switch_follow_value - 0.5 * switch_follow_risk, switch_action_mask)
        )
        switch_subgame_scores = (switch_entry_scores + self.config.switch_follow_discount * switch_follow_scores).masked_fill(
            ~switch_action_mask,
            0.0,
        )

        legal_action_scores = (
            imitation_logits
            + self.config.recoverable_support_weight * recoverable_support
            + branch_logits
            + self.config.head_route_weight * route_bias
            + self.config.prior_score_weight * prior_mix_scores
            + self.config.rule_score_weight * rule_outputs["rule_value_scores"]
            + self.config.response_score_weight * response_value_scores
            + self.config.local_score_weight * local_future_scores
            + self.config.typed_consequence_score_weight * typed_consequence_scores
            + self.config.typed_interaction_score_weight * typed_interaction_scores
            + self.config.consequence_value_weight * consequence_value_residual_scores
            + self.config.switch_subgame_weight * switch_subgame_scores
            - self.config.local_tail_weight * local_tail_penalty
        )

        imitation_logits = imitation_logits.masked_fill(~legal_action_mask, -1e9)
        recoverable_support_logits = raw_recoverable_support_logits.masked_fill(~batch["legal_action_recoverable_flags"].bool(), -1e9)
        move_branch_logits = move_branch_logits.masked_fill(~legal_action_mask, -1e9)
        switch_branch_logits = switch_branch_logits.masked_fill(~legal_action_mask, -1e9)
        branch_logits = branch_logits.masked_fill(~legal_action_mask, -1e9)
        route_bias = route_bias.masked_fill(~legal_action_mask, -1e9)
        typed_consequence_logits = typed_consequence_logits.masked_fill(~legal_action_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        legal_action_scores = legal_action_scores.masked_fill(~legal_action_mask, -1e9)
        belief_uncertainty = (1.0 - initial_state["belief_summary"]).mean(dim=-1)
        combined_uncertainty = 0.5 * (belief_uncertainty + particle_uncertainty.squeeze(-1))
        phase_frontier_pressure = initial_state["phase_posterior"][:, 0] + 0.5 * initial_state["phase_posterior"][:, 1]
        adaptive_epsilon = torch.clamp(
            self.config.frontier_base_epsilon
            + self.config.frontier_uncertainty_weight * combined_uncertainty
            + self.config.frontier_phase_weight * phase_frontier_pressure,
            min=0.05,
            max=1.5,
        )
        best_scores = legal_action_scores.max(dim=-1).values
        frontier_threshold = best_scores - adaptive_epsilon
        frontier_mask = legal_action_mask & legal_action_scores.ge(frontier_threshold.unsqueeze(-1))
        frontier_temperature = torch.clamp(self.config.frontier_temperature_base + combined_uncertainty, min=0.5, max=2.0)
        frontier_policy_logits = (legal_action_scores / frontier_temperature.unsqueeze(-1)).masked_fill(~frontier_mask, -1e9)
        frontier_policy_probs = frontier_policy_logits.softmax(dim=-1)

        return {
            "history_state": initial_state["history"],
            "state": state_context,
            "plan_logits": initial_state["plan_logits"],
            "plan_posterior": initial_state["plan_posterior"],
            "line_logits": initial_state["line_logits"],
            "line_posterior": initial_state["line_posterior"],
            "particle_posterior": particle_posterior,
            "particle_uncertainty": particle_uncertainty.squeeze(-1),
            "reveal_likelihood_logits": reveal_likelihood_logits,
            "reveal_likelihood": reveal_likelihood_logits.sigmoid(),
            "belief_summary": initial_state["belief_summary"],
            "belief_uncertainty": belief_uncertainty,
            "resource_ledger": initial_state["resource_ledger"],
            "phase_logits": initial_state["phase_logits"],
            "phase_posterior": initial_state["phase_posterior"],
            "unlock_logit": initial_state["unlock_logit"].squeeze(-1),
            "unlock_score": initial_state["unlock_score"].squeeze(-1),
            "response_logits": response_logits,
            "win_logit": win_logit,
            "state_future_mean": state_future_mean,
            "state_future_tail": state_future_tail,
            "candidate_future_mean": candidate_future_mean,
            "candidate_future_tail": candidate_future_tail,
            "typed_consequence_logits": typed_consequence_logits,
            "typed_consequence_values": typed_consequence_values.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0),
            "typed_consequence_interactions": typed_consequence_interactions.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0),
            "typed_consequence_scores": typed_consequence_scores.masked_fill(~legal_action_mask, 0.0),
            "typed_interaction_scores": typed_interaction_scores.masked_fill(~legal_action_mask, 0.0),
            "consequence_value_scores": consequence_value_scores,
            "consequence_value_base_scores": consequence_value_base_scores.masked_fill(~legal_action_mask, 0.0),
            "consequence_value_latent_scores": consequence_value_latent_scores.masked_fill(~legal_action_mask, 0.0),
            "consequence_value_action_scales": consequence_value_action_scales,
            "consequence_value_residual_scores": consequence_value_residual_scores,
            "local_future_scores": local_future_scores,
            "local_tail_penalty": local_tail_penalty,
            "local_future_gate": local_future_gate.masked_fill(~legal_action_mask, 0.0),
            "candidate_structure_latents": legal_action_components["structure"],
            "candidate_identity_latents": legal_action_components["identity"],
            "candidate_prior_latents": legal_action_components["prior"],
            "candidate_meta_latents": legal_action_components["meta"],
            "candidate_matchup_latents": legal_action_components["matchup"],
            "legal_action_latents": legal_action_latents,
            "state_self_latents": initial_state["self_state"],
            "state_opp_latents": initial_state["opp_state"],
            "state_history_latents": initial_state["history_component"],
            "state_numeric_latents": initial_state["numeric_state"],
            "state_preview_prior_latents": observation_inputs["preview_prior"],
            "state_plan_latents": initial_state["plan_state"],
            "state_belief_latents": initial_state["belief_state"],
            "state_resource_latents": initial_state["resource_state"],
            "state_phase_latents": initial_state["phase_state"],
            "state_unlock_latents": initial_state["unlock_state"],
            "state_line_latents": initial_state["line_state"],
            "state_decision_context": state_context,
            "state_line_context": line_context,
            "state_route_context": route_context,
            "state_response_context": response_context,
            "state_future_context": future_context,
            "state_branch_context": branch_context,
            "state_support_context": support_context,
            "state_rule_context": rule_context,
            "state_particle_context": particle_context,
            "state_consequence_mean": state_future_mean,
            "state_consequence_tail": state_future_tail,
            "candidate_consequence_mean": candidate_future_mean,
            "candidate_consequence_tail": candidate_future_tail,
            "head_logits": head_logits,
            "imitation_logits": imitation_logits,
            "q_values": torch.sigmoid(
                (legal_action_scores.detach() + q_values) / F.softplus(self.q_temperature)
            ).masked_fill(~legal_action_mask, -1e9),
            "win_logit": torch.sigmoid(win_logit).unsqueeze(-1),
            "rule_value_scores": rule_outputs["rule_value_scores"].masked_fill(~legal_action_mask, 0.0),
            "rule_gate": rule_outputs["rule_gate"],
            "candidate_response_logits": candidate_response_logits,
            "candidate_response_probs": candidate_response_probs.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0),
            "response_value_scores": response_value_scores.masked_fill(~legal_action_mask, 0.0),
            "prior_source_gate": prior_source_gate.masked_fill(~hidden_prior_mask, 0.0),
            "prior_mix_scores": prior_mix_scores.masked_fill(~hidden_prior_mask, 0.0),
            "recoverable_support_logits": recoverable_support_logits,
            "move_branch_logits": move_branch_logits,
            "switch_branch_logits": switch_branch_logits,
            "switch_entry_value": switch_entry_value.masked_fill(~switch_action_mask, 0.0),
            "switch_entry_scores": switch_entry_scores.masked_fill(~switch_action_mask, 0.0),
            "switch_follow_mean": switch_follow_mean,
            "switch_follow_tail": switch_follow_tail,
            "switch_follow_scores": switch_follow_scores.masked_fill(~switch_action_mask, 0.0),
            "switch_subgame_scores": switch_subgame_scores,
            "branch_logits": branch_logits,
            "route_bias": route_bias,
            "adaptive_epsilon": adaptive_epsilon,
            "frontier_threshold": frontier_threshold,
            "frontier_mask": frontier_mask,
            "frontier_policy_logits": frontier_policy_logits,
            "frontier_policy_probs": frontier_policy_probs,
            "legal_action_scores": legal_action_scores,
            "legal_action_mask": legal_action_mask,
        }


class _LegacyStateBaseDecisionModel(PokeStrategistDecisionModel):
    def __init__(self, config: PokeStrategistDecisionConfig | None = None) -> None:
        super().__init__(config)
        hidden = self.config.hidden_size
        self.history_encoder = nn.Sequential(
            nn.Linear(hidden // 4 + hidden // 4 + hidden // 4 + hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.state_projector = _LegacyStateContextProjector(self.config)

    def _encode_history_legacy(self, batch: dict[str, Tensor]) -> Tensor:
        self_species = self.self_species_embedding(batch["self_species_id"])
        opp_species = self.opp_species_embedding(batch["opp_species_id"])
        if self.config.use_static_rule_features:
            self_species = self_species + self.self_species_rule_proj(batch["self_species_rule_features"])
            opp_species = opp_species + self.opp_species_rule_proj(batch["opp_species_rule_features"])
        history_tokens = self.history_embedding(batch["history_token_ids"])
        _, history_hidden = self.history_gru(history_tokens)
        numeric_hidden = self.state_mlp(batch["state_numeric"])
        combined = torch.cat([self_species, opp_species, history_hidden.squeeze(0), numeric_hidden], dim=-1)
        return self.history_encoder(combined)

    def _initial_state_legacy(self, history_state: Tensor) -> TensorMap:
        plan_logits = self.plan_head(history_state)
        phase_logits = self.phase_head(history_state)
        unlock_logit = self.unlock_head(history_state)
        return {
            "history": history_state,
            "plan_logits": plan_logits,
            "plan_posterior": plan_logits.softmax(dim=-1),
            "belief_summary": self.belief_head(history_state).sigmoid(),
            "resource_ledger": self.resource_head(history_state),
            "phase_logits": phase_logits,
            "phase_posterior": phase_logits.softmax(dim=-1),
            "unlock_logit": unlock_logit,
            "unlock_score": unlock_logit.sigmoid(),
        }

    def _legacy_rule_scores(self, rule_pair: Tensor, batch: dict[str, Tensor], mask: Tensor) -> Tensor:
        if not self.config.use_static_rule_features:
            return rule_pair.new_zeros(rule_pair.shape[:2])

        normalized_rule_pair = self.rule_pair_norm(rule_pair)
        move_rule_scores = self.move_rule_value_head(normalized_rule_pair).squeeze(-1)
        switch_rule_scores = self.switch_rule_value_head(normalized_rule_pair).squeeze(-1)
        raw_scores = torch.where(batch["legal_action_head_ids"].eq(1), switch_rule_scores, move_rule_scores)
        rule_gate = torch.sigmoid(self.rule_gate_head(normalized_rule_pair)).squeeze(-1)
        return torch.tanh(_masked_standardize(raw_scores, mask)) * rule_gate


class _LegacyStateDecisionModel(_LegacyStateBaseDecisionModel):
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        history_state = self._encode_history_legacy(batch)
        initial_state = self._initial_state_legacy(history_state)
        state_context = self.state_projector(initial_state)

        legal_action_components = self._encode_legal_actions(batch)
        legal_action_latents = legal_action_components["fused"]
        action_state = state_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        action_pair = torch.cat([legal_action_latents, action_state], dim=-1)
        structure_pair = torch.cat([legal_action_components["structure"] + legal_action_components["identity"], action_state], dim=-1)
        prior_pair = torch.cat([legal_action_components["prior"] + legal_action_components["identity"], action_state], dim=-1)
        meta_pair = torch.cat([legal_action_components["meta"] + legal_action_components["identity"], action_state], dim=-1)
        rule_pair = torch.cat([legal_action_components["matchup"] + legal_action_components["identity"], action_state], dim=-1)
        legal_action_mask = batch["legal_action_mask"].bool()

        head_logits = self.head_router(state_context)
        state_future_mean = self.state_future_mean_head(state_context)
        future_action_pair = self.future_pair_norm(meta_pair.detach())
        candidate_future_mean = self.candidate_future_mean_head(future_action_pair)
        candidate_future_tail = -F.softplus(self.candidate_future_tail_head(future_action_pair))
        plan_future_value = initial_state["plan_posterior"] @ self.plan_value_embedding.weight
        plan_future_tail = F.softplus(initial_state["plan_posterior"] @ self.plan_tail_embedding.weight)
        local_future_scores = (candidate_future_mean * plan_future_value.unsqueeze(1)).sum(dim=-1)
        local_tail_penalty = (candidate_future_tail.abs() * plan_future_tail.unsqueeze(1)).sum(dim=-1)
        local_future_gate = torch.sigmoid(self.local_future_gate_head(future_action_pair)).squeeze(-1)
        local_future_scores = torch.tanh(_masked_standardize(local_future_scores, legal_action_mask)) * local_future_gate
        local_tail_penalty = torch.tanh(_masked_standardize(local_tail_penalty, legal_action_mask)) * local_future_gate

        imitation_logits = self.imitation_head(action_pair).squeeze(-1)
        rule_value_scores = self._legacy_rule_scores(rule_pair, batch, legal_action_mask)
        raw_recoverable_support_logits = self.recoverable_support_head(prior_pair).squeeze(-1)
        recoverable_support = raw_recoverable_support_logits * batch["legal_action_recoverable_flags"].float()

        revealed_move_branch_logits = self.revealed_move_expert_head(structure_pair).squeeze(-1)
        hidden_move_branch_logits = self.hidden_move_expert_head(structure_pair).squeeze(-1)
        abstract_move_branch_logits = self.abstract_move_expert_head(structure_pair).squeeze(-1)
        switch_branch_logits = self.switch_expert_head(structure_pair).squeeze(-1)
        source_ids = batch["legal_action_candidate_source_ids"]
        move_branch_logits = torch.where(
            source_ids.eq(PRIOR_HIDDEN_SOURCE_ID),
            hidden_move_branch_logits,
            revealed_move_branch_logits,
        )
        move_branch_logits = torch.where(
            source_ids.eq(ABSTRACT_HIDDEN_SOURCE_ID),
            abstract_move_branch_logits,
            move_branch_logits,
        )
        branch_logits = torch.where(batch["legal_action_head_ids"].eq(1), switch_branch_logits, move_branch_logits)

        head_log_probs = head_logits.log_softmax(dim=-1)
        move_route_bias = head_log_probs[:, 0].unsqueeze(1)
        switch_route_bias = head_log_probs[:, 1].unsqueeze(1)
        route_bias = torch.where(batch["legal_action_head_ids"].eq(1), switch_route_bias, move_route_bias)

        legal_action_scores = (
            imitation_logits
            + self.config.recoverable_support_weight * recoverable_support
            + branch_logits
            + self.config.head_route_weight * route_bias
            + self.config.rule_score_weight * rule_value_scores
            + self.config.local_score_weight * local_future_scores
            - self.config.local_tail_weight * local_tail_penalty
        )
        legal_action_scores = legal_action_scores.masked_fill(~legal_action_mask, -1e9)
        return {
            "head_logits": head_logits,
            "state_future_mean": state_future_mean,
            "legal_action_scores": legal_action_scores,
        }


class _LegacyFullDecisionModel(_LegacyStateBaseDecisionModel):
    def __init__(self, config: PokeStrategistDecisionConfig | None = None) -> None:
        super().__init__(config)
        hidden = self.config.hidden_size
        reduced = hidden // 8
        intermediate = hidden // 4
        self.action_encoder = nn.Sequential(
            nn.Linear(reduced * 9 + intermediate * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.prior_feature_proj = nn.Sequential(
            nn.Linear(self.config.candidate_prior_feature_dim, reduced),
            nn.ReLU(),
            nn.Linear(reduced, reduced),
        )
        self.candidate_meta_proj = nn.Sequential(
            nn.Linear(7, reduced),
            nn.ReLU(),
            nn.Linear(reduced, reduced),
        )
        if self.config.use_static_rule_features:
            self.matchup_rule_proj = nn.Sequential(
                nn.Linear(self.config.matchup_rule_feature_dim, intermediate),
                nn.ReLU(),
                nn.Linear(intermediate, reduced),
            )

    def _encode_legacy_legal_actions(self, batch: dict[str, Tensor]) -> Tensor:
        head = self.action_head_embedding(batch["legal_action_head_ids"])
        move = self.move_embedding(batch["legal_action_move_ids"])
        if self.config.use_static_rule_features:
            move = move + self.move_rule_proj(batch["legal_action_move_rule_features"])
        family = self.family_embedding(batch["legal_action_family_ids"])
        switch_slot = self.switch_embedding(batch["legal_action_switch_slots"].clamp_min(0).clamp_max(7))
        tera = self.tera_embedding(batch["legal_action_tera_flags"])
        recoverable = self.recoverable_embedding(batch["legal_action_recoverable_flags"].long())
        source = self.source_embedding(batch["legal_action_candidate_source_ids"])
        species = self.candidate_species_embedding(batch["legal_action_species_ids"])
        if self.config.use_static_rule_features:
            species = species + self.candidate_species_rule_proj(batch["legal_action_species_rule_features"])
            matchup = self.matchup_rule_proj(batch["legal_action_matchup_rule_features"])
        else:
            matchup = species.new_zeros(*species.shape[:2], self.config.hidden_size // 8)
        prior_inputs = torch.cat(
            [
                torch.stack([batch["legal_action_prior_probs"], batch["legal_action_support_scores"]], dim=-1),
                batch["legal_action_external_prior_features"],
            ],
            dim=-1,
        )
        prior = self.prior_feature_proj(prior_inputs)
        meta = self.candidate_meta_proj(
            torch.stack(
                [
                    batch["legal_action_known_move_counts"],
                    batch["legal_action_species_hps"],
                    batch["legal_action_species_status_flags"],
                    batch["legal_action_item_known_flags"],
                    batch["legal_action_ability_known_flags"],
                    batch["legal_action_tera_known_flags"],
                    batch["legal_action_switch_hazard_costs"],
                ],
                dim=-1,
            )
        )
        action_inputs = torch.cat(
            [head, move, family, switch_slot, tera, recoverable, source, species, prior, meta, matchup],
            dim=-1,
        )
        return self.action_encoder(action_inputs)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        history_state = self._encode_history_legacy(batch)
        initial_state = self._initial_state_legacy(history_state)
        state_context = self.state_projector(initial_state)

        legal_action_latents = self._encode_legacy_legal_actions(batch)
        action_state = state_context.unsqueeze(1).expand(-1, legal_action_latents.size(1), -1)
        action_pair = torch.cat([legal_action_latents, action_state], dim=-1)
        legal_action_mask = batch["legal_action_mask"].bool()

        head_logits = self.head_router(state_context)
        state_future_mean = self.state_future_mean_head(state_context)
        future_action_pair = self.future_pair_norm(action_pair.detach())
        candidate_future_mean = self.candidate_future_mean_head(future_action_pair)
        candidate_future_tail = -F.softplus(self.candidate_future_tail_head(future_action_pair))
        plan_future_value = initial_state["plan_posterior"] @ self.plan_value_embedding.weight
        plan_future_tail = F.softplus(initial_state["plan_posterior"] @ self.plan_tail_embedding.weight)
        local_future_scores = (candidate_future_mean * plan_future_value.unsqueeze(1)).sum(dim=-1)
        local_tail_penalty = (candidate_future_tail.abs() * plan_future_tail.unsqueeze(1)).sum(dim=-1)
        local_future_gate = torch.sigmoid(self.local_future_gate_head(future_action_pair)).squeeze(-1)
        local_future_scores = torch.tanh(_masked_standardize(local_future_scores, legal_action_mask)) * local_future_gate
        local_tail_penalty = torch.tanh(_masked_standardize(local_tail_penalty, legal_action_mask)) * local_future_gate

        imitation_logits = self.imitation_head(action_pair).squeeze(-1)
        rule_value_scores = self._legacy_rule_scores(action_pair, batch, legal_action_mask)
        raw_recoverable_support_logits = self.recoverable_support_head(action_pair).squeeze(-1)
        recoverable_support = raw_recoverable_support_logits * batch["legal_action_recoverable_flags"].float()

        revealed_move_branch_logits = self.revealed_move_expert_head(action_pair).squeeze(-1)
        hidden_move_branch_logits = self.hidden_move_expert_head(action_pair).squeeze(-1)
        abstract_move_branch_logits = self.abstract_move_expert_head(action_pair).squeeze(-1)
        switch_branch_logits = self.switch_expert_head(action_pair).squeeze(-1)
        source_ids = batch["legal_action_candidate_source_ids"]
        move_branch_logits = torch.where(
            source_ids.eq(PRIOR_HIDDEN_SOURCE_ID),
            hidden_move_branch_logits,
            revealed_move_branch_logits,
        )
        move_branch_logits = torch.where(
            source_ids.eq(ABSTRACT_HIDDEN_SOURCE_ID),
            abstract_move_branch_logits,
            move_branch_logits,
        )
        branch_logits = torch.where(batch["legal_action_head_ids"].eq(1), switch_branch_logits, move_branch_logits)

        head_log_probs = head_logits.log_softmax(dim=-1)
        move_route_bias = head_log_probs[:, 0].unsqueeze(1)
        switch_route_bias = head_log_probs[:, 1].unsqueeze(1)
        route_bias = torch.where(batch["legal_action_head_ids"].eq(1), switch_route_bias, move_route_bias)

        legal_action_scores = (
            imitation_logits
            + self.config.recoverable_support_weight * recoverable_support
            + branch_logits
            + self.config.head_route_weight * route_bias
            + self.config.rule_score_weight * rule_value_scores
            + self.config.local_score_weight * local_future_scores
            - self.config.local_tail_weight * local_tail_penalty
        )
        legal_action_scores = legal_action_scores.masked_fill(~legal_action_mask, -1e9)
        return {
            "head_logits": head_logits,
            "state_future_mean": state_future_mean,
            "legal_action_scores": legal_action_scores,
        }


def _validate_compatible_load(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    allowed_missing_prefixes: tuple[str, ...],
) -> None:
    disallowed_missing = [
        key for key in missing_keys if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            "Incompatible checkpoint load: "
            f"missing={disallowed_missing or '[]'}, unexpected={unexpected_keys or '[]'}"
        )


def build_model_for_checkpoint(
    config: PokeStrategistDecisionConfig,
    model_state: dict[str, Tensor],
) -> nn.Module:
    if any(key.startswith("state_encoder.") for key in model_state):
        model = PokeStrategistDecisionModel(config)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _validate_compatible_load(
            missing_keys,
            unexpected_keys,
            allowed_missing_prefixes=(
                "state_encoder.line_",
                "line_",
                "self_item_rule_proj.",
                "self_ability_rule_proj.",
                "opp_item_rule_proj.",
                "opp_ability_rule_proj.",
                "particle_",
                "reveal_likelihood_head.",
                "typed_consequence_head.",
                "prior_gate_head.",
                "preview_prior_mlp.",
                "switch_entry_head.",
                "switch_follow_mean_head.",
                "switch_follow_tail_head.",
                "win_head.",
                "q_value_head.",
                "q_temperature",
                "consequence_value_head.",
                "consequence_value_latent_head.",
            ),
        )
        return model

    if any(key.startswith("structure_tower.") for key in model_state):
        model = _LegacyStateDecisionModel(config)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _validate_compatible_load(
            missing_keys,
            unexpected_keys,
            allowed_missing_prefixes=(
                "state_encoder.",
                "response_pair_norm.",
                "line_",
                "self_item_rule_proj.",
                "self_ability_rule_proj.",
                "opp_item_rule_proj.",
                "opp_ability_rule_proj.",
                "particle_",
                "reveal_likelihood_head.",
                "typed_consequence_head.",
                "prior_gate_head.",
                "preview_prior_mlp.",
                "switch_entry_head.",
                "switch_follow_mean_head.",
                "switch_follow_tail_head.",
                "win_head.",
                "q_value_head.",
                "q_temperature",
                "consequence_value_head.",
                "consequence_value_latent_head.",
            ),
        )
        return model

    model = _LegacyFullDecisionModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    _validate_compatible_load(
        missing_keys,
        unexpected_keys,
        allowed_missing_prefixes=(
            "state_encoder.",
            "structure_tower.",
            "identity_tower.",
            "action_fusion_norm.",
            "response_pair_norm.",
            "line_",
            "self_item_rule_proj.",
            "self_ability_rule_proj.",
            "opp_item_rule_proj.",
            "opp_ability_rule_proj.",
            "particle_",
            "reveal_likelihood_head.",
            "typed_consequence_head.",
            "prior_gate_head.",
            "preview_prior_mlp.",
            "switch_entry_head.",
            "switch_follow_mean_head.",
            "switch_follow_tail_head.",
            "win_head.",
            "q_value_head.",
            "q_temperature",
            "consequence_value_head.",
            "consequence_value_latent_head.",
        ),
    )
    return model
