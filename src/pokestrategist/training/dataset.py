"""Tensor dataset for the decision-assist v1 model."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from pokestrategist.data.hidden_move_prior import HiddenMovePrior
from pokestrategist.data.schema import ActionHead
from pokestrategist.data.schema import DecisionSample, PhaseLabel, PlanLabel, ResponseCluster
from pokestrategist.data.schema import SemanticIntent
from pokestrategist.data.static_rules import (
    ABILITY_RULE_FEATURE_DIM,
    ITEM_RULE_FEATURE_DIM,
    MATCHUP_RULE_FEATURE_DIM,
    MOVE_RULE_FEATURE_DIM,
    SPECIES_RULE_FEATURE_DIM,
)
from pokestrategist.data.static_rules import (
    ability_rule_features,
    candidate_matchup_rule_features,
    item_rule_features,
    move_rule_features,
    species_rule_features,
)
from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM, TeamPreviewPriorCatalog, team_preview_feature_vector
from pokestrategist.data.usage_priors import USAGE_PRIOR_FEATURE_DIM, UsagePriorCatalog, load_usage_prior_catalog

MAX_LEGAL_ACTIONS = 32
PLAN_TO_ID = {label: index for index, label in enumerate(PlanLabel)}
PHASE_TO_ID = {label: index for index, label in enumerate(PhaseLabel)}
RESPONSE_TO_ID = {label: index for index, label in enumerate(ResponseCluster)}
LINE_TO_ID = {label: index for index, label in enumerate(SemanticIntent)}
FAMILY_TO_ID = {
    "attack": 0,
    "hazard": 1,
    "recovery": 2,
    "pivot": 3,
    "setup": 4,
    "status": 5,
    "utility": 6,
    "protect": 7,
    "switch": 8,
    "unknown": 9,
}
ACTION_HEAD_TO_ID = {"move": 0, "switch": 1, "tera-move": 2, "none": 3}
SOURCE_TO_ID = {"revealed": 0, "switch": 1, "prior_hidden": 2, "abstract_hidden": 3, "pad": 4}
DEFAULT_ABSTRACT_FAMILIES = ("attack", "pivot", "recovery", "setup", "hazard", "status", "utility", "protect")


def stable_hash_id(text: str | None, modulo: int) -> int:
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (modulo - 1) + 1


def _numeric_state(sample: DecisionSample) -> list[float]:
    obs = sample.observation
    own = obs.self_side
    opp = obs.opp_side
    return [
        min(obs.turn / 20.0, 5.0),
        own.active_hp,
        opp.active_hp,
        own.fainted_count / 6.0,
        opp.fainted_count / 6.0,
        float(own.tera_used),
        float(opp.tera_used),
        float(obs.field.trick_room),
        own.hazards["stealthrock"],
        own.hazards["spikes"] / 3.0,
        own.hazards["toxicspikes"] / 2.0,
        own.hazards["stickyweb"],
        opp.hazards["stealthrock"],
        opp.hazards["spikes"] / 3.0,
        opp.hazards["toxicspikes"] / 2.0,
        opp.hazards["stickyweb"],
        len(obs.history_actions) / 16.0,
        len(own.revealed_moves.get(own.active_species or "", [])) / 4.0,
        len(opp.revealed_moves.get(opp.active_species or "", [])) / 4.0,
        sample.unlock_target,
    ]


def _family_id(name: str | None) -> int:
    if not name:
        return FAMILY_TO_ID["unknown"]
    return FAMILY_TO_ID.get(name.lower(), FAMILY_TO_ID["unknown"])


def _dense_plan_posterior(sample: DecisionSample) -> list[float]:
    if sample.plan_posterior:
        return list(sample.plan_posterior)
    return [1.0 if label is sample.plan_label else 0.0 for label in PlanLabel]


def _dense_phase_posterior(sample: DecisionSample) -> list[float]:
    if sample.phase_posterior:
        return list(sample.phase_posterior)
    return [1.0 if label is sample.phase_label else 0.0 for label in PhaseLabel]


def _candidate_species(action, sample: DecisionSample) -> str | None:
    if action.switch_species:
        return action.switch_species
    return sample.observation.self_side.active_species


def _candidate_meta(action, sample: DecisionSample) -> tuple[int, float, float, float, float, float, float]:
    own = sample.observation.self_side
    species = _candidate_species(action, sample)
    revealed_moves = own.revealed_moves.get(species or "", [])
    return (
        stable_hash_id(species, 1024),
        own.species_hp.get(species or "", 1.0 if species else 0.0),
        float((own.species_status.get(species or "") or "") not in {"", "fnt"}),
        min(len(revealed_moves), 4) / 4.0,
        float((species or "") in own.revealed_items),
        float((species or "") in own.revealed_abilities),
        float((species or "") in own.revealed_tera_types),
    )


def _candidate_switch_hazard_cost(action, sample: DecisionSample) -> float:
    if action.head is not ActionHead.SWITCH:
        return 0.0
    hazards = sample.observation.self_side.hazards
    rocks = 0.125 if hazards.get("stealthrock", 0) else 0.0
    spikes = 0.125 * hazards.get("spikes", 0)
    toxic_spikes = 0.05 * hazards.get("toxicspikes", 0)
    webs = 0.05 if hazards.get("stickyweb", 0) else 0.0
    return min(1.0, rocks + spikes + toxic_spikes + webs)


def _normalize_support_count(support_count: int) -> float:
    if support_count <= 0:
        return 0.0
    return min(math.log1p(support_count) / 5.0, 1.0)


def _normalized_candidate_token(move_token: str | None) -> str:
    if not move_token:
        return ""
    if move_token.startswith("["):
        return move_token
    return move_token.strip().lower()


def _candidate_context_species(action, sample: DecisionSample) -> str | None:
    return action.switch_species or sample.observation.self_side.active_species


def _candidate_usage_prior_features(
    action,
    sample: DecisionSample,
    usage_prior_catalog: UsagePriorCatalog | None,
) -> list[float]:
    if usage_prior_catalog is None:
        return [0.0] * USAGE_PRIOR_FEATURE_DIM
    own = sample.observation.self_side
    species = _candidate_context_species(action, sample)
    item = own.revealed_items.get(species or "")
    ability = own.revealed_abilities.get(species or "")
    tera_type = own.revealed_tera_types.get(species or "")
    move_token = action.move_token if action.move_token and not str(action.move_token).startswith("[") else None
    return usage_prior_catalog.usage_prior_features(
        species,
        move_name=move_token,
        move_family=action.move_family,
        item_name=item,
        ability_name=ability,
        tera_type=tera_type,
    )


def _opponent_preview_prior_features(
    sample: DecisionSample,
    team_preview_prior: TeamPreviewPriorCatalog | None,
) -> list[float]:
    opp = sample.observation.opp_side
    active = opp.active_species or ""
    return team_preview_feature_vector(
        team_preview_prior,
        opp.team_order,
        active_species=active,
        revealed_moves=opp.revealed_moves.get(active, []),
        item=opp.revealed_items.get(active),
        ability=opp.revealed_abilities.get(active),
        tera_type=opp.revealed_tera_types.get(active),
    )


def _merged_hidden_move_candidates(
    species: str,
    *,
    revealed_moves: Sequence[str],
    hidden_move_prior: HiddenMovePrior | None,
    usage_prior_catalog: UsagePriorCatalog | None,
    item: str | None,
    limit: int,
) -> list[tuple[str, str, float, float, float]]:
    if limit <= 0:
        return []

    train_candidates = hidden_move_prior.suggest(species, revealed_moves, item=item, topk=max(limit * 2, limit)) if hidden_move_prior else []
    usage_candidates = usage_prior_catalog.top_moves(species, limit=max(limit * 2, limit), exclude_moves=revealed_moves) if usage_prior_catalog else []

    merged: dict[str, dict[str, object]] = {}
    for candidate in train_candidates:
        key = _normalized_candidate_token(candidate.move_token)
        merged[key] = {
            "move_token": candidate.move_token,
            "move_family": candidate.move_family,
            "train_prior": candidate.prior_prob,
            "support_score": _normalize_support_count(candidate.support_count),
            "usage_score": 0.0,
        }
    for candidate in usage_candidates:
        key = _normalized_candidate_token(candidate.move_token)
        record = merged.setdefault(
            key,
            {
                "move_token": candidate.move_token,
                "move_family": candidate.move_family,
                "train_prior": 0.0,
                "support_score": 0.0,
                "usage_score": 0.0,
            },
        )
        record["move_token"] = record.get("move_token") or candidate.move_token
        if not record.get("move_family"):
            record["move_family"] = candidate.move_family
        record["usage_score"] = max(float(record["usage_score"]), candidate.usage_score)

    ordered = sorted(
        merged.values(),
        key=lambda record: (
            -max(float(record["train_prior"]), float(record["usage_score"])),
            -float(record["support_score"]),
            -float(record["usage_score"]),
            -float(record["train_prior"]),
            str(record["move_token"]),
        ),
    )[:limit]
    return [
        (
            str(record["move_token"]),
            str(record["move_family"] or "utility"),
            float(record["train_prior"]),
            float(record["usage_score"]),
            float(record["support_score"]),
        )
        for record in ordered
    ]


def _merged_hidden_family_candidates(
    species: str,
    *,
    hidden_move_prior: HiddenMovePrior | None,
    usage_prior_catalog: UsagePriorCatalog | None,
    limit: int,
) -> list[tuple[str, float, float, float]]:
    if limit <= 0:
        return []

    train_families = hidden_move_prior.abstract_families(species, limit=max(limit * 2, limit)) if hidden_move_prior else []
    usage_families = usage_prior_catalog.top_families(species, limit=max(limit * 2, limit)) if usage_prior_catalog else []
    merged: dict[str, dict[str, float]] = {}
    for family, prior_prob, support_count in train_families:
        merged[family] = {
            "train_prior": prior_prob,
            "support_score": _normalize_support_count(support_count),
            "usage_score": 0.0,
        }
    for candidate in usage_families:
        record = merged.setdefault(candidate.move_family, {"train_prior": 0.0, "support_score": 0.0, "usage_score": 0.0})
        record["usage_score"] = max(float(record["usage_score"]), candidate.usage_score)

    ordered = sorted(
        merged.items(),
        key=lambda item: (
            -max(float(item[1]["train_prior"]), float(item[1]["usage_score"])),
            -item[1]["support_score"],
            -float(item[1]["usage_score"]),
            DEFAULT_ABSTRACT_FAMILIES.index(item[0]) if item[0] in DEFAULT_ABSTRACT_FAMILIES else 99,
        ),
    )[:limit]
    return [
        (
            family,
            float(values["train_prior"]),
            float(values["usage_score"]),
            values["support_score"],
        )
        for family, values in ordered
    ]


def _base_candidate_records(sample: DecisionSample) -> list[tuple[object, str, float, float, float]]:
    records: list[tuple[object, str, float, float, float]] = []
    for action in sample.legal_actions:
        source = "switch" if action.head is ActionHead.SWITCH else "revealed"
        action = action.model_copy(update={
            "candidate_source": source,
            "candidate_prior_prob": 1.0 if source == "revealed" else 0.0,
            "candidate_support_score": 1.0 if source == "revealed" else 0.0,
            "candidate_is_gold_injected": False,
        })
        proposal_score = 1.0 if source == "revealed" else 0.0
        records.append(
            (
                action,
                source,
                1.0 if source == "revealed" else 0.0,
                1.0 if source == "revealed" else 0.0,
                proposal_score,
            )
        )
    return records


def _hidden_candidate_records(
    sample: DecisionSample,
    hidden_move_prior: HiddenMovePrior | None,
    usage_prior_catalog: UsagePriorCatalog | None,
    *,
    hidden_candidate_topk: int,
) -> list[tuple[object, str, float, float, float]]:
    if sample.hidden_move_slots <= 0:
        return []
    own = sample.observation.self_side
    active = own.active_species
    if not active:
        return []
    revealed_moves = own.revealed_moves.get(active, [])
    active_item = own.revealed_items.get(active)
    move_budget = max(1, min(sample.hidden_move_slots, hidden_candidate_topk))
    records: list[tuple[object, str, float, float, float]] = []
    concrete_candidates = _merged_hidden_move_candidates(
        active,
        revealed_moves=revealed_moves,
        hidden_move_prior=hidden_move_prior,
        usage_prior_catalog=usage_prior_catalog,
        item=active_item,
        limit=move_budget,
    )
    for move_token, move_family, train_prior, usage_score, support_score in concrete_candidates:
        proposal_score = max(train_prior, usage_score)
        records.append(
            (
                sample.our_action.__class__(
                    actor=sample.perspective,
                    head=ActionHead.MOVE,
                    move_token=move_token,
                    move_family=move_family,
                    candidate_source="prior_hidden",
                    candidate_prior_prob=train_prior,
                    candidate_support_score=support_score,
                ),
                "prior_hidden",
                train_prior,
                support_score,
                proposal_score,
            )
        )
        if not own.tera_used:
            records.append(
                (
                    sample.our_action.__class__(
                        actor=sample.perspective,
                        head=ActionHead.TERA_MOVE,
                        move_token=move_token,
                        move_family=move_family,
                        tera=True,
                        candidate_source="prior_hidden",
                        candidate_prior_prob=train_prior,
                        candidate_support_score=support_score,
                    ),
                    "prior_hidden",
                    train_prior,
                    support_score,
                    proposal_score,
                )
            )

    abstract_limit = move_budget - len(concrete_candidates)
    if abstract_limit > 0:
        abstract_families = _merged_hidden_family_candidates(
            active,
            hidden_move_prior=hidden_move_prior,
            usage_prior_catalog=usage_prior_catalog,
            limit=abstract_limit,
        )
        if not abstract_families:
            abstract_families = [(family, 0.0, 0.0, 0.0) for family in DEFAULT_ABSTRACT_FAMILIES[:abstract_limit]]
        for hidden_index, (family, train_prior, usage_score, support_score) in enumerate(abstract_families, start=1):
            placeholder = f"[abstract-hidden-{family}-{hidden_index}]"
            proposal_score = max(train_prior, usage_score)
            records.append(
                (
                    sample.our_action.__class__(
                        actor=sample.perspective,
                        head=ActionHead.MOVE,
                        move_token=placeholder,
                        move_family=family,
                        candidate_source="abstract_hidden",
                        candidate_prior_prob=train_prior,
                        candidate_support_score=support_score,
                    ),
                    "abstract_hidden",
                    train_prior,
                    support_score,
                    proposal_score,
                )
            )
            if not own.tera_used:
                records.append(
                    (
                        sample.our_action.__class__(
                            actor=sample.perspective,
                            head=ActionHead.TERA_MOVE,
                            move_token=placeholder,
                            move_family=family,
                            tera=True,
                            candidate_source="abstract_hidden",
                            candidate_prior_prob=train_prior,
                            candidate_support_score=support_score,
                        ),
                        "abstract_hidden",
                        train_prior,
                        support_score,
                        proposal_score,
                    )
                )
    return records


def _candidate_records(
    sample: DecisionSample,
    hidden_move_prior: HiddenMovePrior | None,
    usage_prior_catalog: UsagePriorCatalog | None,
    *,
    hidden_candidate_topk: int,
) -> tuple[list[tuple[object, str, float, float, float]], bool]:
    deduped: dict[str, tuple[object, str, float, float, float]] = {}
    for record in _base_candidate_records(sample):
        deduped.setdefault(record[0].key(), record)
    for record in _hidden_candidate_records(
        sample,
        hidden_move_prior,
        usage_prior_catalog,
        hidden_candidate_topk=hidden_candidate_topk,
    ):
        deduped.setdefault(record[0].key(), record)

    gold_key = sample.our_action.key()
    gold_candidate_covered = gold_key in deduped

    source_rank = {"revealed": 0, "prior_hidden": 1, "abstract_hidden": 2, "switch": 3}
    ordered = sorted(
        deduped.values(),
        key=lambda record: (
            0 if record[0].head.value == "move" else 1 if record[0].head.value == "tera-move" else 2,
            source_rank.get(record[1], 99),
            -(record[4]),
            record[0].move_token or "",
            record[0].switch_slot if record[0].switch_slot is not None else 99,
            record[0].switch_species or "",
        ),
    )
    return ordered, gold_candidate_covered


def tensorize_decision_sample(
    sample: DecisionSample,
    *,
    hidden_move_prior: HiddenMovePrior | None = None,
    hidden_candidate_topk: int = 4,
    usage_prior_catalog: UsagePriorCatalog | None = None,
    team_preview_prior: TeamPreviewPriorCatalog | None = None,
    include_action_metadata: bool = False,
) -> dict[str, object]:
    obs = sample.observation
    history_ids = [stable_hash_id(token, 4096) for token in obs.history_actions[-16:]]
    history_ids = ([0] * (16 - len(history_ids))) + history_ids[-16:]

    gold_key = sample.our_action.key()
    candidate_records, gold_candidate_covered = _candidate_records(
        sample,
        hidden_move_prior,
        usage_prior_catalog,
        hidden_candidate_topk=hidden_candidate_topk,
    )
    legal_actions = [record[0] for record in candidate_records]
    if len(legal_actions) > MAX_LEGAL_ACTIONS:
        gold_in_legal = next((idx for idx, candidate in enumerate(legal_actions) if candidate.key() == gold_key), None)
        if gold_candidate_covered and gold_in_legal is not None and gold_in_legal >= MAX_LEGAL_ACTIONS:
            candidate_records = candidate_records[: MAX_LEGAL_ACTIONS - 1] + [candidate_records[gold_in_legal]]
        else:
            candidate_records = candidate_records[:MAX_LEGAL_ACTIONS]
        legal_actions = [record[0] for record in candidate_records]
        gold_candidate_covered = any(candidate.key() == gold_key for candidate in legal_actions)
    recoverable_keys = {candidate.key() for candidate in sample.recoverable_actions}

    legal_head_ids = [ACTION_HEAD_TO_ID[action.head.value] for action in legal_actions]
    legal_move_ids = [stable_hash_id(action.move_token, 4096) for action in legal_actions]
    legal_family_ids = [_family_id(action.move_family) for action in legal_actions]
    legal_switch_slots = [action.switch_slot if action.switch_slot is not None else 0 for action in legal_actions]
    legal_tera_flags = [1 if action.tera else 0 for action in legal_actions]
    legal_recoverable_flags = [candidate.key() in recoverable_keys for candidate in legal_actions]
    legal_candidate_source_ids = [SOURCE_TO_ID[record[1]] for record in candidate_records]
    legal_candidate_prior_probs = [record[2] for record in candidate_records]
    legal_candidate_support_scores = [record[3] for record in candidate_records]
    legal_external_prior_features = [
        _candidate_usage_prior_features(action, sample, usage_prior_catalog) for action in legal_actions
    ]
    legal_usage_scores = [max(features) if features else 0.0 for features in legal_external_prior_features]
    candidate_meta = [_candidate_meta(action, sample) for action in legal_actions]
    legal_species_ids = [meta[0] for meta in candidate_meta]
    legal_species_hps = [meta[1] for meta in candidate_meta]
    legal_species_status_flags = [meta[2] for meta in candidate_meta]
    legal_known_move_counts = [meta[3] for meta in candidate_meta]
    legal_item_known_flags = [meta[4] for meta in candidate_meta]
    legal_ability_known_flags = [meta[5] for meta in candidate_meta]
    legal_tera_known_flags = [meta[6] for meta in candidate_meta]
    legal_switch_hazard_costs = [_candidate_switch_hazard_cost(action, sample) for action in legal_actions]
    legal_species_rule_features = [species_rule_features(_candidate_species(action, sample)) for action in legal_actions]
    legal_move_rule_features = [move_rule_features(action.move_token, action.move_family) for action in legal_actions]
    legal_matchup_rule_features = [candidate_matchup_rule_features(action, sample) for action in legal_actions]
    switch_candidate_mask = [action.head.value == "switch" for action in legal_actions]
    legal_mask = [1] * len(legal_actions)

    padding = MAX_LEGAL_ACTIONS - len(legal_actions)
    legal_head_ids.extend([ACTION_HEAD_TO_ID["none"]] * padding)
    legal_move_ids.extend([0] * padding)
    legal_family_ids.extend([FAMILY_TO_ID["unknown"]] * padding)
    legal_switch_slots.extend([0] * padding)
    legal_tera_flags.extend([0] * padding)
    legal_recoverable_flags.extend([False] * padding)
    legal_candidate_source_ids.extend([SOURCE_TO_ID["pad"]] * padding)
    legal_candidate_prior_probs.extend([0.0] * padding)
    legal_candidate_support_scores.extend([0.0] * padding)
    legal_external_prior_features.extend([[0.0] * USAGE_PRIOR_FEATURE_DIM for _ in range(padding)])
    legal_usage_scores.extend([0.0] * padding)
    legal_species_ids.extend([0] * padding)
    legal_species_hps.extend([0.0] * padding)
    legal_species_status_flags.extend([0.0] * padding)
    legal_known_move_counts.extend([0.0] * padding)
    legal_item_known_flags.extend([0.0] * padding)
    legal_ability_known_flags.extend([0.0] * padding)
    legal_tera_known_flags.extend([0.0] * padding)
    legal_switch_hazard_costs.extend([0.0] * padding)
    legal_species_rule_features.extend([[0.0] * SPECIES_RULE_FEATURE_DIM for _ in range(padding)])
    legal_move_rule_features.extend([[0.0] * MOVE_RULE_FEATURE_DIM for _ in range(padding)])
    legal_matchup_rule_features.extend([[0.0] * MATCHUP_RULE_FEATURE_DIM for _ in range(padding)])
    switch_candidate_mask.extend([False] * padding)
    legal_mask.extend([0] * padding)

    legal_target = next((candidate_index for candidate_index, candidate in enumerate(legal_actions) if candidate.key() == gold_key), -1)

    family_target = _family_id(sample.family_target)
    supervised_target = legal_target >= 0
    self_active = obs.self_side.active_species or ""
    opp_active = obs.opp_side.active_species or ""
    payload: dict[str, object] = {
        "replay_id": sample.replay_id,
        "state_numeric": _numeric_state(sample),
        "self_species_id": stable_hash_id(obs.self_side.active_species, 1024),
        "opp_species_id": stable_hash_id(obs.opp_side.active_species, 1024),
        "self_species_rule_features": species_rule_features(obs.self_side.active_species),
        "opp_species_rule_features": species_rule_features(obs.opp_side.active_species),
        "self_item_rule_features": item_rule_features(obs.self_side.revealed_items.get(self_active)),
        "self_ability_rule_features": ability_rule_features(obs.self_side.revealed_abilities.get(self_active)),
        "opp_item_rule_features": item_rule_features(obs.opp_side.revealed_items.get(opp_active)),
        "opp_ability_rule_features": ability_rule_features(obs.opp_side.revealed_abilities.get(opp_active)),
        "opp_preview_prior_features": _opponent_preview_prior_features(sample, team_preview_prior),
        "history_token_ids": history_ids,
        "plan_target": PLAN_TO_ID[sample.plan_label],
        "plan_posterior_target": _dense_plan_posterior(sample),
        "phase_target": PHASE_TO_ID[sample.phase_label],
        "phase_posterior_target": _dense_phase_posterior(sample),
        "line_target": LINE_TO_ID[sample.semantic_intent],
        "unlock_target": sample.unlock_target,
        "response_target": RESPONSE_TO_ID[sample.response_cluster],
        "particle_posterior_target": sample.particle_posterior.as_vector(),
        "particle_uncertainty_target": sample.particle_posterior.uncertainty(),
        "reveal_likelihood_target": sample.reveal_likelihood.as_vector(),
        "belief_summary_target": sample.belief_summary.as_vector(),
        "belief_uncertainty_target": sample.belief_summary.uncertainty(),
        "resource_ledger_target": sample.resource_ledger.as_vector(),
        "resource_value_target": sample.resource_ledger.scalar_value(),
        "family_target": family_target,
        "legal_action_head_ids": legal_head_ids,
        "legal_action_move_ids": legal_move_ids,
        "legal_action_family_ids": legal_family_ids,
        "legal_action_switch_slots": legal_switch_slots,
        "legal_action_tera_flags": legal_tera_flags,
        "legal_action_recoverable_flags": legal_recoverable_flags,
        "legal_action_candidate_source_ids": legal_candidate_source_ids,
        "legal_action_prior_probs": legal_candidate_prior_probs,
        "legal_action_support_scores": legal_candidate_support_scores,
        "legal_action_external_prior_features": legal_external_prior_features,
        "legal_action_usage_scores": legal_usage_scores,
        "legal_action_species_ids": legal_species_ids,
        "legal_action_species_hps": legal_species_hps,
        "legal_action_species_status_flags": legal_species_status_flags,
        "legal_action_known_move_counts": legal_known_move_counts,
        "legal_action_item_known_flags": legal_item_known_flags,
        "legal_action_ability_known_flags": legal_ability_known_flags,
        "legal_action_tera_known_flags": legal_tera_known_flags,
        "legal_action_switch_hazard_costs": legal_switch_hazard_costs,
        "legal_action_species_rule_features": legal_species_rule_features,
        "legal_action_move_rule_features": legal_move_rule_features,
        "legal_action_matchup_rule_features": legal_matchup_rule_features,
        "legal_action_mask": legal_mask,
        "switch_candidate_mask": switch_candidate_mask,
        "legal_target": legal_target,
        "future_mean_target": sample.future_target.as_vector(),
        "future_tail_target": sample.future_target.tail_vector(),
        "typed_consequence_target": sample.typed_consequence.as_vector(),
        "typed_consequence_bin_target": sample.typed_consequence.bin_indices(),
        "typed_consequence_interaction_target": sample.typed_consequence.interaction_vector(),
        "gold_is_switch": sample.our_action.head.value == "switch",
        "gold_switch_slot": sample.our_action.switch_slot if sample.our_action.switch_slot is not None else -1,
        "gold_switch_species_id": stable_hash_id(sample.our_action.switch_species, 1024) if sample.our_action.switch_species else 0,
        "head_target": 1 if sample.our_action.head.value == "switch" else 0,
        "gold_candidate_covered": gold_candidate_covered,
        "supervised_target": supervised_target,
    }
    if include_action_metadata:
        payload["legal_action_entries"] = [
            {
                "key": action.key(),
                "head": action.head.value,
                "move_token": action.move_token,
                "move_family": action.move_family,
                "tera": action.tera,
                "switch_slot": action.switch_slot,
                "switch_species": action.switch_species,
                "candidate_source": action.candidate_source,
                "candidate_prior_prob": action.candidate_prior_prob,
                "candidate_support_score": action.candidate_support_score,
            }
            for action in legal_actions
        ]
    return payload


class DecisionTensorDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        hidden_move_prior: HiddenMovePrior | None = None,
        team_preview_prior: TeamPreviewPriorCatalog | None = None,
        hidden_candidate_topk: int = 4,
        use_usage_priors: bool = True,
        usage_data_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.hidden_move_prior = hidden_move_prior
        self.team_preview_prior = team_preview_prior
        self.hidden_candidate_topk = hidden_candidate_topk
        self.usage_prior_catalog = load_usage_prior_catalog(usage_data_dir) if use_usage_priors else None
        offsets: list[int] = []
        replay_ids: list[str] = []
        content_digest = hashlib.blake2b(digest_size=16)
        with self.path.open("rb") as fh:
            while True:
                if max_samples is not None and len(offsets) >= max_samples:
                    break
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                content_digest.update(line)
                payload = json.loads(line)
                offsets.append(offset)
                replay_ids.append(str(payload.get("replay_id") or payload.get("replay_id") or "unknown-replay"))
        self.offsets = offsets
        self.replay_ids = replay_ids
        self.content_fingerprint = content_digest.hexdigest()
        self._file_handle: BinaryIO | None = None
        self._file_handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_file_handle"] = None
        state["_file_handle_pid"] = None
        return state

    def _read_sample(self, index: int) -> DecisionSample:
        current_pid = os.getpid()
        if (
            self._file_handle is None
            or self._file_handle.closed
            or self._file_handle_pid != current_pid
        ):
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.close()
            self._file_handle = self.path.open("rb")
            self._file_handle_pid = current_pid
        self._file_handle.seek(self.offsets[index])
        line = self._file_handle.readline()
        return DecisionSample.model_validate(json.loads(line))

    def raw_sample(self, index: int) -> DecisionSample:
        return self._read_sample(index)

    def iter_raw_samples(self, indices: Iterable[int]) -> Iterable[DecisionSample]:
        for index in indices:
            yield self._read_sample(index)

    def set_hidden_move_prior(self, hidden_move_prior: HiddenMovePrior | None) -> None:
        self.hidden_move_prior = hidden_move_prior

    def set_team_preview_prior(self, team_preview_prior: TeamPreviewPriorCatalog | None) -> None:
        self.team_preview_prior = team_preview_prior

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._read_sample(index)
        return tensorize_decision_sample(
            sample,
            hidden_move_prior=self.hidden_move_prior,
            hidden_candidate_topk=self.hidden_candidate_topk,
            usage_prior_catalog=self.usage_prior_catalog,
            team_preview_prior=self.team_preview_prior,
        )


def collate_decision_batch(items: Sequence[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    tensor_keys = {
        "state_numeric": torch.float32,
        "self_species_id": torch.long,
        "opp_species_id": torch.long,
        "self_species_rule_features": torch.float32,
        "opp_species_rule_features": torch.float32,
        "self_item_rule_features": torch.float32,
        "self_ability_rule_features": torch.float32,
        "opp_item_rule_features": torch.float32,
        "opp_ability_rule_features": torch.float32,
        "opp_preview_prior_features": torch.float32,
        "history_token_ids": torch.long,
        "plan_target": torch.long,
        "plan_posterior_target": torch.float32,
        "phase_target": torch.long,
        "phase_posterior_target": torch.float32,
        "line_target": torch.long,
        "unlock_target": torch.float32,
        "response_target": torch.long,
        "particle_posterior_target": torch.float32,
        "particle_uncertainty_target": torch.float32,
        "reveal_likelihood_target": torch.float32,
        "belief_summary_target": torch.float32,
        "belief_uncertainty_target": torch.float32,
        "resource_ledger_target": torch.float32,
        "resource_value_target": torch.float32,
        "family_target": torch.long,
        "legal_action_head_ids": torch.long,
        "legal_action_move_ids": torch.long,
        "legal_action_family_ids": torch.long,
        "legal_action_switch_slots": torch.long,
        "legal_action_tera_flags": torch.long,
        "legal_action_recoverable_flags": torch.bool,
        "legal_action_candidate_source_ids": torch.long,
        "legal_action_prior_probs": torch.float32,
        "legal_action_support_scores": torch.float32,
        "legal_action_external_prior_features": torch.float32,
        "legal_action_usage_scores": torch.float32,
        "legal_action_species_ids": torch.long,
        "legal_action_species_hps": torch.float32,
        "legal_action_species_status_flags": torch.float32,
        "legal_action_known_move_counts": torch.float32,
        "legal_action_item_known_flags": torch.float32,
        "legal_action_ability_known_flags": torch.float32,
        "legal_action_tera_known_flags": torch.float32,
        "legal_action_switch_hazard_costs": torch.float32,
        "legal_action_species_rule_features": torch.float32,
        "legal_action_move_rule_features": torch.float32,
        "legal_action_matchup_rule_features": torch.float32,
        "legal_action_mask": torch.bool,
        "switch_candidate_mask": torch.bool,
        "legal_target": torch.long,
        "future_mean_target": torch.float32,
        "future_tail_target": torch.float32,
        "typed_consequence_target": torch.float32,
        "typed_consequence_bin_target": torch.long,
        "typed_consequence_interaction_target": torch.float32,
        "gold_is_switch": torch.bool,
        "gold_switch_slot": torch.long,
        "gold_switch_species_id": torch.long,
        "head_target": torch.long,
        "gold_candidate_covered": torch.bool,
        "supervised_target": torch.bool,
    }
    batch: dict[str, torch.Tensor | list[str]] = {"replay_id": [str(item["replay_id"]) for item in items]}
    for key, dtype in tensor_keys.items():
        batch[key] = torch.tensor([item[key] for item in items], dtype=dtype)
    return batch


def _stable_split_score(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def grouped_replay_split(replay_ids_by_sample: Sequence[str], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    replay_ids = sorted(set(replay_ids_by_sample))
    scored = sorted((_stable_split_score(replay_id, seed), replay_id) for replay_id in replay_ids)
    val_replays = {replay_id for score, replay_id in scored if score < val_fraction}
    if not val_replays and scored:
        val_replay_count = max(1, int(len(scored) * val_fraction))
        val_replays = {replay_id for _, replay_id in scored[:val_replay_count]}
    if len(val_replays) == len(replay_ids) and scored:
        val_replay_count = max(1, int(len(scored) * val_fraction))
        val_replays = {replay_id for _, replay_id in scored[:val_replay_count]}
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, replay_id in enumerate(replay_ids_by_sample):
        if replay_id in val_replays:
            val_indices.append(index)
        else:
            train_indices.append(index)
    return train_indices, val_indices