"""Team-preview hidden-set prior builder and scorer.

This module extracts per-species hindsight template evidence from processed
decision JSONL, builds a species-level template bank, and adds a first
team-conditioned reranking layer based on teammate compatibility statistics.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pokepilot.data.schema import ActionHead, DecisionSample
from pokepilot.data.static_rules import load_static_rule_catalog, normalize_name

DEFAULT_TEAM_WEIGHT = 0.75
DEFAULT_TEAMMATE_ALPHA = 0.25
ROLE_TAGS = (
    "attack",
    "pivot",
    "hazard",
    "recovery",
    "setup",
    "protect",
    "status",
    "removal",
    "speed_control",
    "utility",
)
TEAM_PREVIEW_ROLE_ORDER = ("attack", "pivot", "hazard", "recovery", "setup", "utility")
TEAM_PREVIEW_FEATURE_DIM = 7 + len(TEAM_PREVIEW_ROLE_ORDER) + 7  # +7 for team-level features


def _canonical_name(text: str | None) -> str:
    return (text or "").strip()


def _unique_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        display = _canonical_name(value)
        key = normalize_name(display)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(display)
    return tuple(ordered)


def _role_tags_for_moves(moves: Sequence[str]) -> tuple[str, ...]:
    catalog = load_static_rule_catalog()
    tags: set[str] = set()
    for move in moves:
        entry = catalog.move_entry(move)
        if not entry:
            continue
        raw_tags = {str(tag).lower() for tag in entry.get("tags", [])}
        tags.update(tag for tag in raw_tags if tag in ROLE_TAGS)
        if "boost" in raw_tags:
            tags.add("setup")
        if float(entry.get("power") or 0.0) > 0.0 and str(entry.get("category", "")).lower() != "status":
            tags.add("attack")
    return tuple(sorted(tags))


def _normalized_support_count(support_count: int) -> float:
    if support_count <= 0:
        return 0.0
    return min(math.log1p(support_count) / 5.0, 1.0)


def _role_probability(predictions: Sequence["TeamPreviewTemplatePrediction"], role_name: str) -> float:
    if role_name == "utility":
        utility_tags = {"utility", "status", "protect"}
        return sum(
            prediction.probability
            for prediction in predictions
            if utility_tags.intersection(prediction.template.role_tags)
        )
    return sum(
        prediction.probability
        for prediction in predictions
        if role_name in prediction.template.role_tags
    )


@dataclass(frozen=True, slots=True)
class PreviewSetObservation:
    species: str
    moves: tuple[str, ...] = ()
    item: str | None = None
    ability: str | None = None
    tera_type: str | None = None
    spread_bucket: str = "unknown"

    def evidence_count(self) -> int:
        return (
            len(self.moves)
            + int(bool(self.item))
            + int(bool(self.ability))
            + int(bool(self.tera_type))
            + int(self.spread_bucket != "unknown")
        )


@dataclass(frozen=True, slots=True)
class TeamPreviewExample:
    team_species: tuple[str, ...]
    set_observations: tuple[PreviewSetObservation, ...]

    def observation_for_species(self, species_name: str) -> PreviewSetObservation | None:
        target = normalize_name(species_name)
        for observation in self.set_observations:
            if normalize_name(observation.species) == target:
                return observation
        return None


@dataclass(slots=True)
class TeamPreviewTemplate:
    template_id: str
    species: str
    moves: list[str]
    item: str | None
    ability: str | None
    tera_type: str | None
    spread_bucket: str
    role_tags: list[str]
    support_count: int
    prior_prob: float
    teammate_log_lift: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "species": self.species,
            "moves": self.moves,
            "item": self.item,
            "ability": self.ability,
            "tera_type": self.tera_type,
            "spread_bucket": self.spread_bucket,
            "role_tags": self.role_tags,
            "support_count": self.support_count,
            "prior_prob": self.prior_prob,
            "teammate_log_lift": self.teammate_log_lift,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TeamPreviewTemplate:
        return cls(
            template_id=str(payload["template_id"]),
            species=str(payload["species"]),
            moves=[str(move) for move in payload.get("moves", [])],
            item=payload.get("item"),
            ability=payload.get("ability"),
            tera_type=payload.get("tera_type"),
            spread_bucket=str(payload.get("spread_bucket") or "unknown"),
            role_tags=[str(tag) for tag in payload.get("role_tags", [])],
            support_count=int(payload.get("support_count", 0)),
            prior_prob=float(payload.get("prior_prob", 0.0)),
            teammate_log_lift={
                str(key): float(value) for key, value in dict(payload.get("teammate_log_lift", {})).items()
            },
        )


@dataclass(slots=True)
class TeamPreviewTemplatePrediction:
    template: TeamPreviewTemplate
    score: float
    probability: float
    species_prior: float
    team_boost: float
    reveal_consistency: float


def _template_reveal_consistency(
    template: TeamPreviewTemplate,
    *,
    revealed_moves: Sequence[str] = (),
    item: str | None = None,
    ability: str | None = None,
    tera_type: str | None = None,
) -> float:
    score = 0.0
    revealed_count = 0

    template_moves = {normalize_name(move) for move in template.moves}
    for move in revealed_moves:
        move_key = normalize_name(move)
        if not move_key:
            continue
        revealed_count += 1
        score += 1.0 if move_key in template_moves else -1.0

    if item:
        revealed_count += 1
        score += 1.0 if normalize_name(item) == normalize_name(template.item) else -1.0
    if ability:
        revealed_count += 1
        score += 1.0 if normalize_name(ability) == normalize_name(template.ability) else -1.0
    if tera_type:
        revealed_count += 1
        score += 1.0 if normalize_name(tera_type) == normalize_name(template.tera_type) else -1.0

    if revealed_count <= 0:
        return 0.0
    return score / float(revealed_count)


@dataclass(slots=True)
class TeamPreviewPriorCatalog:
    templates_by_species: dict[str, list[TeamPreviewTemplate]]
    global_species_presence: dict[str, float]
    example_count: int
    team_condition_weight: float = DEFAULT_TEAM_WEIGHT
    teammate_alpha: float = DEFAULT_TEAMMATE_ALPHA

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "example_count": self.example_count,
                "team_condition_weight": self.team_condition_weight,
                "teammate_alpha": self.teammate_alpha,
            },
            "global_species_presence": self.global_species_presence,
            "species_templates": {
                species: [template.to_dict() for template in templates]
                for species, templates in self.templates_by_species.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TeamPreviewPriorCatalog:
        metadata = dict(payload.get("metadata", {}))
        return cls(
            templates_by_species={
                species: [TeamPreviewTemplate.from_dict(template) for template in templates]
                for species, templates in dict(payload.get("species_templates", {})).items()
            },
            global_species_presence={
                str(key): float(value) for key, value in dict(payload.get("global_species_presence", {})).items()
            },
            example_count=int(metadata.get("example_count", 0)),
            team_condition_weight=float(metadata.get("team_condition_weight", DEFAULT_TEAM_WEIGHT)),
            teammate_alpha=float(metadata.get("teammate_alpha", DEFAULT_TEAMMATE_ALPHA)),
        )

    def top_templates(self, species_name: str, *, limit: int = 5) -> list[TeamPreviewTemplate]:
        templates = self.templates_by_species.get(normalize_name(species_name), [])
        return templates[:limit]

    def predict_species(
        self,
        species_name: str,
        team_species: Sequence[str],
        *,
        limit: int = 5,
        team_condition_weight: float | None = None,
        revealed_moves: Sequence[str] = (),
        item: str | None = None,
        ability: str | None = None,
        tera_type: str | None = None,
    ) -> list[TeamPreviewTemplatePrediction]:
        templates = list(self.top_templates(species_name, limit=max(limit, 1) * 4))
        if not templates:
            return []
        target_species = normalize_name(species_name)
        teammate_keys = [
            normalize_name(species)
            for species in team_species
            if normalize_name(species) and normalize_name(species) != target_species
        ]
        weight = self.team_condition_weight if team_condition_weight is None else float(team_condition_weight)
        scored: list[tuple[TeamPreviewTemplate, float, float]] = []
        for template in templates:
            species_prior = math.log(max(template.prior_prob, 1e-8))
            team_boost = 0.0
            if teammate_keys:
                team_boost = sum(template.teammate_log_lift.get(teammate, 0.0) for teammate in teammate_keys) / float(len(teammate_keys))
            reveal_consistency = _template_reveal_consistency(
                template,
                revealed_moves=revealed_moves,
                item=item,
                ability=ability,
                tera_type=tera_type,
            )
            scored.append((template, species_prior + weight * team_boost + reveal_consistency, team_boost, reveal_consistency))
        max_score = max(score for _, score, _, _ in scored)
        probs = [math.exp(score - max_score) for _, score, _, _ in scored]
        total = sum(probs) or 1.0
        predictions = [
            TeamPreviewTemplatePrediction(
                template=template,
                score=score,
                probability=prob / total,
                species_prior=template.prior_prob,
                team_boost=team_boost,
                reveal_consistency=reveal_consistency,
            )
            for (template, score, team_boost, reveal_consistency), prob in zip(scored, probs)
        ]
        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions[:limit]

    def team_analysis(
        self,
        team_species: Sequence[str],
        *,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Analyze an opponent's full 6-Pokemon team for role closure, synergy, and lead candidates.

        Returns a structured dict suitable for downstream feature computation and UI rendering.
        """
        roster = _unique_preserve_order(team_species)
        if not roster:
            return {"roster": [], "predictions": {}, "synergy": [], "role_coverage": {}, "lead_candidates": []}

        # Per-species template predictions
        predictions: dict[str, list[TeamPreviewTemplatePrediction]] = {}
        for species in roster:
            preds = self.predict_species(species, roster, limit=limit, team_condition_weight=self.team_condition_weight)
            if preds:
                predictions[species] = preds

        # Pairwise synergy: average teammate_log_lift between best templates of each pair
        synergy_pairs: list[dict[str, Any]] = []
        for i, sa in enumerate(roster):
            for sb in roster[i + 1 :]:
                ta = predictions.get(sa, [None])[0] if predictions.get(sa) else None
                tb = predictions.get(sb, [None])[0] if predictions.get(sb) else None
                if ta and tb:
                    lift_ab = ta.template.teammate_log_lift.get(normalize_name(sb), 0.0)
                    lift_ba = tb.template.teammate_log_lift.get(normalize_name(sa), 0.0)
                    synergy_pairs.append({
                        "species_a": sa, "species_b": sb,
                        "synergy": round((lift_ab + lift_ba) / 2.0, 4),
                    })

        # Role coverage: scan best template for each species, count role tags
        role_counts: dict[str, int] = {role: 0 for role in ROLE_TAGS}
        for species in roster:
            preds = predictions.get(species, [])
            if preds:
                for tag in preds[0].template.role_tags:
                    if tag in role_counts:
                        role_counts[tag] += 1

        # Role closure gaps
        role_gaps = {
            "has_hazard": role_counts.get("hazard", 0) >= 1,
            "has_removal": role_counts.get("removal", 0) >= 1,
            "has_speed_control": role_counts.get("speed_control", 0) >= 1,
            "has_pivot": role_counts.get("pivot", 0) >= 1,
            "has_setup": role_counts.get("setup", 0) >= 1,
            "has_recovery": role_counts.get("recovery", 0) >= 1,
            "hazard_count": role_counts.get("hazard", 0),
            "removal_count": role_counts.get("removal", 0),
            "pivot_count": role_counts.get("pivot", 0),
            "setup_count": role_counts.get("setup", 0),
        }
        closure_score = sum([
            1.0 if role_gaps["has_hazard"] else 0.0,
            1.0 if role_gaps["has_removal"] else 0.0,
            0.8 if role_gaps["has_speed_control"] else 0.0,
            0.8 if role_gaps["has_pivot"] else 0.0,
            0.5 if role_gaps["has_setup"] else 0.0,
            0.4 if role_gaps["has_recovery"] else 0.0,
        ]) / 4.5  # normalized to [0, 1]

        # Lead candidates: score each species by prior_prob + team_boost
        lead_scores: list[dict[str, Any]] = []
        for species in roster:
            preds = predictions.get(species, [])
            if preds:
                best = preds[0]
                lead_scores.append({
                    "species": species,
                    "lead_score": round(best.species_prior + 0.3 * best.team_boost, 4),
                    "prior_prob": best.template.prior_prob,
                    "role_tags": best.template.role_tags,
                })
        lead_scores.sort(key=lambda x: x["lead_score"], reverse=True)

        return {
            "roster": list(roster),
            "predictions": {s: [p.template.to_dict() for p in preds[:2]] for s, preds in predictions.items()},
            "synergy": sorted(synergy_pairs, key=lambda x: x["synergy"], reverse=True),
            "role_coverage": {**role_gaps, "closure_score": round(closure_score, 4)},
            "lead_candidates": lead_scores[:3],
        }


@dataclass
class _AggregatedSetObservation:
    species: str
    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    tera_type: str | None = None
    spread_bucket: str = "unknown"


@dataclass
class _AggregatedPreviewState:
    team_species: list[str] = field(default_factory=list)
    observations: dict[str, _AggregatedSetObservation] = field(default_factory=dict)


def _ensure_species(state: _AggregatedPreviewState, species_name: str | None) -> _AggregatedSetObservation | None:
    display = _canonical_name(species_name)
    key = normalize_name(display)
    if not key:
        return None
    if key not in state.observations:
        state.observations[key] = _AggregatedSetObservation(species=display)
    if all(normalize_name(species) != key for species in state.team_species):
        state.team_species.append(display)
    return state.observations[key]


def _merge_observation_side(state: _AggregatedPreviewState, side_payload: dict[str, Any]) -> None:
    for species in side_payload.get("team_order", []):
        _ensure_species(state, species)
    for species, moves in dict(side_payload.get("revealed_moves", {})).items():
        record = _ensure_species(state, species)
        if record is not None:
            record.moves.update(_canonical_name(move) for move in moves if _canonical_name(move))
    for species, item_name in dict(side_payload.get("revealed_items", {})).items():
        record = _ensure_species(state, species)
        if record is not None and _canonical_name(item_name):
            record.item = _canonical_name(item_name)
    for species, ability_name in dict(side_payload.get("revealed_abilities", {})).items():
        record = _ensure_species(state, species)
        if record is not None and _canonical_name(ability_name):
            record.ability = _canonical_name(ability_name)
    for species, tera_type in dict(side_payload.get("revealed_tera_types", {})).items():
        record = _ensure_species(state, species)
        if record is not None and _canonical_name(tera_type):
            record.tera_type = _canonical_name(tera_type)


def _merge_opponent_action(state: _AggregatedPreviewState, sample: DecisionSample) -> None:
    action = sample.opponent_action
    if action is None:
        return
    if action.head in {ActionHead.MOVE, ActionHead.TERA_MOVE}:
        record = _ensure_species(state, sample.observation.opp_side.active_species)
        if record is not None and _canonical_name(action.move_token):
            record.moves.add(_canonical_name(action.move_token))
    if action.head is ActionHead.SWITCH and action.switch_species:
        _ensure_species(state, action.switch_species)


def iter_team_preview_examples(
    samples: Iterable[DecisionSample],
    *,
    min_team_size: int = 2,
    min_evidence_fields: int = 1,
) -> Iterator[TeamPreviewExample]:
    grouped: dict[tuple[str, str], _AggregatedPreviewState] = {}
    for sample in samples:
        key = (sample.replay_id, sample.perspective.value)
        state = grouped.setdefault(key, _AggregatedPreviewState())
        _merge_observation_side(state, sample.observation.opp_side.model_dump())
        _merge_opponent_action(state, sample)

    for state in grouped.values():
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


def iter_team_preview_examples_from_processed(
    path: str | Path,
    *,
    min_team_size: int = 2,
    min_evidence_fields: int = 1,
) -> Iterator[TeamPreviewExample]:
    source = Path(path)
    def _iter_samples() -> Iterator[DecisionSample]:
        with source.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield DecisionSample.model_validate(json.loads(line))

    yield from iter_team_preview_examples(
        _iter_samples(),
        min_team_size=min_team_size,
        min_evidence_fields=min_evidence_fields,
    )


def _template_key(observation: PreviewSetObservation) -> str:
    parts = [
        normalize_name(observation.species),
        ",".join(normalize_name(move) for move in observation.moves),
        normalize_name(observation.item),
        normalize_name(observation.ability),
        normalize_name(observation.tera_type),
        normalize_name(observation.spread_bucket) or "unknown",
    ]
    return "||".join(parts)


def build_team_preview_prior(
    examples: Iterable[TeamPreviewExample],
    *,
    min_template_count: int = 1,
    teammate_alpha: float = DEFAULT_TEAMMATE_ALPHA,
    team_condition_weight: float = DEFAULT_TEAM_WEIGHT,
) -> TeamPreviewPriorCatalog:
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    template_metadata: dict[str, PreviewSetObservation] = {}
    template_teammates: dict[str, Counter[str]] = defaultdict(Counter)
    global_species_counts: Counter[str] = Counter()
    example_count = 0

    for example in examples:
        normalized_team = [normalize_name(species) for species in example.team_species if normalize_name(species)]
        if len(normalized_team) < 2:
            continue
        example_count += 1
        global_species_counts.update(normalized_team)
        for observation in example.set_observations:
            species_key = normalize_name(observation.species)
            if not species_key:
                continue
            template_id = _template_key(observation)
            template_counts[species_key][template_id] += 1
            template_metadata.setdefault(template_id, observation)
            for teammate_key in normalized_team:
                if teammate_key != species_key:
                    template_teammates[template_id][teammate_key] += 1

    global_species_presence = {
        species_key: count / float(example_count)
        for species_key, count in global_species_counts.items()
        if example_count > 0
    }
    species_templates: dict[str, list[TeamPreviewTemplate]] = {}
    for species_key, counter in template_counts.items():
        kept = [(template_id, count) for template_id, count in counter.items() if count >= min_template_count]
        total_support = sum(count for _, count in kept)
        if total_support <= 0:
            continue
        templates: list[TeamPreviewTemplate] = []
        for template_id, count in sorted(kept, key=lambda item: (-item[1], item[0])):
            observation = template_metadata[template_id]
            teammate_log_lift: dict[str, float] = {}
            for teammate_key, teammate_count in template_teammates.get(template_id, {}).items():
                conditional_prob = (teammate_count + teammate_alpha) / float(count + 2.0 * teammate_alpha)
                baseline_prob = max(global_species_presence.get(teammate_key, 0.0), 1e-6)
                teammate_log_lift[teammate_key] = math.log(max(conditional_prob, 1e-6) / baseline_prob)
            templates.append(
                TeamPreviewTemplate(
                    template_id=template_id,
                    species=observation.species,
                    moves=list(observation.moves),
                    item=observation.item,
                    ability=observation.ability,
                    tera_type=observation.tera_type,
                    spread_bucket=observation.spread_bucket,
                    role_tags=list(_role_tags_for_moves(observation.moves)),
                    support_count=count,
                    prior_prob=count / float(total_support),
                    teammate_log_lift=teammate_log_lift,
                )
            )
        species_templates[species_key] = templates

    return TeamPreviewPriorCatalog(
        templates_by_species=species_templates,
        global_species_presence=global_species_presence,
        example_count=example_count,
        team_condition_weight=team_condition_weight,
        teammate_alpha=teammate_alpha,
    )


def build_team_preview_prior_from_processed(
    path: str | Path,
    *,
    min_team_size: int = 2,
    min_evidence_fields: int = 1,
    min_template_count: int = 1,
    teammate_alpha: float = DEFAULT_TEAMMATE_ALPHA,
    team_condition_weight: float = DEFAULT_TEAM_WEIGHT,
) -> TeamPreviewPriorCatalog:
    return build_team_preview_prior(
        iter_team_preview_examples_from_processed(
            path,
            min_team_size=min_team_size,
            min_evidence_fields=min_evidence_fields,
        ),
        min_template_count=min_template_count,
        teammate_alpha=teammate_alpha,
        team_condition_weight=team_condition_weight,
    )


def load_team_preview_prior_catalog(path: str | Path) -> TeamPreviewPriorCatalog | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    return TeamPreviewPriorCatalog.from_dict(payload)


def team_preview_feature_vector(
    catalog: TeamPreviewPriorCatalog | None,
    team_species: Sequence[str],
    *,
    active_species: str | None,
    revealed_moves: Sequence[str] = (),
    item: str | None = None,
    ability: str | None = None,
    tera_type: str | None = None,
    limit: int = 3,
) -> list[float]:
    if catalog is None:
        return [0.0] * TEAM_PREVIEW_FEATURE_DIM

    roster = _unique_preserve_order(team_species)
    if not roster:
        return [0.0] * TEAM_PREVIEW_FEATURE_DIM

    roster_size_norm = min(len(roster) / 6.0, 1.0)
    roster_match_rate = sum(
        1 for species in roster if normalize_name(species) in catalog.templates_by_species
    ) / float(len(roster))

    # Team-level analysis
    analysis = catalog.team_analysis(roster, limit=limit)
    closure = analysis.get("role_coverage", {})
    closure_score = closure.get("closure_score", 0.0)
    has_hazard = 1.0 if closure.get("has_hazard") else 0.0
    has_removal = 1.0 if closure.get("has_removal") else 0.0
    synergies = analysis.get("synergy", [])
    avg_synergy = sum(s["synergy"] for s in synergies) / max(len(synergies), 1)
    max_synergy = max((s["synergy"] for s in synergies), default=0.0)
    min_synergy = min((s["synergy"] for s in synergies), default=0.0)
    lead_candidates = analysis.get("lead_candidates", [])
    top_lead_score = lead_candidates[0]["lead_score"] if lead_candidates else 0.0

    predictions = (
        catalog.predict_species(
            active_species or "",
            roster,
            limit=limit,
            revealed_moves=revealed_moves,
            item=item,
            ability=ability,
            tera_type=tera_type,
        )
        if active_species
        else []
    )
    if predictions:
        best = predictions[0]
        role_probs = [_role_probability(predictions, role_name) for role_name in TEAM_PREVIEW_ROLE_ORDER]
        return [
            roster_size_norm, roster_match_rate, 1.0,
            best.probability, best.species_prior,
            math.tanh(best.team_boost),
            _normalized_support_count(best.template.support_count),
            *role_probs,
            closure_score, has_hazard, has_removal,
            float(avg_synergy), float(max_synergy), float(min_synergy),
            float(top_lead_score),
        ]

    role_probs = [0.0] * len(TEAM_PREVIEW_ROLE_ORDER)
    return [
        roster_size_norm, roster_match_rate,
        0.0, 0.0, 0.0, 0.0, 0.0,
        *role_probs,
        closure_score, has_hazard, has_removal,
        float(avg_synergy), float(max_synergy), float(min_synergy),
        float(top_lead_score),
    ]


__all__ = [
    "PreviewSetObservation",
    "TEAM_PREVIEW_FEATURE_DIM",
    "TEAM_PREVIEW_ROLE_ORDER",
    "TeamPreviewExample",
    "TeamPreviewPriorCatalog",
    "TeamPreviewTemplate",
    "TeamPreviewTemplatePrediction",
    "build_team_preview_prior",
    "build_team_preview_prior_from_processed",
    "iter_team_preview_examples",
    "iter_team_preview_examples_from_processed",
    "load_team_preview_prior_catalog",
    "team_preview_feature_vector",
]