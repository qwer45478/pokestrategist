"""Train-only hidden move prior used to expand honest candidate sets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from pokestrategist.data.schema import ActionHead, DecisionSample

ABSTRACT_FAMILY_ORDER = ("attack", "pivot", "recovery", "setup", "hazard", "status", "utility", "protect")


@dataclass(frozen=True, slots=True)
class HiddenMoveCandidate:
    move_token: str
    move_family: str
    prior_prob: float
    support_count: int


@dataclass(slots=True)
class HiddenMovePrior:
    exact_contexts: dict[str, list[HiddenMoveCandidate]]
    item_contexts: dict[str, list[HiddenMoveCandidate]]
    species_defaults: dict[str, list[HiddenMoveCandidate]]
    species_family_defaults: dict[str, list[tuple[str, float, int]]]
    global_family_defaults: list[tuple[str, float, int]]
    default_topk: int = 4

    def suggest(
        self,
        species: str | None,
        revealed_moves: Sequence[str],
        *,
        item: str | None = None,
        topk: int | None = None,
    ) -> list[HiddenMoveCandidate]:
        if not species:
            return []
        limit = topk or self.default_topk
        excluded = set(revealed_moves)
        ordered: list[HiddenMoveCandidate] = []
        seen: set[str] = set()
        for candidate in self.exact_contexts.get(_context_key(species, revealed_moves), []):
            if candidate.move_token in excluded or candidate.move_token in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate.move_token)
            if len(ordered) >= limit:
                break
        if len(ordered) < limit and item:
            for candidate in self.item_contexts.get(_item_context_key(species, item), []):
                if candidate.move_token in excluded or candidate.move_token in seen:
                    continue
                ordered.append(candidate)
                seen.add(candidate.move_token)
                if len(ordered) >= limit:
                    break
        if len(ordered) < limit:
            for candidate in self.species_defaults.get(species, []):
                if candidate.move_token in excluded or candidate.move_token in seen:
                    continue
                ordered.append(candidate)
                seen.add(candidate.move_token)
                if len(ordered) >= limit:
                    break
        if not ordered:
            return []
        total_support = sum(candidate.support_count for candidate in ordered)
        if total_support <= 0:
            return ordered
        return [
            HiddenMoveCandidate(
                move_token=candidate.move_token,
                move_family=candidate.move_family,
                prior_prob=candidate.support_count / total_support,
                support_count=candidate.support_count,
            )
            for candidate in ordered
        ]

    def abstract_families(self, species: str | None, *, limit: int) -> list[tuple[str, float, int]]:
        if limit <= 0:
            return []
        families = self.species_family_defaults.get(species or "", self.global_family_defaults)
        if not families:
            return [(family, 0.0, 0) for family in ABSTRACT_FAMILY_ORDER[:limit]]
        return families[:limit]

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_contexts": {key: [asdict(candidate) for candidate in value] for key, value in self.exact_contexts.items()},
            "item_contexts": {key: [asdict(candidate) for candidate in value] for key, value in self.item_contexts.items()},
            "species_defaults": {key: [asdict(candidate) for candidate in value] for key, value in self.species_defaults.items()},
            "species_family_defaults": {
                key: [{"move_family": family, "prior_prob": prob, "support_count": support} for family, prob, support in value]
                for key, value in self.species_family_defaults.items()
            },
            "global_family_defaults": [
                {"move_family": family, "prior_prob": prob, "support_count": support} for family, prob, support in self.global_family_defaults
            ],
            "default_topk": self.default_topk,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> HiddenMovePrior | None:
        if not payload:
            return None
        return cls(
            exact_contexts={
                key: [HiddenMoveCandidate(**candidate) for candidate in value]
                for key, value in dict(payload.get("exact_contexts", {})).items()
            },
            item_contexts={
                key: [HiddenMoveCandidate(**candidate) for candidate in value]
                for key, value in dict(payload.get("item_contexts", {})).items()
            },
            species_defaults={
                key: [HiddenMoveCandidate(**candidate) for candidate in value]
                for key, value in dict(payload.get("species_defaults", {})).items()
            },
            species_family_defaults={
                key: [(entry["move_family"], float(entry["prior_prob"]), int(entry["support_count"])) for entry in value]
                for key, value in dict(payload.get("species_family_defaults", {})).items()
            },
            global_family_defaults=[
                (entry["move_family"], float(entry["prior_prob"]), int(entry["support_count"]))
                for entry in list(payload.get("global_family_defaults", []))
            ],
            default_topk=int(payload.get("default_topk", 4)),
        )


def _context_key(species: str, revealed_moves: Sequence[str]) -> str:
    return f"{species}||{'|'.join(sorted(revealed_moves))}"


def _item_context_key(species: str, item: str) -> str:
    return f"{species}||{item.strip().lower()}"


def _counter_to_candidates(counter: Counter[str], move_families: dict[str, str]) -> list[HiddenMoveCandidate]:
    total = sum(counter.values())
    if total <= 0:
        return []
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        HiddenMoveCandidate(
            move_token=move_token,
            move_family=move_families.get(move_token, "utility"),
            prior_prob=count / total,
            support_count=count,
        )
        for move_token, count in ordered
    ]


def _counter_to_family_defaults(counter: Counter[str]) -> list[tuple[str, float, int]]:
    total = sum(counter.values())
    if total <= 0:
        return []
    ordered = sorted(counter.items(), key=lambda item: (-item[1], ABSTRACT_FAMILY_ORDER.index(item[0]) if item[0] in ABSTRACT_FAMILY_ORDER else 99))
    return [(family, count / total, count) for family, count in ordered]


def build_hidden_move_prior(samples: Iterable[DecisionSample], *, default_topk: int = 4) -> HiddenMovePrior:
    exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    item_counts: dict[str, Counter[str]] = defaultdict(Counter)
    item_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    species_counts: dict[str, Counter[str]] = defaultdict(Counter)
    species_move_families: dict[str, dict[str, str]] = defaultdict(dict)
    species_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_family_counts: Counter[str] = Counter()

    for sample in samples:
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

    return HiddenMovePrior(
        exact_contexts={key: _counter_to_candidates(counter, exact_move_families[key]) for key, counter in exact_counts.items()},
        item_contexts={key: _counter_to_candidates(counter, item_move_families[key]) for key, counter in item_counts.items()},
        species_defaults={key: _counter_to_candidates(counter, species_move_families[key]) for key, counter in species_counts.items()},
        species_family_defaults={key: _counter_to_family_defaults(counter) for key, counter in species_family_counts.items()},
        global_family_defaults=_counter_to_family_defaults(global_family_counts),
        default_topk=default_topk,
    )