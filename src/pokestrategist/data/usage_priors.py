"""Usage-driven priors derived from Smogon monthly stats.

These priors do not replace exact legal-action reasoning. Instead they inject a
meta prior into candidate proposal and candidate scoring, while fixed rule facts
continue to come from the static Gen 9 catalog.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pokepilot.data.static_rules import normalize_name
from pokepilot.data.static_rules import load_static_rule_catalog

ABSTRACT_FAMILY_ORDER = ("attack", "pivot", "recovery", "setup", "hazard", "status", "utility", "protect")
USAGE_PRIOR_FEATURE_DIM = 4


def _repo_usage_dir() -> Path:
    env_path = os.environ.get("POKEPILOT_USAGE_DATA")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[3] / "data" / "static" / "gen9ou"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_usage(value: float, *, cap: float) -> float:
    if value <= 0.0:
        return 0.0
    return min(math.log1p(value) / math.log1p(cap), 1.0)


def _normalized_entry_map(entries: Iterable[dict[str, Any]]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        normalized[normalize_name(name)] = float(entry.get("usage_percent") or 0.0)
    return normalized


def _entry_display_name_map(entries: Iterable[dict[str, Any]]) -> dict[str, str]:
    display: dict[str, str] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if name:
            display[normalize_name(name)] = name
    return display


def _infer_move_family(move_name: str | None) -> str:
    catalog = load_static_rule_catalog()
    entry = catalog.move_entry(move_name)
    if not entry:
        return "utility"
    tags = {str(tag).lower() for tag in entry.get("tags", [])}
    if "pivot" in tags:
        return "pivot"
    if "recovery" in tags:
        return "recovery"
    if "setup" in tags or "boost" in tags:
        return "setup"
    if "hazard" in tags:
        return "hazard"
    if "protect" in tags:
        return "protect"
    if "status" in tags:
        return "status"
    if str(entry.get("category", "")).lower() == "status":
        return "utility"
    if float(entry.get("power") or 0.0) > 0.0:
        return "attack"
    return "utility"


@dataclass(frozen=True, slots=True)
class UsageMoveCandidate:
    move_token: str
    move_family: str
    usage_percent: float
    usage_score: float


@dataclass(frozen=True, slots=True)
class UsageFamilyCandidate:
    move_family: str
    usage_percent: float
    usage_score: float


@dataclass(frozen=True, slots=True)
class SpeciesUsagePrior:
    usage_percent: float
    rank: int | None
    moves: dict[str, float]
    move_display_names: dict[str, str]
    items: dict[str, float]
    abilities: dict[str, float]
    tera_types: dict[str, float]
    family_usage: dict[str, float]


@dataclass(frozen=True, slots=True)
class UsagePriorCatalog:
    species: dict[str, SpeciesUsagePrior]

    def species_prior(self, species_name: str | None) -> SpeciesUsagePrior | None:
        return self.species.get(normalize_name(species_name))

    def species_usage_percent(self, species_name: str | None) -> float:
        prior = self.species_prior(species_name)
        return 0.0 if prior is None else prior.usage_percent

    def species_usage_score(self, species_name: str | None) -> float:
        return _normalize_usage(self.species_usage_percent(species_name), cap=100.0)

    def move_usage_percent(self, species_name: str | None, move_name: str | None) -> float:
        prior = self.species_prior(species_name)
        if prior is None:
            return 0.0
        return float(prior.moves.get(normalize_name(move_name), 0.0))

    def move_usage_score(self, species_name: str | None, move_name: str | None) -> float:
        return _normalize_usage(self.move_usage_percent(species_name, move_name), cap=100.0)

    def family_usage_percent(self, species_name: str | None, move_family: str | None) -> float:
        prior = self.species_prior(species_name)
        if prior is None:
            return 0.0
        return float(prior.family_usage.get((move_family or "").lower(), 0.0))

    def family_usage_score(self, species_name: str | None, move_family: str | None) -> float:
        return _normalize_usage(self.family_usage_percent(species_name, move_family), cap=250.0)

    def item_usage_score(self, species_name: str | None, item_name: str | None) -> float:
        prior = self.species_prior(species_name)
        if prior is None:
            return 0.0
        return _normalize_usage(float(prior.items.get(normalize_name(item_name), 0.0)), cap=100.0)

    def ability_usage_score(self, species_name: str | None, ability_name: str | None) -> float:
        prior = self.species_prior(species_name)
        if prior is None:
            return 0.0
        return _normalize_usage(float(prior.abilities.get(normalize_name(ability_name), 0.0)), cap=100.0)

    def tera_usage_score(self, species_name: str | None, tera_type: str | None) -> float:
        prior = self.species_prior(species_name)
        if prior is None:
            return 0.0
        return _normalize_usage(float(prior.tera_types.get(normalize_name(tera_type), 0.0)), cap=100.0)

    def set_context_score(
        self,
        species_name: str | None,
        *,
        item_name: str | None = None,
        ability_name: str | None = None,
        tera_type: str | None = None,
    ) -> float:
        scores = [
            self.item_usage_score(species_name, item_name),
            self.ability_usage_score(species_name, ability_name),
            self.tera_usage_score(species_name, tera_type),
        ]
        nonzero = [score for score in scores if score > 0.0]
        if not nonzero:
            return 0.0
        return sum(nonzero) / len(nonzero)

    def usage_prior_features(
        self,
        species_name: str | None,
        *,
        move_name: str | None = None,
        move_family: str | None = None,
        item_name: str | None = None,
        ability_name: str | None = None,
        tera_type: str | None = None,
    ) -> list[float]:
        return [
            self.species_usage_score(species_name),
            self.move_usage_score(species_name, move_name),
            self.family_usage_score(species_name, move_family),
            self.set_context_score(species_name, item_name=item_name, ability_name=ability_name, tera_type=tera_type),
        ]

    def top_moves(
        self,
        species_name: str | None,
        *,
        limit: int,
        exclude_moves: Iterable[str] = (),
    ) -> list[UsageMoveCandidate]:
        if limit <= 0:
            return []
        prior = self.species_prior(species_name)
        if prior is None:
            return []
        excluded = {normalize_name(move_name) for move_name in exclude_moves}
        ordered = sorted(prior.moves.items(), key=lambda item: (-item[1], prior.move_display_names.get(item[0], item[0])))
        candidates: list[UsageMoveCandidate] = []
        for normalized_move, usage_percent in ordered:
            if normalized_move in excluded:
                continue
            display_name = prior.move_display_names.get(normalized_move, normalized_move)
            candidates.append(
                UsageMoveCandidate(
                    move_token=display_name,
                    move_family=_infer_move_family(display_name),
                    usage_percent=usage_percent,
                    usage_score=_normalize_usage(usage_percent, cap=100.0),
                )
            )
            if len(candidates) >= limit:
                break
        return candidates

    def top_families(self, species_name: str | None, *, limit: int) -> list[UsageFamilyCandidate]:
        if limit <= 0:
            return []
        prior = self.species_prior(species_name)
        if prior is None:
            return []
        ordered = sorted(
            prior.family_usage.items(),
            key=lambda item: (-item[1], ABSTRACT_FAMILY_ORDER.index(item[0]) if item[0] in ABSTRACT_FAMILY_ORDER else 99),
        )
        return [
            UsageFamilyCandidate(
                move_family=family,
                usage_percent=usage_percent,
                usage_score=_normalize_usage(usage_percent, cap=250.0),
            )
            for family, usage_percent in ordered[:limit]
        ]


def _species_usage_table(payload: dict[str, Any]) -> dict[str, tuple[float, int | None]]:
    table: dict[str, tuple[float, int | None]] = {}
    for entry in payload.get("species", []):
        if not isinstance(entry, dict):
            continue
        species_name = str(entry.get("species") or "").strip()
        if not species_name:
            continue
        rank_value = entry.get("rank")
        rank = int(rank_value) if rank_value is not None else None
        table[normalize_name(species_name)] = (float(entry.get("usage_percent") or 0.0), rank)
    return table


def _build_species_prior(entry: dict[str, Any], usage_percent: float, rank: int | None) -> SpeciesUsagePrior:
    move_entries = list(entry.get("moves", [])) if isinstance(entry.get("moves"), list) else []
    family_usage: dict[str, float] = {family: 0.0 for family in ABSTRACT_FAMILY_ORDER}
    for move_entry in move_entries:
        if not isinstance(move_entry, dict):
            continue
        move_name = str(move_entry.get("name") or "").strip()
        if not move_name:
            continue
        usage = float(move_entry.get("usage_percent") or 0.0)
        family = _infer_move_family(move_name)
        family_usage[family] = family_usage.get(family, 0.0) + usage
    return SpeciesUsagePrior(
        usage_percent=usage_percent,
        rank=rank,
        moves=_normalized_entry_map(move_entries),
        move_display_names=_entry_display_name_map(move_entries),
        items=_normalized_entry_map(list(entry.get("items", [])) if isinstance(entry.get("items"), list) else []),
        abilities=_normalized_entry_map(list(entry.get("abilities", [])) if isinstance(entry.get("abilities"), list) else []),
        tera_types=_normalized_entry_map(list(entry.get("tera_types", [])) if isinstance(entry.get("tera_types"), list) else []),
        family_usage=family_usage,
    )


@lru_cache(maxsize=4)
def load_usage_prior_catalog(data_dir: str | Path | None = None) -> UsagePriorCatalog | None:
    usage_dir = Path(data_dir) if data_dir is not None else _repo_usage_dir()
    usage_payload = _load_json(usage_dir / "usage.json")
    movesets = usage_payload.get("movesets")
    if not isinstance(movesets, dict) or not movesets:
        return None
    species_usage = _species_usage_table(usage_payload)
    species: dict[str, SpeciesUsagePrior] = {}
    for species_name, entry in movesets.items():
        if not isinstance(entry, dict):
            continue
        normalized_species = normalize_name(species_name)
        usage_percent, rank = species_usage.get(normalized_species, (0.0, None))
        species[normalized_species] = _build_species_prior(entry, usage_percent, rank)
    return UsagePriorCatalog(species=species)


__all__ = [
    "ABSTRACT_FAMILY_ORDER",
    "USAGE_PRIOR_FEATURE_DIM",
    "UsageFamilyCandidate",
    "UsageMoveCandidate",
    "UsagePriorCatalog",
    "load_usage_prior_catalog",
]