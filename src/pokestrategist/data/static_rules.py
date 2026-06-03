"""Static Gen 9 rule features for the decision model.

The catalog is intentionally deterministic: type charts, base stats, move
attributes, ability tags, and item tags are fixed game knowledge rather than
replay-derived labels. Missing entries fall back to zero/neutral features so the
current JSONL datasets remain usable while the catalog is expanded.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TYPE_NAMES = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
)
TYPE_TO_INDEX = {name: index for index, name in enumerate(TYPE_NAMES)}
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
MOVE_CATEGORIES = ("physical", "special", "status", "unknown")
MOVE_TAGS = (
    "pivot",
    "hazard",
    "recovery",
    "setup",
    "protect",
    "status",
    "removal",
    "contact",
    "boost",
    "speed_control",
    "priority_damage",
    "utility",
)
def normalize_name(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _repo_static_dir() -> Path:
    env_path = os.environ.get("POKEPILOT_STATIC_DATA")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[3] / "data" / "static" / "gen9"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_tag_vocab(filename: str) -> tuple[str, ...]:
    payload = _load_json(_repo_static_dir() / filename)
    tags = {
        str(tag).lower()
        for entry in payload.values()
        if isinstance(entry, dict)
        for tag in entry.get("tags", [])
        if str(tag).strip()
    }
    return tuple(sorted(tags))


ABILITY_TAGS = _catalog_tag_vocab("abilities.json")
ITEM_TAGS = _catalog_tag_vocab("items.json")

SPECIES_RULE_FEATURE_DIM = len(TYPE_NAMES) + len(STAT_NAMES) + 4
MOVE_RULE_FEATURE_DIM = len(TYPE_NAMES) + len(MOVE_CATEGORIES) + 3 + len(MOVE_TAGS)
MATCHUP_RULE_FEATURE_DIM = 16
ABILITY_RULE_FEATURE_DIM = 2 + len(ABILITY_TAGS)
ITEM_RULE_FEATURE_DIM = 2 + len(ITEM_TAGS)


def _normalized_mapping(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {normalize_name(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


@dataclass(frozen=True, slots=True)
class StaticRuleCatalog:
    type_chart: dict[str, dict[str, float]]
    species: dict[str, dict[str, Any]]
    moves: dict[str, dict[str, Any]]
    abilities: dict[str, dict[str, Any]]
    items: dict[str, dict[str, Any]]

    def species_entry(self, species_name: str | None) -> dict[str, Any] | None:
        return self.species.get(normalize_name(species_name))

    def move_entry(self, move_name: str | None) -> dict[str, Any] | None:
        return self.moves.get(normalize_name(move_name))

    def ability_entry(self, ability_name: str | None) -> dict[str, Any] | None:
        return self.abilities.get(normalize_name(ability_name))

    def item_entry(self, item_name: str | None) -> dict[str, Any] | None:
        return self.items.get(normalize_name(item_name))

    def species_types(self, species_name: str | None, tera_type: str | None = None) -> tuple[str, ...]:
        if tera_type:
            normalized_type = tera_type.strip().lower()
            if normalized_type in TYPE_TO_INDEX:
                return (normalized_type,)
        entry = self.species_entry(species_name)
        if not entry:
            return ()
        return tuple(type_name for type_name in entry.get("types", []) if type_name in TYPE_TO_INDEX)

    def base_speed(self, species_name: str | None) -> float:
        entry = self.species_entry(species_name)
        if not entry:
            return 0.0
        return float(dict(entry.get("base_stats", {})).get("spe", 0.0))

    def move_type(self, move_name: str | None) -> str | None:
        entry = self.move_entry(move_name)
        if not entry:
            return None
        move_type = str(entry.get("type", "")).lower()
        return move_type if move_type in TYPE_TO_INDEX else None

    def type_multiplier(self, attack_type: str | None, defender_types: tuple[str, ...]) -> float:
        if attack_type not in TYPE_TO_INDEX or not defender_types:
            return 1.0
        chart_row = self.type_chart.get(str(attack_type), {})
        multiplier = 1.0
        for defender_type in defender_types:
            multiplier *= float(chart_row.get(defender_type, 1.0))
        return multiplier


@lru_cache(maxsize=1)
def load_static_rule_catalog() -> StaticRuleCatalog:
    static_dir = _repo_static_dir()
    return StaticRuleCatalog(
        type_chart={
            attack_type: {defender_type: float(value) for defender_type, value in row.items()}
            for attack_type, row in _load_json(static_dir / "type_chart.json").items()
        },
        species=_normalized_mapping(_load_json(static_dir / "species.json")),
        moves=_normalized_mapping(_load_json(static_dir / "moves.json")),
        abilities=_normalized_mapping(_load_json(static_dir / "abilities.json")),
        items=_normalized_mapping(_load_json(static_dir / "items.json")),
    )


def _type_one_hot(types: tuple[str, ...]) -> list[float]:
    values = [0.0] * len(TYPE_NAMES)
    for type_name in types:
        type_index = TYPE_TO_INDEX.get(type_name)
        if type_index is not None:
            values[type_index] = 1.0
    return values


def species_rule_features(species_name: str | None) -> list[float]:
    catalog = load_static_rule_catalog()
    entry = catalog.species_entry(species_name)
    if not entry:
        return [0.0] * SPECIES_RULE_FEATURE_DIM

    stats = dict(entry.get("base_stats", {}))
    stat_values = [float(stats.get(stat_name, 0.0)) for stat_name in STAT_NAMES]
    atk = max(stat_values[1], 0.0)
    spa = max(stat_values[3], 0.0)
    offense_total = max(atk + spa, 1.0)
    features = _type_one_hot(catalog.species_types(species_name))
    features.extend(min(value / 255.0, 1.0) for value in stat_values)
    features.extend(
        [
            min(sum(stat_values) / 720.0, 1.0),
            atk / offense_total,
            spa / offense_total,
            min((stat_values[0] + stat_values[2] + stat_values[4]) / (255.0 * 3.0), 1.0),
        ]
    )
    return features


def _family_tags(move_family: str | None) -> set[str]:
    family = (move_family or "").lower()
    if family == "attack":
        return set()
    if family in MOVE_TAGS:
        return {family}
    if family == "hazard_control":
        return {"removal", "utility"}
    return set()


def move_rule_features(move_name: str | None, move_family: str | None = None) -> list[float]:
    catalog = load_static_rule_catalog()
    entry = catalog.move_entry(move_name)
    move_type = str(entry.get("type", "")).lower() if entry else None
    if move_type not in TYPE_TO_INDEX:
        move_type = None
    category = str(entry.get("category", "unknown")).lower() if entry else "unknown"
    if category not in MOVE_CATEGORIES:
        category = "unknown"
    power = float(entry.get("power") or 0.0) if entry else 0.0
    accuracy_value = entry.get("accuracy") if entry else None
    accuracy = 1.0 if accuracy_value is None else max(0.0, min(float(accuracy_value) / 100.0, 1.0))
    priority = float(entry.get("priority") or 0.0) if entry else 0.0
    tags = set(str(tag).lower() for tag in entry.get("tags", [])) if entry else set()
    tags.update(_family_tags(move_family))

    features = _type_one_hot((move_type,) if move_type else ())
    features.extend(1.0 if category == name else 0.0 for name in MOVE_CATEGORIES)
    features.extend([min(power / 150.0, 1.0), accuracy, (max(-7.0, min(priority, 7.0)) + 7.0) / 14.0])
    features.extend(1.0 if tag in tags else 0.0 for tag in MOVE_TAGS)
    return features


def ability_rule_features(ability_name: str | None) -> list[float]:
    catalog = load_static_rule_catalog()
    entry = catalog.ability_entry(ability_name)
    if not entry:
        return [0.0] * ABILITY_RULE_FEATURE_DIM
    tags = {str(tag).lower() for tag in entry.get("tags", [])}
    rating = max(0.0, min(float(entry.get("rating") or 0.0) / 5.0, 1.0))
    has_desc = 1.0 if entry.get("short_desc") else 0.0
    return [rating, has_desc] + [1.0 if tag in tags else 0.0 for tag in ABILITY_TAGS]


def item_rule_features(item_name: str | None) -> list[float]:
    catalog = load_static_rule_catalog()
    entry = catalog.item_entry(item_name)
    if not entry:
        return [0.0] * ITEM_RULE_FEATURE_DIM
    tags = {str(tag).lower() for tag in entry.get("tags", [])}
    gen_value = max(0.0, min(float(entry.get("gen") or 0.0) / 9.0, 1.0))
    has_desc = 1.0 if entry.get("short_desc") else 0.0
    return [gen_value, has_desc] + [1.0 if tag in tags else 0.0 for tag in ITEM_TAGS]


def _log_multiplier(multiplier: float) -> float:
    if multiplier <= 0.0:
        return -1.0
    return max(-1.0, min(math.log2(multiplier) / 2.0, 1.0))


def _speed_advantage(own_speed: float, opp_speed: float) -> float:
    if own_speed <= 0.0 or opp_speed <= 0.0:
        return 0.5
    if abs(own_speed - opp_speed) < 1e-6:
        return 0.5
    return 1.0 if own_speed > opp_speed else 0.0


def candidate_matchup_rule_features(action: Any, sample: Any) -> list[float]:
    catalog = load_static_rule_catalog()
    obs = sample.observation
    own = obs.self_side
    opp = obs.opp_side
    actor_species = action.switch_species or own.active_species
    opp_species = opp.active_species
    own_tera_type = own.revealed_tera_types.get(actor_species or "") if getattr(action, "tera", False) else None
    opp_tera_type = opp.revealed_tera_types.get(opp_species or "") if opp.tera_used else None
    actor_types = catalog.species_types(actor_species, tera_type=own_tera_type)
    opp_types = catalog.species_types(opp_species, tera_type=opp_tera_type)
    move_entry = catalog.move_entry(getattr(action, "move_token", None))
    move_type = catalog.move_type(getattr(action, "move_token", None))

    multiplier = 1.0
    stab = 0.0
    power_norm = 0.0
    accuracy = 1.0
    priority_norm = 0.5
    is_status = 0.0
    if move_entry:
        multiplier = catalog.type_multiplier(move_type, opp_types)
        stab = float(move_type in actor_types) if move_type else 0.0
        power_norm = min(float(move_entry.get("power") or 0.0) / 150.0, 1.0)
        accuracy_value = move_entry.get("accuracy")
        accuracy = 1.0 if accuracy_value is None else max(0.0, min(float(accuracy_value) / 100.0, 1.0))
        priority = float(move_entry.get("priority") or 0.0)
        priority_norm = (max(-7.0, min(priority, 7.0)) + 7.0) / 14.0
        is_status = float(str(move_entry.get("category", "")).lower() == "status")

    incoming_multipliers: list[float] = []
    if getattr(action, "head", None) is not None and getattr(action.head, "value", "") == "switch":
        for revealed_move in opp.revealed_moves.get(opp_species or "", []):
            incoming_type = catalog.move_type(revealed_move)
            incoming_multipliers.append(catalog.type_multiplier(incoming_type, actor_types))
    incoming_max = max(incoming_multipliers) if incoming_multipliers else 1.0
    incoming_mean = sum(incoming_multipliers) / len(incoming_multipliers) if incoming_multipliers else 1.0

    own_speed = catalog.base_speed(actor_species)
    opp_speed = catalog.base_speed(opp_species)
    known_species = float(bool(catalog.species_entry(actor_species)) and bool(catalog.species_entry(opp_species)))
    known_move = float(bool(move_entry) or getattr(action.head, "value", "") == "switch")
    return [
        _log_multiplier(multiplier),
        min(multiplier / 4.0, 1.0),
        stab,
        float(multiplier == 0.0),
        float(0.0 < multiplier < 1.0),
        float(multiplier > 1.0),
        power_norm,
        accuracy,
        priority_norm,
        is_status,
        _speed_advantage(own_speed, opp_speed),
        min(own_speed / 200.0, 1.0),
        min(opp_speed / 200.0, 1.0),
        min(incoming_max / 4.0, 1.0),
        min(incoming_mean / 4.0, 1.0),
        0.5 * (known_species + known_move),
    ]


__all__ = [
    "ABILITY_RULE_FEATURE_DIM",
    "ITEM_RULE_FEATURE_DIM",
    "MATCHUP_RULE_FEATURE_DIM",
    "MOVE_RULE_FEATURE_DIM",
    "SPECIES_RULE_FEATURE_DIM",
    "TYPE_NAMES",
    "ability_rule_features",
    "candidate_matchup_rule_features",
    "item_rule_features",
    "load_static_rule_catalog",
    "move_rule_features",
    "normalize_name",
    "species_rule_features",
]