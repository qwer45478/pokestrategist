"""Build static Pokemon rule catalogs and usage-based active subsets.

Pokemon Showdown is the canonical machine-readable source for fixed rule facts,
while Smogon usage stats describe the Gen 9 OU metagame tail. This module keeps
those two responsibilities separate: full facts stay on disk, and usage cutoffs
only define a smaller active subset for priors/candidate generation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests

from pokestrategist.data.static_rules import normalize_name

SHOWDOWN_POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"
SHOWDOWN_MOVES_URL = "https://play.pokemonshowdown.com/data/moves.json"
SHOWDOWN_ITEMS_URL = "https://play.pokemonshowdown.com/data/items.js"
SHOWDOWN_ABILITIES_URL = "https://play.pokemonshowdown.com/data/abilities.js"
SMOGON_STATS_ROOT = "https://www.smogon.com/stats"

NONSTANDARD_VALUES = {"past", "future", "custom", "cap", "lgpe", "unobtainable", "gigantamax"}
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True, slots=True)
class SpeciesUsageRow:
    rank: int
    species: str
    usage_percent: float
    raw_count: int
    raw_percent: float
    real_count: int
    real_percent: float


@dataclass(frozen=True, slots=True)
class MovesetEntry:
    name: str
    usage_percent: float


@dataclass(slots=True)
class SpeciesMovesetUsage:
    species: str
    raw_count: int | None = None
    abilities: list[MovesetEntry] = field(default_factory=list)
    items: list[MovesetEntry] = field(default_factory=list)
    moves: list[MovesetEntry] = field(default_factory=list)
    tera_types: list[MovesetEntry] = field(default_factory=list)


@dataclass(slots=True)
class ObservedUsage:
    species: Counter[str] = field(default_factory=Counter)
    moves: Counter[str] = field(default_factory=Counter)
    items: Counter[str] = field(default_factory=Counter)
    abilities: Counter[str] = field(default_factory=Counter)
    samples_scanned: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "samples_scanned": self.samples_scanned,
            "species": dict(self.species.most_common()),
            "moves": dict(self.moves.most_common()),
            "items": dict(self.items.most_common()),
            "abilities": dict(self.abilities.most_common()),
        }


@dataclass(frozen=True, slots=True)
class ActiveSubsetConfig:
    species_min_usage: float = 0.05
    species_min_raw_count: int = 100
    move_min_usage: float = 1.0
    item_min_usage: float = 1.0
    ability_min_usage: float = 1.0
    max_species: int | None = None


def previous_month(today: date | None = None) -> str:
    current = today or date.today()
    year = current.year
    month = current.month - 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}-{month:02d}"


def smogon_usage_url(month: str, formatid: str, rating_cutoff: int = 0) -> str:
    return f"{SMOGON_STATS_ROOT}/{month}/{formatid}-{rating_cutoff}.txt"


def smogon_moveset_url(month: str, formatid: str, rating_cutoff: int = 0) -> str:
    return f"{SMOGON_STATS_ROOT}/{month}/moveset/{formatid}-{rating_cutoff}.txt"


def _session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.setdefault("User-Agent", user_agent)
    return session


def fetch_text(session: requests.Session, url: str, *, timeout: float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _extract_exported_object(js_text: str, export_name: str) -> str:
    marker = f"exports.{export_name} ="
    marker_index = js_text.find(marker)
    if marker_index < 0:
        raise ValueError(f"missing {marker!r} in Showdown JS payload")
    start = js_text.find("{", marker_index + len(marker))
    if start < 0:
        raise ValueError(f"missing object start for {export_name}")

    depth = 0
    in_string = False
    string_quote = ""
    escaped = False
    for index in range(start, len(js_text)):
        char = js_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            string_quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return js_text[start : index + 1]
    raise ValueError(f"unterminated object for {export_name}")


def _is_identifier_start(char: str) -> bool:
    return char == "_" or char == "$" or char.isalpha()


def _is_identifier_char(char: str) -> bool:
    return char == "_" or char == "$" or char.isalnum()


def _quote_js_object_keys(js_object: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    string_quote = ""
    escaped = False
    while index < len(js_object):
        char = js_object[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
            index += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_quote = char
            output.append(char)
            index += 1
            continue

        if char in "{,":
            output.append(char)
            index += 1
            while index < len(js_object) and js_object[index].isspace():
                output.append(js_object[index])
                index += 1
            if index < len(js_object) and _is_identifier_start(js_object[index]):
                key_start = index
                index += 1
                while index < len(js_object) and _is_identifier_char(js_object[index]):
                    index += 1
                key_end = index
                lookahead = index
                while lookahead < len(js_object) and js_object[lookahead].isspace():
                    lookahead += 1
                if lookahead < len(js_object) and js_object[lookahead] == ":":
                    output.append(json.dumps(js_object[key_start:key_end]))
                    output.append(js_object[key_end:lookahead])
                    index = lookahead
                    continue
                output.append(js_object[key_start:key_end])
            continue

        if js_object.startswith("undefined", index):
            before = js_object[index - 1] if index > 0 else ""
            after_index = index + len("undefined")
            after = js_object[after_index] if after_index < len(js_object) else ""
            if not _is_identifier_char(before) and not _is_identifier_char(after):
                output.append("null")
                index = after_index
                continue

        output.append(char)
        index += 1
    return "".join(output)


def parse_showdown_js_export(js_text: str, export_name: str) -> dict[str, Any]:
    js_object = _extract_exported_object(js_text, export_name)
    return json.loads(_quote_js_object_keys(js_object))


def parse_species_usage(text: str) -> list[SpeciesUsageRow]:
    rows: list[SpeciesUsageRow] = []
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*([0-9.]+)%\s*\|\s*(\d+)\s*\|\s*([0-9.]+)%\s*\|\s*(\d+)\s*\|\s*([0-9.]+)%\s*\|"
    )
    for raw_line in text.splitlines():
        match = pattern.match(raw_line)
        if not match:
            continue
        rows.append(
            SpeciesUsageRow(
                rank=int(match.group(1)),
                species=match.group(2).strip(),
                usage_percent=float(match.group(3)),
                raw_count=int(match.group(4)),
                raw_percent=float(match.group(5)),
                real_count=int(match.group(6)),
                real_percent=float(match.group(7)),
            )
        )
    return rows


def _parse_moveset_entry(content: str) -> MovesetEntry | None:
    if content.startswith("Other "):
        return None
    match = re.match(r"^(.*?)\s+([0-9.]+)%$", content)
    if not match:
        return None
    return MovesetEntry(name=match.group(1).strip(), usage_percent=float(match.group(2)))


def parse_moveset_usage(text: str) -> dict[str, SpeciesMovesetUsage]:
    lines = text.splitlines()
    usage: dict[str, SpeciesMovesetUsage] = {}
    current: SpeciesMovesetUsage | None = None
    section: str | None = None
    section_map = {"Abilities": "abilities", "Items": "items", "Moves": "moves", "Tera Types": "tera_types"}

    stripped_line = [line.strip() for line in lines]
    for index, line in enumerate(stripped_line):
        if not line.startswith("|") or not line.endswith("|"):
            continue
        content = line.strip("| ")
        if not content:
            continue

        previous_is_rule = index > 0 and stripped_line[index - 1].startswith("+")
        next_is_rule = index + 1 < len(stripped_line) and stripped_line[index + 1].startswith("+")
        if previous_is_rule and next_is_rule and content not in section_map and ":" not in content:
            current = SpeciesMovesetUsage(species=content)
            usage[content] = current
            section = None
            continue

        if current is None:
            continue
        if content.startswith("Raw count:"):
            raw_count = content.split(":", 1)[1].strip()
            current.raw_count = int(raw_count) if raw_count.isdigit() else None
            continue
        if content in section_map:
            section = section_map[content]
            continue
        if content in {"Spreads", "Teammates", "Checks and Counters"}:
            section = None
            continue
        if section is None:
            continue
        entry = _parse_moveset_entry(content)
        if entry is not None:
            getattr(current, section).append(entry)
    return usage


def _is_nonstandard(entry: dict[str, Any]) -> bool:
    value = str(entry.get("isNonstandard", "")).lower()
    return bool(value) and value in NONSTANDARD_VALUES


def _standard_name(entry: dict[str, Any], fallback_key: str) -> str:
    return str(entry.get("name") or fallback_key)


def _include_species(entry: dict[str, Any], *, include_nonstandard: bool) -> bool:
    if not include_nonstandard and _is_nonstandard(entry):
        return False
    if int(entry.get("num") or 0) <= 0:
        return False
    return bool(entry.get("types")) and bool(entry.get("baseStats"))


def _include_move(entry: dict[str, Any], *, include_nonstandard: bool) -> bool:
    if not include_nonstandard and _is_nonstandard(entry):
        return False
    if int(entry.get("num") or 0) <= 0:
        return False
    if entry.get("isZ") or entry.get("isMax"):
        return include_nonstandard
    return bool(entry.get("type")) and bool(entry.get("category"))


def _include_item(entry: dict[str, Any], *, include_nonstandard: bool) -> bool:
    if not include_nonstandard and _is_nonstandard(entry):
        return False
    if int(entry.get("num") or 0) <= 0:
        return False
    if entry.get("isPokeball"):
        return False
    return bool(entry.get("name"))


def _include_ability(entry: dict[str, Any], *, include_nonstandard: bool) -> bool:
    if not include_nonstandard and _is_nonstandard(entry):
        return False
    if int(entry.get("num") or 0) <= 0:
        return False
    return bool(entry.get("name"))


def _ability_values(entry: dict[str, Any]) -> list[str]:
    abilities = entry.get("abilities") or {}
    if not isinstance(abilities, dict):
        return []
    return [str(value) for _, value in sorted(abilities.items()) if value]


def build_species_catalog(
    payload: dict[str, Any],
    usage_rows: Iterable[SpeciesUsageRow] = (),
    *,
    include_nonstandard: bool = False,
) -> dict[str, dict[str, Any]]:
    usage_by_name = {normalize_name(row.species): row for row in usage_rows}
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or not _include_species(value, include_nonstandard=include_nonstandard):
            continue
        name = _standard_name(value, key)
        stats = value.get("baseStats") or {}
        entry: dict[str, Any] = {
            "num": int(value.get("num") or 0),
            "types": [str(type_name).lower() for type_name in value.get("types", [])],
            "base_stats": {stat: int(stats.get(stat, 0) or 0) for stat in STAT_NAMES},
            "abilities": _ability_values(value),
            "tier": value.get("tier"),
            "gen": int(value.get("gen") or 0) if value.get("gen") is not None else None,
            "heightm": value.get("heightm"),
            "weightkg": value.get("weightkg"),
        }
        usage = usage_by_name.get(normalize_name(name))
        if usage:
            entry["gen9ou_usage_percent"] = usage.usage_percent
            entry["gen9ou_raw_count"] = usage.raw_count
            entry["gen9ou_rank"] = usage.rank
        catalog[name] = entry
    return dict(sorted(catalog.items(), key=lambda item: (item[1].get("num") or 0, item[0])))


def _desc_blob(entry: dict[str, Any]) -> str:
    return " ".join(str(entry.get(key, "")) for key in ("name", "shortDesc", "desc")).lower()


def infer_move_tags(entry: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    flags = entry.get("flags") if isinstance(entry.get("flags"), dict) else {}
    category = str(entry.get("category", "")).lower()
    base_power = float(entry.get("basePower") or 0.0)
    priority = float(entry.get("priority") or 0.0)
    desc = _desc_blob(entry)

    if base_power > 0:
        tags.add("attack")
    if flags.get("contact"):
        tags.add("contact")
    if entry.get("selfSwitch") or " switches out" in desc or "user switches" in desc:
        tags.add("pivot")
    if entry.get("sideCondition") in {"stealthrock", "spikes", "toxicspikes", "stickyweb"} or "sets up a hazard" in desc:
        tags.add("hazard")
    if "hazards are removed" in desc or "removes" in desc and "hazard" in desc or entry.get("name") in {"Defog", "Rapid Spin", "Mortal Spin", "Tidy Up"}:
        tags.add("removal")
        tags.add("utility")
    if entry.get("heal") or entry.get("drain") or flags.get("heal") or "restores" in desc or "recovers" in desc:
        tags.add("recovery")
    if entry.get("boosts") or entry.get("selfBoost") or "raises the user's" in desc:
        tags.add("boost")
    if category == "status" and (entry.get("boosts") or entry.get("selfBoost") or "raises the user's" in desc):
        tags.add("setup")
    if entry.get("status") or " paralyze" in desc or " poison" in desc or " burn" in desc or " sleep" in desc:
        tags.add("status")
    if priority > 0 and base_power > 0:
        tags.add("priority_damage")
    if "speed" in desc or entry.get("sideCondition") == "tailwind" or priority > 0:
        tags.add("speed_control")
    if entry.get("stallingMove") or entry.get("volatileStatus") == "protect" or entry.get("name") in {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Burning Bulwark", "Silk Trap"}:
        tags.add("protect")
    if category == "status" and not {"hazard", "recovery", "setup", "protect"} & tags:
        tags.add("utility")
    return sorted(tags)


def build_moves_catalog(payload: dict[str, Any], *, include_nonstandard: bool = False) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or not _include_move(value, include_nonstandard=include_nonstandard):
            continue
        name = _standard_name(value, key)
        accuracy = value.get("accuracy")
        normalized_accuracy = None if accuracy is True else accuracy
        flags = value.get("flags") if isinstance(value.get("flags"), dict) else {}
        catalog[name] = {
            "num": int(value.get("num") or 0),
            "type": str(value.get("type", "")).lower(),
            "category": str(value.get("category", "unknown")).lower(),
            "power": int(value.get("basePower") or 0),
            "accuracy": normalized_accuracy,
            "priority": int(value.get("priority") or 0),
            "pp": int(value.get("pp") or 0),
            "target": value.get("target"),
            "flags": sorted(str(flag) for flag, enabled in flags.items() if enabled),
            "tags": infer_move_tags(value),
            "short_desc": value.get("shortDesc"),
        }
    return dict(sorted(catalog.items(), key=lambda item: (item[1].get("num") or 0, item[0])))


def infer_ability_tags(entry: dict[str, Any]) -> list[str]:
    name = str(entry.get("name", ""))
    desc = _desc_blob(entry)
    tags: set[str] = set()
    if name == "Levitate" or "ground" in desc and "immune" in desc:
        tags.add("ground_immunity")
    if "immune" in desc or "immunity" in desc:
        tags.add("immunity")
    if any(word in desc for word in ("burned", "paralyzed", "poisoned", "asleep", "status")) and "cannot" in desc:
        tags.add("status_immunity")
    if name in {"Drizzle", "Drought", "Sand Stream", "Snow Warning"} or "summons" in desc and any(word in desc for word in ("rain", "sun", "sandstorm", "snow")):
        tags.add("weather_setter")
    if name.endswith("Surge") or "terrain" in desc and "on switch-in" in desc:
        tags.add("terrain_setter")
    if "on switch-in" in desc or name in {"Intimidate", "Regenerator"}:
        tags.add("switch_trigger")
    if "power" in desc or "damage" in desc or "attack" in desc and "boost" in desc:
        tags.add("damage_modifier")
    if "speed" in desc and any(word in desc for word in ("doubled", "boost", "raised")):
        tags.add("speed_modifier")
    if "heals" in desc or "restores" in desc:
        tags.add("healing")
    if name in {"Good as Gold", "Magic Bounce"}:
        tags.add("status_control")
    return sorted(tags)


def build_abilities_catalog(payload: dict[str, Any], *, include_nonstandard: bool = False) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or not _include_ability(value, include_nonstandard=include_nonstandard):
            continue
        name = _standard_name(value, key)
        catalog[name] = {
            "num": int(value.get("num") or 0),
            "rating": value.get("rating"),
            "tags": infer_ability_tags(value),
            "short_desc": value.get("shortDesc"),
        }
    return dict(sorted(catalog.items(), key=lambda item: (item[1].get("num") or 0, item[0])))


def infer_item_tags(entry: dict[str, Any]) -> list[str]:
    name = str(entry.get("name", ""))
    desc = _desc_blob(entry)
    tags: set[str] = set()
    if name == "Heavy-Duty Boots" or "entry hazards" in desc and "not affected" in desc:
        tags.add("hazard_immunity")
    if name.startswith("Choice "):
        tags.add("choice_lock")
        tags.add("damage_boost")
    if name == "Choice Scarf" or "speed" in desc and "1.5x" in desc:
        tags.add("speed_boost")
    if name in {"Leftovers", "Black Sludge"} or "restores" in desc and "hp" in desc:
        tags.add("passive_recovery")
    if name == "Booster Energy":
        tags.add("one_time_stat_boost")
    if name == "Focus Sash":
        tags.add("survival_once")
        tags.add("lead_item")
    if name == "Life Orb" or "1.3x" in desc or "1.2x" in desc:
        tags.add("damage_boost")
    if name == "Air Balloon":
        tags.add("temporary_ground_immunity")
    if name == "Rocky Helmet":
        tags.add("contact_punish")
    if name == "Assault Vest":
        tags.add("special_bulk")
        tags.add("status_lock")
    if name == "Loaded Dice":
        tags.add("multi_hit_boost")
    if name == "Clear Amulet":
        tags.add("stat_drop_immunity")
    if name in {"Eject Button", "Eject Pack", "Red Card"}:
        tags.add("forced_pivot")
    if entry.get("isBerry"):
        tags.add("berry")
    return sorted(tags)


def build_items_catalog(payload: dict[str, Any], *, include_nonstandard: bool = False) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or not _include_item(value, include_nonstandard=include_nonstandard):
            continue
        name = _standard_name(value, key)
        catalog[name] = {
            "num": int(value.get("num") or 0),
            "gen": int(value.get("gen") or 0) if value.get("gen") is not None else None,
            "tags": infer_item_tags(value),
            "short_desc": value.get("shortDesc"),
        }
    return dict(sorted(catalog.items(), key=lambda item: (item[1].get("num") or 0, item[0])))


def collect_observed_usage_from_processed(path: str | Path, *, max_samples: int | None = None) -> ObservedUsage:
    usage = ObservedUsage()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and usage.samples_scanned >= max_samples:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            usage.samples_scanned += 1
            observation = payload.get("observation") or {}
            for side_name in ("self_side", "opp_side"):
                side = observation.get(side_name) or {}
                _count_species_values(usage.species, [side.get("active_species")])
                _count_species_values(usage.species, side.get("team_order") or [])
                _count_nested_values(usage.moves, side.get("revealed_moves") or {})
                _count_species_values(usage.items, (side.get("revealed_items") or {}).values())
                _count_species_values(usage.abilities, (side.get("revealed_abilities") or {}).values())
            for action in [payload.get("our_action"), payload.get("opponent_action"), *(payload.get("legal_actions") or [])]:
                if not isinstance(action, dict):
                    continue
                if action.get("switch_species"):
                    usage.species[str(action["switch_species"])] += 1
                move_token = action.get("move_token")
                if move_token and not str(move_token).startswith("["):
                    usage.moves[str(move_token)] += 1
    return usage


def _count_species_values(counter: Counter[str], values: Iterable[Any]) -> None:
    for value in values:
        if value:
            counter[str(value)] += 1


def _count_nested_values(counter: Counter[str], mapping: dict[str, Any]) -> None:
    for values in mapping.values():
        if isinstance(values, list):
            _count_species_values(counter, values)


def _entries_above(entries: list[MovesetEntry], threshold: float) -> list[dict[str, object]]:
    return [asdict(entry) for entry in entries if entry.usage_percent >= threshold]


def build_active_subset(
    species_usage: list[SpeciesUsageRow],
    moveset_usage: dict[str, SpeciesMovesetUsage],
    *,
    config: ActiveSubsetConfig,
    observed_usage: ObservedUsage | None = None,
) -> dict[str, Any]:
    active_species_rows = [
        row
        for row in species_usage
        if row.usage_percent >= config.species_min_usage and row.raw_count >= config.species_min_raw_count
    ]
    if config.max_species is not None:
        active_species_rows = active_species_rows[: config.max_species]
    active_species = {row.species for row in active_species_rows}

    observed_usage = observed_usage or ObservedUsage()
    for species in observed_usage.species:
        active_species.add(species)

    active_moves: set[str] = set(observed_usage.moves.keys())
    active_items: set[str] = set(observed_usage.items.keys())
    active_abilities: set[str] = set(observed_usage.abilities.keys())
    per_species: dict[str, dict[str, object]] = {}
    for species in sorted(active_species, key=lambda name: (normalize_name(name))):
        moveset = moveset_usage.get(species)
        if not moveset:
            per_species[species] = {"moves": [], "items": [], "abilities": [], "tera_types": []}
            continue
        moves = _entries_above(moveset.moves, config.move_min_usage)
        items = _entries_above(moveset.items, config.item_min_usage)
        abilities = _entries_above(moveset.abilities, config.ability_min_usage)
        tera_types = _entries_above(moveset.tera_types, 0.0)
        active_moves.update(str(entry["name"]) for entry in moves)
        active_items.update(str(entry["name"]) for entry in items)
        active_abilities.update(str(entry["name"]) for entry in abilities)
        per_species[species] = {
            "raw_count": moveset.raw_count,
            "moves": moves,
            "items": items,
            "abilities": abilities,
            "tera_types": tera_types,
        }

    return {
        "policy": {
            "species_min_usage": config.species_min_usage,
            "species_min_raw_count": config.species_min_raw_count,
            "move_min_usage": config.move_min_usage,
            "item_min_usage": config.item_min_usage,
            "ability_min_usage": config.ability_min_usage,
            "max_species": config.max_species,
            "observed_actions_are_preserved": True,
        },
        "species": sorted(active_species, key=normalize_name),
        "moves": sorted(active_moves, key=normalize_name),
        "items": sorted(active_items, key=normalize_name),
        "abilities": sorted(active_abilities, key=normalize_name),
        "per_species": per_species,
        "summary": {
            "all_usage_species": len(species_usage),
            "active_species": len(active_species),
            "tail_species": max(0, len(species_usage) - len(active_species_rows)),
            "active_moves": len(active_moves),
            "active_items": len(active_items),
            "active_abilities": len(active_abilities),
            "observed_samples_scanned": observed_usage.samples_scanned,
        },
    }


def build_catalogs_from_sources(
    *,
    month: str,
    formatid: str,
    rating_cutoff: int,
    timeout: float,
    user_agent: str,
    include_nonstandard: bool = False,
    active_config: ActiveSubsetConfig | None = None,
    observed_usage: ObservedUsage | None = None,
) -> dict[str, Any]:
    session = _session(user_agent)
    pokedex_payload = json.loads(fetch_text(session, SHOWDOWN_POKEDEX_URL, timeout=timeout))
    moves_payload = json.loads(fetch_text(session, SHOWDOWN_MOVES_URL, timeout=timeout))
    items_payload = parse_showdown_js_export(fetch_text(session, SHOWDOWN_ITEMS_URL, timeout=timeout), "BattleItems")
    abilities_payload = parse_showdown_js_export(fetch_text(session, SHOWDOWN_ABILITIES_URL, timeout=timeout), "BattleAbilities")
    usage_text = fetch_text(session, smogon_usage_url(month, formatid, rating_cutoff), timeout=timeout)
    moveset_text = fetch_text(session, smogon_moveset_url(month, formatid, rating_cutoff), timeout=timeout)
    species_usage = parse_species_usage(usage_text)
    moveset_usage = parse_moveset_usage(moveset_text)
    active_config = active_config or ActiveSubsetConfig()

    species = build_species_catalog(pokedex_payload, species_usage, include_nonstandard=include_nonstandard)
    moves = build_moves_catalog(moves_payload, include_nonstandard=include_nonstandard)
    items = build_items_catalog(items_payload, include_nonstandard=include_nonstandard)
    abilities = build_abilities_catalog(abilities_payload, include_nonstandard=include_nonstandard)
    active_subset = build_active_subset(species_usage, moveset_usage, config=active_config, observed_usage=observed_usage)
    usage_payload = {
        "formatid": formatid,
        "month": month,
        "rating_cutoff": rating_cutoff,
        "species": [asdict(row) for row in species_usage],
        "movesets": {
            species_name: {
                "raw_count": usage.raw_count,
                "abilities": [asdict(entry) for entry in usage.abilities],
                "items": [asdict(entry) for entry in usage.items],
                "moves": [asdict(entry) for entry in usage.moves],
                "tera_types": [asdict(entry) for entry in usage.tera_types],
            }
            for species_name, usage in moveset_usage.items()
        },
    }
    manifest = {
        "formatid": formatid,
        "month": month,
        "rating_cutoff": rating_cutoff,
        "sources": {
            "pokedex": SHOWDOWN_POKEDEX_URL,
            "moves": SHOWDOWN_MOVES_URL,
            "items": SHOWDOWN_ITEMS_URL,
            "abilities": SHOWDOWN_ABILITIES_URL,
            "usage": smogon_usage_url(month, formatid, rating_cutoff),
            "moveset": smogon_moveset_url(month, formatid, rating_cutoff),
        },
        "counts": {
            "species": len(species),
            "moves": len(moves),
            "items": len(items),
            "abilities": len(abilities),
            "usage_species": len(species_usage),
            **active_subset["summary"],
        },
        "active_subset_policy": active_subset["policy"],
        "full_catalog_does_not_expand_tensor_dims": True,
    }
    return {
        "species": species,
        "moves": moves,
        "items": items,
        "abilities": abilities,
        "usage": usage_payload,
        "active_subset": active_subset,
        "manifest": manifest,
        "observed_usage": (observed_usage or ObservedUsage()).to_dict(),
    }


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ActiveSubsetConfig",
    "ObservedUsage",
    "build_abilities_catalog",
    "build_active_subset",
    "build_catalogs_from_sources",
    "build_items_catalog",
    "build_moves_catalog",
    "build_species_catalog",
    "collect_observed_usage_from_processed",
    "parse_moveset_usage",
    "parse_showdown_js_export",
    "parse_species_usage",
    "previous_month",
    "smogon_moveset_url",
    "smogon_usage_url",
    "write_json",
]