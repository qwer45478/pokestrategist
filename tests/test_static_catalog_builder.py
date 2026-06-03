from __future__ import annotations

from pokestrategist.data.static_catalog_builder import ActiveSubsetConfig
from pokestrategist.data.static_catalog_builder import ObservedUsage
from pokestrategist.data.static_catalog_builder import build_active_subset
from pokestrategist.data.static_catalog_builder import build_moves_catalog
from pokestrategist.data.static_catalog_builder import parse_moveset_usage
from pokestrategist.data.static_catalog_builder import parse_showdown_js_export
from pokestrategist.data.static_catalog_builder import parse_species_usage


def test_parse_showdown_js_export_quotes_nested_keys_without_touching_strings():
    payload = 'exports.BattleItems = {heavydutyboots:{name:"Heavy-Duty Boots",fling:{basePower:80},desc:"Text with, fake:key inside."}};'

    parsed = parse_showdown_js_export(payload, "BattleItems")

    assert parsed["heavydutyboots"]["name"] == "Heavy-Duty Boots"
    assert parsed["heavydutyboots"]["fling"]["basePower"] == 80
    assert parsed["heavydutyboots"]["desc"] == "Text with, fake:key inside."


def test_parse_smogon_usage_and_moveset_tables():
    usage_text = """
+ ---- + ------------------ + --------- + ------ + ------- + ------ + ------- +
| Rank | Pokemon            | Usage %   | Raw    | %       | Real   | %       |
+ ---- + ------------------ + --------- + ------ + ------- + ------ + ------- +
| 1    | Great Tusk         | 27.86978% | 632915 | 27.870% | 499694 | 27.846% |
| 2    | Kingambit          | 20.27599% | 460462 | 20.276% | 307849 | 17.155% |
"""
    moveset_text = """
+----------------------------------------+
| Great Tusk                             |
+----------------------------------------+
| Raw count: 668457                      |
+----------------------------------------+
| Abilities                              |
| Protosynthesis 100.000%                |
+----------------------------------------+
| Items                                  |
| Booster Energy 29.517%                 |
| Other 4.094%                           |
+----------------------------------------+
| Moves                                  |
| Rapid Spin 91.960%                     |
| Headlong Rush 78.201%                  |
| Other 18.638%                          |
+----------------------------------------+
| Tera Types                             |
| Steel 25.677%                          |
+----------------------------------------+
"""

    usage_rows = parse_species_usage(usage_text)
    movesets = parse_moveset_usage(moveset_text)

    assert usage_rows[0].species == "Great Tusk"
    assert usage_rows[0].usage_percent == 27.86978
    assert movesets["Great Tusk"].raw_count == 668457
    assert [entry.name for entry in movesets["Great Tusk"].moves] == ["Rapid Spin", "Headlong Rush"]
    assert movesets["Great Tusk"].items[0].name == "Booster Energy"


def test_build_catalog_and_active_subset_preserve_observed_tail():
    move_payload = {
        "rapidspin": {
            "num": 229,
            "accuracy": 100,
            "basePower": 50,
            "category": "Physical",
            "name": "Rapid Spin",
            "pp": 40,
            "priority": 0,
            "flags": {"contact": 1},
            "target": "normal",
            "type": "Normal",
            "desc": "Hazards are removed from the user's side. Raises the user's Speed by 1 stage.",
        },
        "ancientpastmove": {
            "num": 1,
            "isNonstandard": "Past",
            "accuracy": 100,
            "basePower": 40,
            "category": "Physical",
            "name": "Ancient Past Move",
            "pp": 20,
            "priority": 0,
            "flags": {},
            "target": "normal",
            "type": "Normal",
        },
    }
    usage_rows = parse_species_usage("| 1 | Great Tusk | 1.00000% | 1000 | 1.000% | 900 | 1.000% |\n")
    movesets = parse_moveset_usage(
        """
+----------------------------------------+
| Great Tusk                             |
+----------------------------------------+
| Raw count: 1000                        |
+----------------------------------------+
| Abilities                              |
| Protosynthesis 100.000%                |
+----------------------------------------+
| Items                                  |
| Booster Energy 29.517%                 |
+----------------------------------------+
| Moves                                  |
| Rapid Spin 91.960%                     |
+----------------------------------------+
"""
    )
    observed = ObservedUsage()
    observed.moves["Rare Replay Move"] = 3

    moves = build_moves_catalog(move_payload)
    subset = build_active_subset(usage_rows, movesets, config=ActiveSubsetConfig(), observed_usage=observed)

    assert "Rapid Spin" in moves
    assert "Ancient Past Move" not in moves
    assert "removal" in moves["Rapid Spin"]["tags"]
    assert "speed_control" in moves["Rapid Spin"]["tags"]
    assert "Rare Replay Move" in subset["moves"]
    assert subset["summary"]["active_species"] == 1