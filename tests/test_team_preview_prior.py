from __future__ import annotations

import copy
import json

from pokestrategist.data.team_preview_prior import (
    PreviewSetObservation,
    TEAM_PREVIEW_FEATURE_DIM,
    TeamPreviewExample,
    build_team_preview_prior,
    iter_team_preview_examples_from_processed,
    team_preview_feature_vector,
)

from tests.v1_fixtures import write_dataset


def test_iter_team_preview_examples_from_processed_collects_hindsight(tmp_path):
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("preview-seed",))
    base_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seed_row = base_rows[0]

    row_one = copy.deepcopy(seed_row)
    row_one["replay_id"] = "preview-a"
    row_one["observation"]["replay_id"] = "preview-a"
    row_one["perspective"] = "p1"
    row_one["observation"]["perspective"] = "p1"
    row_one["observation"]["opp_side"]["team_order"] = ["Dragapult", "Great Tusk", "Kingambit"]
    row_one["observation"]["opp_side"]["active_species"] = "Dragapult"
    row_one["observation"]["opp_side"]["revealed_moves"] = {"Dragapult": ["Hex"]}
    row_one["observation"]["opp_side"]["revealed_items"] = {"Dragapult": "Heavy-Duty Boots"}
    row_one["opponent_action"] = {
        "actor": "p2",
        "head": "move",
        "move_token": "U-turn",
        "move_family": "pivot",
        "tera": False,
        "switch_slot": None,
        "switch_species": None,
        "candidate_source": None,
        "candidate_prior_prob": 0.0,
        "candidate_support_score": 0.0,
        "candidate_is_gold_injected": False,
    }

    row_two = copy.deepcopy(row_one)
    row_two["observation"]["opp_side"]["revealed_abilities"] = {"Dragapult": "Infiltrator"}
    row_two["observation"]["opp_side"]["revealed_tera_types"] = {"Dragapult": "Ghost"}
    row_two["opponent_action"]["move_token"] = "Draco Meteor"
    row_two["opponent_action"]["move_family"] = "attack"

    preview_path = tmp_path / "preview_processed.jsonl"
    preview_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in (row_one, row_two)) + "\n",
        encoding="utf-8",
    )

    examples = list(iter_team_preview_examples_from_processed(preview_path))

    assert len(examples) == 1
    example = examples[0]
    dragapult = example.observation_for_species("Dragapult")
    assert example.team_species == ("Dragapult", "Great Tusk", "Kingambit")
    assert dragapult is not None
    assert set(dragapult.moves) == {"Hex", "U-turn", "Draco Meteor"}
    assert dragapult.item == "Heavy-Duty Boots"
    assert dragapult.ability == "Infiltrator"
    assert dragapult.tera_type == "Ghost"


def test_team_preview_prior_reranks_templates_by_teammates():
    examples = [
        TeamPreviewExample(
            team_species=("Dragapult", "Great Tusk", "Kingambit"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Hex", "U-turn", "Draco Meteor"),
                    item="Heavy-Duty Boots",
                    ability="Infiltrator",
                    tera_type="Ghost",
                ),
            ),
        ),
        TeamPreviewExample(
            team_species=("Dragapult", "Great Tusk", "Cinderace"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Hex", "U-turn", "Draco Meteor"),
                    item="Heavy-Duty Boots",
                    ability="Infiltrator",
                    tera_type="Ghost",
                ),
            ),
        ),
        TeamPreviewExample(
            team_species=("Dragapult", "Corviknight", "Gholdengo"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Shadow Ball", "Draco Meteor", "Flamethrower"),
                    item="Choice Specs",
                    ability="Infiltrator",
                    tera_type="Dragon",
                ),
            ),
        ),
    ]

    catalog = build_team_preview_prior(examples)
    predictions = catalog.predict_species("Dragapult", ["Dragapult", "Great Tusk", "Kingambit"], limit=2)

    assert len(predictions) == 2
    assert predictions[0].template.item == "Heavy-Duty Boots"
    assert predictions[0].probability >= predictions[1].probability
    assert predictions[0].team_boost >= predictions[1].team_boost


def test_team_preview_prior_uses_revealed_evidence_to_break_template_ties():
    examples = [
        TeamPreviewExample(
            team_species=("Dragapult", "Great Tusk", "Kingambit"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Hex", "U-turn", "Draco Meteor"),
                    item="Heavy-Duty Boots",
                    ability="Infiltrator",
                    tera_type="Ghost",
                ),
            ),
        ),
        TeamPreviewExample(
            team_species=("Dragapult", "Great Tusk", "Kingambit"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Shadow Ball", "Draco Meteor", "Flamethrower"),
                    item="Choice Specs",
                    ability="Infiltrator",
                    tera_type="Dragon",
                ),
            ),
        ),
    ]

    catalog = build_team_preview_prior(examples)
    predictions = catalog.predict_species(
        "Dragapult",
        ["Dragapult", "Great Tusk", "Kingambit"],
        limit=2,
        revealed_moves=["Hex"],
        item="Heavy-Duty Boots",
    )

    assert len(predictions) == 2
    assert predictions[0].template.item == "Heavy-Duty Boots"
    assert predictions[0].reveal_consistency > predictions[1].reveal_consistency


def test_team_preview_feature_vector_exposes_active_template_signal():
    examples = [
        TeamPreviewExample(
            team_species=("Dragapult", "Great Tusk", "Kingambit"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Hex", "U-turn", "Draco Meteor"),
                    item="Heavy-Duty Boots",
                    ability="Infiltrator",
                    tera_type="Ghost",
                ),
            ),
        ),
        TeamPreviewExample(
            team_species=("Dragapult", "Corviknight", "Gholdengo"),
            set_observations=(
                PreviewSetObservation(
                    species="Dragapult",
                    moves=("Shadow Ball", "Draco Meteor", "Flamethrower"),
                    item="Choice Specs",
                    ability="Infiltrator",
                    tera_type="Dragon",
                ),
            ),
        ),
    ]

    catalog = build_team_preview_prior(examples)
    features = team_preview_feature_vector(
        catalog,
        ["Dragapult", "Great Tusk", "Kingambit"],
        active_species="Dragapult",
    )

    assert len(features) == TEAM_PREVIEW_FEATURE_DIM
    assert features[2] == 1.0
    assert features[3] > 0.0
    assert max(features[7:]) > 0.0