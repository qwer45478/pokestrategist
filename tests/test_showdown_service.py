from __future__ import annotations

from dataclasses import asdict

import torch

from pokestrategist.models import PokeStrategistDecisionModel
from pokestrategist.serving.showdown import PokeStrategistLocalPredictor, ShowdownBattleSnapshot, snapshot_to_decision_sample
from pokestrategist.training.dataset import DecisionTensorDataset, tensorize_decision_sample

from tests.v1_fixtures import write_dataset


def test_tensorize_decision_sample_includes_action_metadata(tmp_path):
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("replay-a",))
    dataset = DecisionTensorDataset(dataset_path, use_usage_priors=False)
    sample = dataset.raw_sample(0)

    item = tensorize_decision_sample(sample, include_action_metadata=True)

    assert item["legal_action_entries"]
    assert item["legal_action_entries"][0]["head"] in {"move", "switch", "tera-move"}
    assert len(item["legal_action_entries"]) <= 32


def test_showdown_predictor_returns_structured_suggestions(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = PokeStrategistDecisionModel()
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "trainer_config": {"hidden_candidate_topk": 4, "use_usage_priors": False},
            "hidden_move_prior": None,
        },
        checkpoint_path,
    )
    predictor = PokeStrategistLocalPredictor(checkpoint_path, device="cpu", use_usage_priors=False)
    snapshot = ShowdownBattleSnapshot.model_validate(
        {
            "source": "showdown-dom",
            "pageTitle": "[Gen 9] OU battle",
            "turn": "Turn 3",
            "forcedSwitch": False,
            "self": {"name": "Great Tusk", "hp": "45%"},
            "opponent": {"name": "Gholdengo", "hp": "35%"},
            "legalMoves": [
                {"label": "Headlong Rush"},
                {"label": "Knock Off"},
                {"label": "Rapid Spin"},
            ],
            "legalSwitches": [
                {"label": "Dragapult"},
                {"label": "Kingambit"},
            ],
            "recentLog": [
                "Turn 1",
                "Great Tusk used Headlong Rush!",
                "The opposing Gholdengo used Make It Rain!",
                "Turn 2",
                "Great Tusk used Knock Off!",
            ],
        }
    )

    sample = snapshot_to_decision_sample(snapshot)
    result = predictor.predict_sample(sample, topk=3)

    assert result["source"] == "pokestrategist-local"
    assert result["metadata"]["legal_action_count"] >= 3
    assert len(result["suggestions"]) == 3
    assert all(suggestion["label"] for suggestion in result["suggestions"])
    assert all(0.0 <= suggestion["confidence"] <= 1.0 for suggestion in result["suggestions"])