from __future__ import annotations

import json
import sys
from dataclasses import asdict

import torch

from pokestrategist.cli.evaluate_v1 import main as evaluate_main
from pokestrategist.cli.train_duel_rl import main as duel_main
from pokestrategist.models import PokeStrategistDecisionModel

from tests.v1_fixtures import write_dataset


def test_train_duel_rl_cli_smoke(tmp_path, monkeypatch):
    dataset_path = write_dataset(tmp_path / "samples.jsonl")
    checkpoint_path = tmp_path / "seed_model.pt"
    model = PokeStrategistDecisionModel()
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "trainer_config": {
                "hidden_candidate_topk": 3,
                "use_usage_priors": False,
                "use_team_preview_prior": False,
            },
            "hidden_move_prior": None,
            "team_preview_prior": None,
        },
        checkpoint_path,
    )
    output_dir = tmp_path / "rl_run"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_duel_rl",
            "--data",
            str(dataset_path),
            "--checkpoint",
            str(checkpoint_path),
            "--reference-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--eval-num-workers",
            "0",
            "--disable-team-preview-prior",
            "--disable-usage-priors",
        ],
    )
    duel_main()

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "best_ranking_model.pt").exists()
    assert (output_dir / "best_switch_model.pt").exists()
    assert (output_dir / "final_model.pt").exists()
    assert (output_dir / "model_artifacts.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "final_metrics.json").exists()

    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu")
    assert "rl_config" in checkpoint
    assert checkpoint["reference_checkpoint"] == str(checkpoint_path)
    assert checkpoint["trainer_config"]["use_usage_priors"] is False
    assert checkpoint["metrics"]["gold_policy_score"] >= 0.0

    manifest = json.loads((output_dir / "model_artifacts.json").read_text(encoding="utf-8"))
    assert manifest["best_model"]["path"] == "best_model.pt"
    assert manifest["final_model"]["path"] == "final_model.pt"

    best_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "gold_policy_score" in best_metrics

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_v1",
            "--data",
            str(dataset_path),
            "--checkpoint",
            str(output_dir / "model.pt"),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--disable-team-preview-prior",
            "--disable-usage-priors",
        ],
    )
    evaluate_main()