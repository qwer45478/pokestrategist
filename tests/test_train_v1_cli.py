from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from pokestrategist.cli.average_checkpoints import main as average_main
from pokestrategist.cli.evaluate_v1 import main as evaluate_main
from pokestrategist.cli.train_v1 import main as train_main

from tests.v1_fixtures import write_dataset


def test_train_and_evaluate_cli_smoke(tmp_path, monkeypatch):
    dataset_path = write_dataset(tmp_path / "samples.jsonl")
    output_dir = tmp_path / "run"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v1",
            "--data",
            str(dataset_path),
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
            "--split-seed",
            "7",
            "--hidden-candidate-topk",
            "3",
            "--recoverable-support-weight",
            "0.35",
            "--local-score-weight",
            "0.7",
            "--local-tail-weight",
            "0.4",
            "--rule-score-weight",
            "0.15",
            "--prior-score-weight",
            "0.12",
            "--switch-subgame-weight",
            "0.3",
            "--switch-follow-discount",
            "0.65",
            "--head-route-weight",
            "0.45",
            "--state-future-weight",
            "0.8",
            "--local-future-weight",
            "0.6",
            "--candidate-response-weight",
            "0.45",
            "--switch-follow-weight",
            "0.25",
            "--response-score-weight",
            "0.05",
            "--recoverable-bc-weight",
            "0.3",
            "--head-weight",
            "0.35",
            "--switch-ranking-weight",
            "0.6",
            "--switch-sample-weight",
            "2.2",
            "--action-score-ce-weight",
            "0.42",
            "--route-consistency-weight",
            "0.0",
            "--consequence-value-weight",
            "0.05",
            "--consequence-value-latent-weight",
            "0.8",
            "--consequence-value-move-scale",
            "0.35",
            "--consequence-value-switch-scale",
            "1.0",
            "--consequence-value-margin-weight",
            "0.2",
            "--consequence-value-margin",
            "0.07",
            "--consequence-value-negative-topk",
            "3",
            "--max-grad-norm",
            "0.9",
            "--warmup-epochs",
            "2",
            "--min-learning-rate-ratio",
            "0.15",
        ],
    )
    train_main()

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "best_ranking_model.pt").exists()
    assert (output_dir / "best_switch_model.pt").exists()
    assert (output_dir / "final_model.pt").exists()
    assert (output_dir / "model_artifacts.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "ranking_metrics.json").exists()
    assert (output_dir / "switch_metrics.json").exists()
    assert (output_dir / "final_metrics.json").exists()
    assert (output_dir / "hidden_move_prior.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "average_checkpoints",
            "--checkpoint-a",
            str(output_dir / "best_ranking_model.pt"),
            "--checkpoint-b",
            str(output_dir / "best_switch_model.pt"),
            "--output",
            str(output_dir / "averaged_model.pt"),
            "--alpha",
            "0.45",
        ],
    )
    average_main()

    assert (output_dir / "averaged_model.pt").exists()
    averaged_checkpoint = torch.load(output_dir / "averaged_model.pt", map_location="cpu")
    assert averaged_checkpoint["averaging"]["alpha"] == pytest.approx(0.45)
    assert averaged_checkpoint["averaging"]["averaged_tensor_count"] > 0

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "covered_top1" in metrics
    assert "covered_top3" in metrics
    assert "switch_target_species_accuracy" in metrics
    assert "consequence_value_margin_mean" in metrics

    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu")
    assert checkpoint["trainer_config"]["train_num_workers"] == 0
    assert checkpoint["trainer_config"]["eval_num_workers"] == 0
    assert checkpoint["trainer_config"]["candidate_response_weight"] == pytest.approx(0.45)
    assert checkpoint["trainer_config"]["switch_follow_weight"] == pytest.approx(0.25)
    assert checkpoint["trainer_config"]["action_score_ce_weight"] == pytest.approx(0.42)
    assert checkpoint["trainer_config"]["route_consistency_weight"] == pytest.approx(0.0)
    assert checkpoint["trainer_config"]["consequence_value_margin_weight"] == pytest.approx(0.2)
    assert checkpoint["trainer_config"]["consequence_value_margin"] == pytest.approx(0.07)
    assert checkpoint["trainer_config"]["consequence_value_negative_topk"] == 3
    assert checkpoint["trainer_config"]["max_grad_norm"] == pytest.approx(0.9)
    assert checkpoint["trainer_config"]["warmup_epochs"] == 2
    assert checkpoint["trainer_config"]["min_learning_rate_ratio"] == pytest.approx(0.15)
    assert checkpoint["model_config"]["response_score_weight"] == pytest.approx(0.05)
    assert checkpoint["model_config"]["prior_score_weight"] == pytest.approx(0.12)
    assert checkpoint["model_config"]["switch_subgame_weight"] == pytest.approx(0.3)
    assert checkpoint["model_config"]["switch_follow_discount"] == pytest.approx(0.65)
    assert checkpoint["model_config"]["consequence_value_weight"] == pytest.approx(0.05)
    assert checkpoint["model_config"]["consequence_value_latent_weight"] == pytest.approx(0.8)
    assert checkpoint["model_config"]["consequence_value_move_scale"] == pytest.approx(0.35)
    assert checkpoint["model_config"]["consequence_value_switch_scale"] == pytest.approx(1.0)

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
        ],
    )
    evaluate_main()


def test_evaluate_cli_loads_legacy_usageprior_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = Path("runs/pokestrategist_v1_metamon_recent10000_honestv4_usageprior_v1/model.pt")
    if not checkpoint_path.exists():
        pytest.skip("legacy usageprior checkpoint not available in workspace")

    dataset_path = write_dataset(tmp_path / "samples.jsonl")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_v1",
            "--data",
            str(dataset_path),
            "--checkpoint",
            str(checkpoint_path),
            "--batch-size",
            "2",
            "--max-samples",
            "4",
            "--device",
            "cpu",
        ],
    )
    evaluate_main()


def test_evaluate_cli_loads_candidate_decoupled_legacy_state_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = Path("runs/pokestrategist_v1_metamon_recent10000_honestv4_usageprior_decoupled_pilot_v1/model.pt")
    if not checkpoint_path.exists():
        pytest.skip("candidate-decoupled legacy-state checkpoint not available in workspace")

    dataset_path = write_dataset(tmp_path / "samples.jsonl")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_v1",
            "--data",
            str(dataset_path),
            "--checkpoint",
            str(checkpoint_path),
            "--batch-size",
            "2",
            "--max-samples",
            "4",
            "--device",
            "cpu",
        ],
    )
    evaluate_main()