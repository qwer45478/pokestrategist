from __future__ import annotations

from torch.utils.data import DataLoader

from pokestrategist.data.team_preview_prior import TEAM_PREVIEW_FEATURE_DIM, build_team_preview_prior, iter_team_preview_examples
from pokestrategist.training.dataset import DecisionTensorDataset, collate_decision_batch

from tests.v1_fixtures import write_dataset


def test_hidden_candidate_usage_scores_are_exposed(tmp_path) -> None:
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("replay-a",))
    dataset = DecisionTensorDataset(dataset_path)
    dataset.set_team_preview_prior(build_team_preview_prior(iter_team_preview_examples(dataset.iter_raw_samples(range(len(dataset))))))

    row = dataset[0]

    assert "legal_action_usage_scores" in row
    assert "opp_preview_prior_features" in row
    assert "particle_posterior_target" in row
    assert "reveal_likelihood_target" in row
    assert "typed_consequence_bin_target" in row
    assert len(row["legal_action_usage_scores"]) == 32
    assert len(row["opp_preview_prior_features"]) == TEAM_PREVIEW_FEATURE_DIM
    assert len(row["particle_posterior_target"]) == 10
    assert len(row["reveal_likelihood_target"]) == 5
    assert len(row["typed_consequence_bin_target"]) == 10
    assert max(row["legal_action_usage_scores"]) >= 0.0
    assert max(row["opp_preview_prior_features"]) > 0.0


def test_tensor_dataset_supports_multi_worker_loading(tmp_path) -> None:
    dataset_path = write_dataset(tmp_path / "samples.jsonl", replay_ids=("replay-a", "replay-b"))
    dataset = DecisionTensorDataset(dataset_path, use_usage_priors=False)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_decision_batch, num_workers=2)
    batches = list(loader)

    assert batches
    assert sum(batch["state_numeric"].shape[0] for batch in batches) == len(dataset)