from __future__ import annotations

import pytest
import torch

from pokestrategist.training.trainer import _expected_calibration_error


def test_expected_calibration_error_counts_confidence_one_rows() -> None:
    probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)

    ece = _expected_calibration_error(probs, targets, bins=10)

    assert ece == pytest.approx(1.0)