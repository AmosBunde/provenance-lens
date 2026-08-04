"""Calibration tests: ECE hand cases and the validation-only guard."""

import numpy as np
import pytest

from provenance_lens.eval.calibration import (
    apply_temperature,
    ece,
    fit_temperature,
)


def test_ece_hand_case():
    # two bins in use: perfect low bin, overconfident high bin
    confidences = np.array([0.1, 0.1, 0.9, 0.9])
    correct = np.array([0, 0, 1, 0])
    out = ece(confidences, correct)
    # bin (0.066..0.133]: conf 0.1, acc 0 -> |0.1| * 0.5 weight
    # bin (0.866..0.933]: conf 0.9, acc 0.5 -> 0.4 * 0.5 weight
    assert abs(out["ece"] - (0.5 * 0.1 + 0.5 * 0.4)) < 1e-9
    assert sum(b["count"] for b in out["diagram"]) == 4


def test_temperature_fit_improves_overconfident_scores():
    # genuinely overconfident: 97 percent stated confidence, 80 percent
    # actual accuracy; a perfectly separated fixture would instead be
    # underconfident and correctly fit a sharpening temperature
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 4000)
    correct = rng.random(4000) < 0.8
    predicted = np.where(correct, labels, 1 - labels)
    overconfident = np.where(predicted == 1, 0.97, 0.03).astype(float)
    temperature = fit_temperature(overconfident, labels, split_tag="val")
    assert temperature > 1.0  # cooling required
    before = ece(overconfident, (overconfident > 0.5) == labels)["ece"]
    scaled = apply_temperature(overconfident, temperature)
    after = ece(scaled, (scaled > 0.5) == labels)["ece"]
    assert after < before


def test_fit_refuses_test_split():
    with pytest.raises(ValueError) as excinfo:
        fit_temperature(np.array([0.5]), np.array([1]), split_tag="test")
    assert "validation only" in str(excinfo.value)


def test_apply_temperature_identity_at_one():
    scores = np.array([0.2, 0.5, 0.8])
    assert np.allclose(apply_temperature(scores, 1.0), scores, atol=1e-6)
