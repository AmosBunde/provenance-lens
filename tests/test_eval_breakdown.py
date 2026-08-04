"""Breakdown tests: slicing math, confound note, confusion plots."""

import numpy as np
import pandas as pd

from provenance_lens.eval.breakdown import (
    CONFOUND_NOTE,
    confusion_matrix,
    per_type_metrics,
    write_breakdown,
)
from provenance_lens.eval.scorer import Track


def _track(predictions):
    return Track(
        name="baseline",
        predictions=predictions,
        scores=predictions.astype(float),
        failure_mask=np.zeros(len(predictions), dtype=bool),
        failure_taxonomy={},
    )


def test_confusion_matrix_hand_case():
    labels = np.array([0, 0, 1, 1, 1])
    predictions = np.array([0, 1, 1, 1, 0])
    m = confusion_matrix(labels, predictions)
    assert m == {
        "true_authentic_pred_authentic": 1,
        "true_authentic_pred_manipulated": 1,
        "true_manipulated_pred_authentic": 1,
        "true_manipulated_pred_manipulated": 2,
    }


def test_per_type_slices_carry_note_and_sizes():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    types = pd.Series(np.where(labels == 1, "ai_generated", "none"))
    types.iloc[:20] = "copy_move"
    out = per_type_metrics(labels, types, _track(labels.copy()))
    assert out["note"] == CONFOUND_NOTE
    assert out["types"]["copy_move"]["n"] == 20
    total = sum(v["n"] for v in out["types"].values())
    assert total == 300


def test_small_stratum_interval_is_wide():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 400)
    predictions = labels.copy()
    flip = rng.random(400) < 0.15
    predictions[flip] = 1 - predictions[flip]
    types = pd.Series(["none"] * 380 + ["copy_move"] * 20)
    out = per_type_metrics(labels, types, _track(predictions))
    wide = out["types"]["copy_move"]["accuracy"]
    narrow = out["types"]["none"]["accuracy"]
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_write_breakdown_emits_artifacts(tmp_path):
    labels = np.array([0, 1] * 30)
    types = pd.Series(["none", "ai_generated"] * 30)
    write_breakdown(labels, types, _track(labels.copy()), tmp_path)
    assert (tmp_path / "breakdown_baseline.json").exists()
    assert (tmp_path / "confusion_baseline.png").stat().st_size > 5000
