"""Scorer tests: metric math against hand-computed cases, shared resamples."""

import json

import numpy as np
import pandas as pd

from provenance_lens.eval.scorer import (
    Track,
    basic_metrics,
    bootstrap_metrics,
    force_failures_wrong,
    load_baseline_track,
    load_reasoner_track,
    paired_delta,
    rank_auc,
)


def test_basic_metrics_hand_case():
    labels = np.array([1, 1, 1, 0, 0])
    predictions = np.array([1, 1, 0, 0, 1])
    m = basic_metrics(labels, predictions)
    assert m["accuracy"] == 0.6
    assert abs(m["precision"] - 2 / 3) < 1e-12
    assert abs(m["recall"] - 2 / 3) < 1e-12
    # f1_pos = 2/3, f1_neg = 1/2 -> macro 7/12
    assert abs(m["macro_f1"] - 7 / 12) < 1e-12


def test_rank_auc_perfect_and_random():
    labels = np.array([0, 0, 1, 1])
    assert rank_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert rank_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0


def _track(predictions, labels):
    return Track(
        name="t",
        predictions=predictions,
        scores=predictions.astype(float),
        failure_mask=np.zeros(len(labels), dtype=bool),
        failure_taxonomy={},
    )


def test_bootstrap_intervals_bracket_the_point():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    predictions = labels.copy()
    flip = rng.random(400) < 0.2
    predictions[flip] = 1 - predictions[flip]
    out = bootstrap_metrics(labels, _track(predictions, labels))
    accuracy = out["accuracy"]
    assert accuracy["ci_low"] <= accuracy["value"] <= accuracy["ci_high"]
    assert 0.02 < accuracy["ci_high"] - accuracy["ci_low"] < 0.12


def test_paired_delta_detects_a_better_system():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 600)
    good = labels.copy()
    flip = rng.random(600) < 0.1
    good[flip] = 1 - good[flip]
    bad = labels.copy()
    flip = rng.random(600) < 0.3
    bad[flip] = 1 - bad[flip]
    delta = paired_delta(labels, _track(good, labels), _track(bad, labels))
    assert delta["favors"] == "t"
    assert delta["ci_low"] > 0


def test_reasoner_track_forces_failures_wrong(tmp_path):
    order = ["a" * 64, "b" * 64, "c" * 64]
    records = [
        {"sha256": order[0], "ok": True, "label": "manipulated", "confidence": 0.9},
        {"sha256": order[1], "ok": False, "label": None, "confidence": None, "failure": "not_json"},
    ]
    path = tmp_path / "v.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    track = load_reasoner_track(path, order)
    labels = np.array([1, 1, 0])
    forced = force_failures_wrong(track, labels)
    assert forced.predictions[0] == 1  # honest answer kept
    assert forced.predictions[1] == 0  # failure forced wrong against label 1
    assert forced.predictions[2] == 1  # missing forced wrong against label 0
    assert track.failure_taxonomy == {"not_json": 1, "missing": 1}


def test_baseline_track_loads_scores(tmp_path):
    order = ["a" * 64, "b" * 64]
    pd.DataFrame({"sha256": order, "score": [0.9, 0.2]}).to_parquet(
        tmp_path / "p.parquet", index=False
    )
    track = load_baseline_track(tmp_path / "p.parquet", order)
    assert list(track.predictions) == [1, 0]
    assert track.failure_mask.sum() == 0
