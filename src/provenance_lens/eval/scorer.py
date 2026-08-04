"""Scoring: both tracks, one frozen split, bootstrap confidence intervals.

Predictions arrive as files (the baseline's score file, the reasoner's
verdict JSONL); the scorer joins them to labels inside the harness and
computes accuracy, precision, recall, macro F1, and AUROC with 1000-resample
bootstrap intervals. The baseline-versus-reasoner delta uses the same
resample indices for both tracks, which is what makes the interval on the
difference meaningful. Parse failures count as wrong predictions and are
additionally reported as separate rates by taxonomy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260803


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels.astype(bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def basic_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "macro_f1": (f1_pos + f1_neg) / 2,
    }


@dataclass
class Track:
    """One system's predictions aligned to the split order."""

    name: str
    predictions: np.ndarray  # 0/1, failures already forced wrong
    scores: np.ndarray  # continuous score for AUROC (confidence-signed)
    failure_mask: np.ndarray  # True where the response failed to parse
    failure_taxonomy: dict


def bootstrap_metrics(labels: np.ndarray, track: Track) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(labels), size=(BOOTSTRAP_RESAMPLES, len(labels)))
    point = basic_metrics(labels, track.predictions)
    point["auroc"] = rank_auc(track.scores, labels)
    samples = {k: [] for k in point}
    for row in indices:
        m = basic_metrics(labels[row], track.predictions[row])
        m["auroc"] = rank_auc(track.scores[row], labels[row])
        for key, value in m.items():
            samples[key].append(value)
    out = {}
    for key, value in point.items():
        arr = np.asarray(samples[key], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            # single-class slices make AUROC undefined in every resample;
            # report nan bounds rather than crash or invent an interval
            out[key] = {"value": value, "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        out[key] = {
            "value": value,
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    out["parse_failure_rate"] = float(track.failure_mask.mean())
    out["failure_taxonomy"] = track.failure_taxonomy
    return out


def paired_delta(labels: np.ndarray, a: Track, b: Track, metric: str = "macro_f1") -> dict:
    """CI on metric(a) - metric(b) over shared resample indices."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(labels), size=(BOOTSTRAP_RESAMPLES, len(labels)))
    deltas = []
    for row in indices:
        ma = basic_metrics(labels[row], a.predictions[row])[metric]
        mb = basic_metrics(labels[row], b.predictions[row])[metric]
        deltas.append(ma - mb)
    deltas = np.asarray(deltas)
    point = (
        basic_metrics(labels, a.predictions)[metric] - basic_metrics(labels, b.predictions)[metric]
    )
    return {
        "metric": metric,
        "delta": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "favors": a.name if point > 0 else b.name,
    }


def load_reasoner_track(jsonl_path: Path, order: list[str]) -> Track:
    records = {
        r["sha256"]: r
        for r in (json.loads(line) for line in jsonl_path.read_text().splitlines() if line)
    }
    predictions, scores, failures = [], [], []
    taxonomy: dict[str, int] = {}
    for sha in order:
        record = records.get(sha)
        if record is None or not record["ok"]:
            predictions.append(-1)  # forced wrong later
            scores.append(0.5)
            failures.append(True)
            key = record["failure"] if record else "missing"
            taxonomy[key] = taxonomy.get(key, 0) + 1
        else:
            label = 1 if record["label"] == "manipulated" else 0
            predictions.append(label)
            confidence = float(record["confidence"])
            scores.append(confidence if label == 1 else 1.0 - confidence)
            failures.append(False)
    return Track(
        name="reasoner",
        predictions=np.asarray(predictions),
        scores=np.asarray(scores),
        failure_mask=np.asarray(failures),
        failure_taxonomy=taxonomy,
    )


def force_failures_wrong(track: Track, labels: np.ndarray) -> Track:
    """A failed response scores as an incorrect prediction, by protocol."""
    forced = track.predictions.copy()
    forced[track.failure_mask] = 1 - labels[track.failure_mask]
    return Track(track.name, forced, track.scores, track.failure_mask, track.failure_taxonomy)


def load_baseline_track(parquet_path: Path, order: list[str]) -> Track:
    frame = pd.read_parquet(parquet_path).set_index("sha256").loc[order]
    scores = frame.score.to_numpy(float)
    return Track(
        name="baseline",
        predictions=(scores > 0.5).astype(int),
        scores=scores,
        failure_mask=np.zeros(len(order), dtype=bool),
        failure_taxonomy={},
    )
