"""Calibration: reliability bins, ECE, and validation-only temperature scaling.

The temperature is fit by minimizing negative log likelihood on validation
predictions; the fit function refuses anything tagged as test, structurally,
so the protocol rule is code rather than convention.
"""

from __future__ import annotations

import numpy as np

ECE_BINS = 15


def ece(confidences: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    value = 0.0
    diagram = []
    for i in range(bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            mask |= confidences == 0.0
        count = int(mask.sum())
        if count:
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correct[mask].mean())
            value += (count / total) * abs(avg_conf - avg_acc)
        else:
            avg_conf, avg_acc = float("nan"), float("nan")
        diagram.append(
            {
                "bin_low": float(edges[i]),
                "bin_high": float(edges[i + 1]),
                "count": count,
                "confidence": avg_conf,
                "accuracy": avg_acc,
            }
        )
    return {"ece": float(value), "diagram": diagram}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _nll(scores: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = 1 / (1 + np.exp(-_logit(scores) / temperature))
    scaled = np.clip(scaled, 1e-9, 1 - 1e-9)
    return float(-(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)).mean())


def fit_temperature(scores: np.ndarray, labels: np.ndarray, split_tag: str) -> float:
    """Grid-plus-refine NLL minimization; refuses non-validation data."""
    if split_tag != "val":
        raise ValueError(f"temperature scaling fits on validation only; got split {split_tag!r}")
    grid = np.geomspace(0.1, 10.0, 61)
    best = min(grid, key=lambda t: _nll(scores, labels, t))
    fine = np.linspace(best * 0.7, best * 1.4, 61)
    return float(min(fine, key=lambda t: _nll(scores, labels, t)))


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(scores) / temperature))
