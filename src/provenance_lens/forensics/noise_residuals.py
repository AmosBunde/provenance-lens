"""Noise residual analysis: per-region statistics and cross-region structure.

The residual (image minus a median-denoised copy) removes scene content and
keeps sensor noise plus processing artifacts. Three kinds of evidence live
there:

- A pasted patch from another image or a generator carries noise statistics
  that disagree with the rest of the image (variance mismatch).
- Copy-moved content duplicates its residual, so two regions correlating
  strongly is the signature of cloned pixels (PRNU-style correlation used
  within one image).
- AI-generated images show globally depressed high-frequency residual
  energy, the strongest within-source signal in the EDA.

Per-region signals:

- ``noise_residual_std`` (context), ``noise_residual_kurtosis`` (context)
- ``noise_region_mismatch``: deviation of the region's residual spread from
  the median of the other regions (higher is suspicious)
- ``noise_residual_correlation_max``: strongest correlation of this region's
  residual with any other region (higher is suspicious; cloned content)

Global signals:

- ``noise_variance_mismatch``: maximum regional mismatch (higher suspicious)
- ``noise_max_pair_correlation``: strongest cross-region residual
  correlation (higher suspicious)
"""

from __future__ import annotations

import itertools

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from provenance_lens.forensics.signals import Direction, Signal, region_boxes

MEDIAN_SIZE = 3
MAX_SIDE = 1024


def _to_gray(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    if max(gray.size) > MAX_SIDE:
        gray.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64) / 255.0


def residual(gray: np.ndarray) -> np.ndarray:
    return gray - median_filter(gray, size=MEDIAN_SIZE)


def _kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    variance = float((centered**2).mean())
    if variance < 1e-16:
        return 0.0
    return float((centered**4).mean() / variance**2 - 3.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    fa = a[:height, :width].ravel()
    fb = b[:height, :width].ravel()
    fa = fa - fa.mean()
    fb = fb - fb.mean()
    denom = float(np.sqrt((fa**2).sum() * (fb**2).sum()))
    if denom < 1e-16:
        return 0.0
    return float((fa * fb).sum() / denom)


def extract(image: Image.Image) -> list[Signal]:
    gray = _to_gray(image)
    res = residual(gray)
    boxes = region_boxes(*gray.shape)
    region_res = {region: res[rows, cols] for region, (rows, cols) in boxes.items()}

    stds = {region: float(r.std()) for region, r in region_res.items()}
    signals: list[Signal] = []
    mismatches = {}
    for region, r in region_res.items():
        others = [stds[other] for other in stds if other != region]
        median_other = float(np.median(others))
        mismatch = abs(stds[region] - median_other) / (median_other + 1e-12)
        mismatches[region] = mismatch
        signals.append(Signal("noise_residual_std", region, stds[region], Direction.CONTEXT))
        signals.append(Signal("noise_residual_kurtosis", region, _kurtosis(r), Direction.CONTEXT))
        signals.append(
            Signal("noise_region_mismatch", region, mismatch, Direction.HIGHER_SUSPICIOUS)
        )

    pair_corr = {region: 0.0 for region in region_res}
    max_pair = 0.0
    for a, b in itertools.combinations(sorted(region_res), 2):
        corr = abs(_correlation(region_res[a], region_res[b]))
        max_pair = max(max_pair, corr)
        pair_corr[a] = max(pair_corr[a], corr)
        pair_corr[b] = max(pair_corr[b], corr)
    for region, corr in pair_corr.items():
        signals.append(
            Signal(
                "noise_residual_correlation_max",
                region,
                corr,
                Direction.HIGHER_SUSPICIOUS,
            )
        )
    signals.append(
        Signal(
            "noise_variance_mismatch",
            "global",
            float(max(mismatches.values())),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    signals.append(
        Signal(
            "noise_max_pair_correlation",
            "global",
            max_pair,
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    return signals
