"""Blocking-artifact grid estimation per region.

JPEG compression leaves an 8-pixel lattice of small discontinuities. Within
one image the lattice phase is uniform; a pasted patch that was compressed
elsewhere, or shifted during compositing, carries a lattice whose phase
disagrees with the rest of the image. Estimating strength and phase per grid
region exposes exactly that disagreement, which is the regional evidence the
EDA showed a global measure cannot provide.

Per-region signals:

- ``blocking_grid_strength``: comb contrast of the best 8-phase alignment of
  boundary energy, averaged over horizontal and vertical (context: strong
  blocking only says the region has JPEG history).
- ``blocking_grid_phase_h`` / ``blocking_grid_phase_v``: estimated lattice
  phase in pixels, 0 through 7 (context; citable when phases disagree).

Global signals:

- ``blocking_phase_misalignment``: strength-weighted share of regions whose
  phase disagrees with the majority (higher is suspicious).
- ``blocking_strength_spread``: range of per-region strengths (higher is
  suspicious).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from provenance_lens.forensics.signals import Direction, Signal, region_boxes

PERIOD = 8
MIN_STRENGTH_FOR_PHASE_VOTE = 0.02


def _to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _phase_profile(diff_means: np.ndarray) -> tuple[float, int]:
    """Comb contrast and best phase for a 1D boundary-energy profile."""
    if len(diff_means) < 2 * PERIOD:
        return 0.0, 0
    energies = np.array([diff_means[p::PERIOD].mean() for p in range(PERIOD)], dtype=np.float64)
    best = int(np.argmax(energies))
    others = np.delete(energies, best)
    contrast = float((energies[best] - others.mean()) / (energies.mean() + 1e-12))
    return contrast, best


def region_grid_stats(gray: np.ndarray) -> dict[str, tuple[float, int, int]]:
    """Per region: (strength, phase_h, phase_v)."""
    col_diff = np.abs(np.diff(gray, axis=1))
    row_diff = np.abs(np.diff(gray, axis=0))
    stats = {}
    for region, (rows, cols) in region_boxes(*gray.shape).items():
        vertical_profile = col_diff[rows, cols.start : max(cols.stop - 1, cols.start)]
        horizontal_profile = row_diff[rows.start : max(rows.stop - 1, rows.start), cols]
        contrast_v, phase_v = _phase_profile(
            vertical_profile.mean(axis=0) if vertical_profile.size else np.zeros(0)
        )
        contrast_h, phase_h = _phase_profile(
            horizontal_profile.mean(axis=1) if horizontal_profile.size else np.zeros(0)
        )
        # phases are measured inside the region; convert to absolute image
        # phase so regions are comparable
        abs_phase_v = (phase_v + cols.start) % PERIOD
        abs_phase_h = (phase_h + rows.start) % PERIOD
        stats[region] = ((contrast_v + contrast_h) / 2.0, abs_phase_h, abs_phase_v)
    return stats


def extract(image: Image.Image) -> list[Signal]:
    stats = region_grid_stats(_to_gray(image))
    signals: list[Signal] = []
    strengths = {}
    votes: dict[tuple[int, int], float] = {}
    for region, (strength, phase_h, phase_v) in stats.items():
        strengths[region] = strength
        signals.append(Signal("blocking_grid_strength", region, strength, Direction.CONTEXT))
        signals.append(Signal("blocking_grid_phase_h", region, float(phase_h), Direction.CONTEXT))
        signals.append(Signal("blocking_grid_phase_v", region, float(phase_v), Direction.CONTEXT))
        if strength >= MIN_STRENGTH_FOR_PHASE_VOTE:
            votes[(phase_h, phase_v)] = votes.get((phase_h, phase_v), 0.0) + strength
    if votes:
        majority = max(sorted(votes), key=lambda k: votes[k])
        total = sum(votes.values())
        disagreement = sum(w for k, w in votes.items() if k != majority) / total
    else:
        disagreement = 0.0
    values = list(strengths.values())
    signals.append(
        Signal(
            "blocking_phase_misalignment",
            "global",
            float(disagreement),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    signals.append(
        Signal(
            "blocking_strength_spread",
            "global",
            float(max(values) - min(values)),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    return signals
