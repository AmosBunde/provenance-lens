"""Edge coherence and lighting direction estimators.

Two families close out the extractor set. Edge statistics expose splice
seams: a pasted patch changes edge density and gradient-orientation
coherence, and the paste boundary itself is a gradient ridge that does not
belong to the scene. Lighting direction exposes physically inconsistent
composites: the shading gradient of a smoothly lit surface points toward
the light, and a patch lit from elsewhere disagrees with the scene
consensus.

Per-region signals:

- ``edge_density`` (context): mean gradient magnitude.
- ``edge_orientation_coherence`` (context): structure-tensor coherence, 0
  for isotropic texture, 1 for a single dominant orientation.
- ``edge_seam_discontinuity`` (higher suspicious): gradient energy in a
  thin band along the region border relative to the region interior.
- ``lighting_direction_deg`` (context): estimated light azimuth, degrees.
- ``lighting_disagreement_deg`` (higher suspicious): angular distance from
  the confidence-weighted scene consensus direction.

Global signals:

- ``lighting_inconsistency_deg`` (higher suspicious): confidence-weighted
  mean angular deviation across regions.
- ``edge_seam_max`` (higher suspicious): maximum regional seam score.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

from provenance_lens.forensics.signals import Direction, Signal, region_boxes

MAX_SIDE = 1024
SHADING_SIGMA = 12.0
SEAM_BAND = 4


def _to_gray(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    if max(gray.size) > MAX_SIDE:
        gray.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64) / 255.0


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    return gx, gy


def _coherence(gx: np.ndarray, gy: np.ndarray) -> float:
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    trace = jxx + jyy
    if trace < 1e-16:
        return 0.0
    return float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / trace)


def _seam_score(magnitude: np.ndarray, rows: slice, cols: slice) -> float:
    region = magnitude[rows, cols]
    if min(region.shape) <= 2 * SEAM_BAND:
        return 0.0
    border = np.concatenate(
        [
            region[:SEAM_BAND, :].ravel(),
            region[-SEAM_BAND:, :].ravel(),
            region[:, :SEAM_BAND].ravel(),
            region[:, -SEAM_BAND:].ravel(),
        ]
    )
    interior = region[SEAM_BAND:-SEAM_BAND, SEAM_BAND:-SEAM_BAND]
    return float(border.mean() / (interior.mean() + 1e-12))


def _light_direction(gray_region: np.ndarray) -> tuple[float, float]:
    """(azimuth degrees, confidence) from the smoothed shading gradient."""
    shading = gaussian_filter(gray_region, SHADING_SIGMA)
    gx = sobel(shading, axis=1)
    gy = sobel(shading, axis=0)
    mean_x = float(gx.mean())
    mean_y = float(gy.mean())
    magnitude = float(np.hypot(mean_x, mean_y))
    typical = float(np.hypot(gx, gy).mean())
    confidence = magnitude / (typical + 1e-12)
    # brightness increases toward the light; screen y grows downward
    azimuth = float(np.degrees(np.arctan2(-mean_y, mean_x)) % 360.0)
    return azimuth, confidence


def _angular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def extract(image: Image.Image) -> list[Signal]:
    gray = _to_gray(image)
    gx, gy = _gradients(gray)
    magnitude = np.hypot(gx, gy)
    boxes = region_boxes(*gray.shape)

    signals: list[Signal] = []
    directions: dict[str, tuple[float, float]] = {}
    seam_scores: dict[str, float] = {}
    for region, (rows, cols) in boxes.items():
        density = float(magnitude[rows, cols].mean())
        coherence = _coherence(gx[rows, cols], gy[rows, cols])
        seam = _seam_score(magnitude, rows, cols)
        azimuth, confidence = _light_direction(gray[rows, cols])
        directions[region] = (azimuth, confidence)
        seam_scores[region] = seam
        signals.append(Signal("edge_density", region, density, Direction.CONTEXT))
        signals.append(Signal("edge_orientation_coherence", region, coherence, Direction.CONTEXT))
        signals.append(Signal("edge_seam_discontinuity", region, seam, Direction.HIGHER_SUSPICIOUS))
        signals.append(Signal("lighting_direction_deg", region, azimuth, Direction.CONTEXT))

    total_confidence = sum(c for _, c in directions.values()) + 1e-12
    consensus_x = sum(c * np.cos(np.radians(a)) for a, c in directions.values())
    consensus_y = sum(c * np.sin(np.radians(a)) for a, c in directions.values())
    consensus = float(np.degrees(np.arctan2(consensus_y, consensus_x)) % 360.0)

    weighted_dev = 0.0
    for region, (azimuth, confidence) in directions.items():
        deviation = _angular_distance(azimuth, consensus)
        weighted_dev += confidence * deviation
        signals.append(
            Signal(
                "lighting_disagreement_deg",
                region,
                deviation,
                Direction.HIGHER_SUSPICIOUS,
            )
        )
    signals.append(
        Signal(
            "lighting_inconsistency_deg",
            "global",
            float(weighted_dev / total_confidence),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    signals.append(
        Signal(
            "edge_seam_max",
            "global",
            float(max(seam_scores.values())),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    return signals


def render_debug(image: Image.Image, out_path) -> None:
    """Write an overlay with the region grid and lighting arrows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = _to_gray(image)
    boxes = region_boxes(*gray.shape)
    fig, ax = plt.subplots(figsize=(7, 7 * gray.shape[0] / gray.shape[1]))
    ax.imshow(gray, cmap="gray")
    for region, (rows, cols) in boxes.items():
        azimuth, confidence = _light_direction(gray[rows, cols])
        cy = (rows.start + rows.stop) / 2
        cx = (cols.start + cols.stop) / 2
        length = 0.35 * min(rows.stop - rows.start, cols.stop - cols.start)
        dx = length * np.cos(np.radians(azimuth))
        dy = -length * np.sin(np.radians(azimuth))
        ax.arrow(cx, cy, dx, dy, color="red", width=1.2, head_width=8)
        ax.add_patch(
            plt.Rectangle(
                (cols.start, rows.start),
                cols.stop - cols.start,
                rows.stop - rows.start,
                fill=False,
                edgecolor="cyan",
                linewidth=0.6,
            )
        )
        ax.text(cols.start + 4, rows.start + 14, region, color="cyan", fontsize=7)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
