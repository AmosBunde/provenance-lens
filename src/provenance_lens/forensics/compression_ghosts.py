"""JPEG ghost detection: regional requantization-error curves.

Recompressing an image across a ladder of JPEG qualities and measuring the
squared error against the original produces, for a region that was earlier
compressed at some quality q, a local minimum near q (the ghost). A region
whose ghost differs from the rest of the image was compressed on a different
path, which is the signature of a pasted or locally recompressed patch.

Emitted signals per grid region:

- ``jpeg_ghost_min_quality``: ladder quality at the strongest interior local
  minimum of the error curve (context). The global minimum always sits at
  the final save quality and is uninformative; the ghost is the secondary
  dip left by an earlier, regional compression.
- ``jpeg_ghost_depth``: normalized prominence of that interior minimum;
  higher values mean a sharper ghost.

Global signals:

- ``jpeg_ghost_quality_spread``: spread of per-region minima (higher is
  suspicious; a clean image agrees with itself).
- ``jpeg_ghost_depth_range``: range of per-region depths (higher is
  suspicious).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from provenance_lens.forensics.signals import Direction, Signal, region_boxes

QUALITY_LADDER = tuple(range(45, 100, 5))
MAX_SIDE = 1024  # analysis resolution cap; ghosts survive moderate downscale


def _to_analysis_gray(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    if max(gray.size) > MAX_SIDE:
        gray.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64)


def _recompress(gray: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(gray.astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer), dtype=np.float64)


def ghost_curves(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Per-region squared-error curve across the quality ladder."""
    boxes = region_boxes(*gray.shape)
    curves = {region: np.zeros(len(QUALITY_LADDER)) for region in boxes}
    for i, quality in enumerate(QUALITY_LADDER):
        error = (gray - _recompress(gray, quality)) ** 2
        for region, (rows, cols) in boxes.items():
            curves[region][i] = float(error[rows, cols].mean())
    return curves


def _strongest_interior_minimum(curve: np.ndarray) -> tuple[float, float]:
    """Quality and normalized prominence of the strongest interior local
    minimum of the error curve.

    The global minimum always sits at the quality of the final save and is
    therefore uninformative; the ghost of an earlier, regional compression
    is a secondary dip between higher neighbors. A curve with no interior
    dip reports the global-minimum quality with depth 0.
    """
    mean = float(curve.mean())
    best_quality = float(QUALITY_LADDER[int(np.argmin(curve))])
    best_depth = 0.0
    for i in range(1, len(curve) - 1):
        prominence = float(min(curve[i - 1], curve[i + 1]) - curve[i])
        if prominence <= 0:
            continue
        depth = prominence / (mean + 1e-12)
        if depth > best_depth:
            best_depth = depth
            best_quality = float(QUALITY_LADDER[i])
    return best_quality, best_depth


def extract(image: Image.Image) -> list[Signal]:
    gray = _to_analysis_gray(image)
    curves = ghost_curves(gray)
    signals: list[Signal] = []
    minima, depths = [], []
    for region, curve in curves.items():
        min_quality, depth = _strongest_interior_minimum(curve)
        minima.append(min_quality)
        depths.append(depth)
        signals.append(Signal("jpeg_ghost_min_quality", region, min_quality, Direction.CONTEXT))
        signals.append(Signal("jpeg_ghost_depth", region, depth, Direction.HIGHER_SUSPICIOUS))
    signals.append(
        Signal(
            "jpeg_ghost_quality_spread",
            "global",
            float(np.std(minima)),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    signals.append(
        Signal(
            "jpeg_ghost_depth_range",
            "global",
            float(np.max(depths) - np.min(depths)),
            Direction.HIGHER_SUSPICIOUS,
        )
    )
    return signals
