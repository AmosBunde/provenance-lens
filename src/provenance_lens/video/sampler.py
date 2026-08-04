"""Frame sampling: uniform plus scene-change boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_FRAMES = 8
SCENE_THRESHOLD = 0.35  # normalized histogram distance that marks a cut


def _read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    return [np.asarray(f) for f in iio.imiter(path)]


def _hist(frame: np.ndarray) -> np.ndarray:
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    h, _ = np.histogram(gray, bins=32, range=(0, 255))
    return h / max(h.sum(), 1)


def scene_changes(frames: list[np.ndarray]) -> list[int]:
    cuts = []
    for i in range(1, len(frames)):
        distance = float(np.abs(_hist(frames[i]) - _hist(frames[i - 1])).sum()) / 2
        if distance > SCENE_THRESHOLD:
            cuts.append(i)
    return cuts


def sample(path: Path, count: int = DEFAULT_FRAMES) -> list[dict]:
    """Uniform indices plus scene-change boundary frames, deduplicated."""
    frames = _read_frames(path)
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    uniform = np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)
    indices = sorted(set(uniform.tolist()) | set(scene_changes(frames)))
    clip_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return [
        {
            "clip_sha256": clip_hash,
            "frame_index": int(i),
            "image": Image.fromarray(frames[i]),
            "is_scene_boundary": i not in uniform,
        }
        for i in indices
    ]
