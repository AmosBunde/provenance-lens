"""Sampler tests on synthetic clips written with imageio."""

import numpy as np
import pytest

from provenance_lens.video.sampler import DEFAULT_FRAMES, sample, scene_changes


def _write_clip(path, segments, size=64, fps=8):
    """Write a clip from (frame_count, seed) segments; each segment is a
    distinct noise texture, so segment boundaries are hard cuts."""
    import imageio.v3 as iio

    frames = []
    for count, seed in segments:
        rng = np.random.default_rng(seed)
        # segments differ in brightness distribution, not just texture:
        # uniform noise has the same histogram at any seed, which is
        # statistically cutless for a histogram detector
        mean = 80 + (seed % 7) * 20
        base = np.clip(rng.normal(mean, 25, (size, size, 3)), 0, 255).astype(np.uint8)
        for i in range(count):
            drift = np.roll(base, i, axis=1)
            frames.append(drift)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
    return len(frames)


@pytest.fixture()
def single_scene_clip(tmp_path):
    path = tmp_path / "single.mp4"
    total = _write_clip(path, [(24, 1)])
    return path, total


@pytest.fixture()
def hard_cut_clip(tmp_path):
    path = tmp_path / "cut.mp4"
    total = _write_clip(path, [(12, 1), (12, 4)])
    return path, total


def test_uniform_sampling_returns_configured_count(single_scene_clip):
    path, total = single_scene_clip
    frames = sample(path)
    uniform = [f for f in frames if not f["is_scene_boundary"]]
    assert len(uniform) == DEFAULT_FRAMES
    indices = [f["frame_index"] for f in uniform]
    assert indices == sorted(indices)
    assert indices[0] == 0 and indices[-1] == total - 1


def test_hard_cut_yields_boundary_frame(hard_cut_clip):
    path, total = hard_cut_clip
    import imageio.v3 as iio

    frames = [np.asarray(f) for f in iio.imiter(path)]
    cuts = scene_changes(frames)
    assert any(abs(c - 12) <= 1 for c in cuts), cuts
    sampled = sample(path)
    assert any(abs(f["frame_index"] - 12) <= 1 for f in sampled), [
        f["frame_index"] for f in sampled
    ]


def test_single_scene_has_no_cuts(single_scene_clip):
    path, _ = single_scene_clip
    import imageio.v3 as iio

    frames = [np.asarray(f) for f in iio.imiter(path)]
    assert scene_changes(frames) == []


def test_sampling_is_deterministic(single_scene_clip):
    path, _ = single_scene_clip
    a = [(f["frame_index"], f["clip_sha256"]) for f in sample(path)]
    b = [(f["frame_index"], f["clip_sha256"]) for f in sample(path)]
    assert a == b
