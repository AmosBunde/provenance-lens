"""Ghost extractor tests on synthetic recompression fixtures."""

import io
import time

import numpy as np
import pytest
from PIL import Image

from provenance_lens.forensics.compression_ghosts import extract
from provenance_lens.forensics.signals import ALL_REGIONS, Direction, Signal


def _textured(seed=0, size=384):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 96 + 64 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    noise = rng.normal(0, 12, (size, size))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(array, quality):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer)).copy()


@pytest.fixture()
def tampered_image():
    """Center region recompressed at quality 60 inside a quality 90 image."""
    clean = _jpeg_roundtrip(_textured(), 90)
    patch = _jpeg_roundtrip(clean[128:256, 128:256], 60)
    composite = clean.copy()
    composite[128:256, 128:256] = patch
    return Image.fromarray(_jpeg_roundtrip(composite, 90))


def _by_region(signals, name):
    return {s.region: s for s in signals if s.name == name}


def test_ghost_fires_in_tampered_region_only(tampered_image):
    signals = extract(tampered_image)
    depth = _by_region(signals, "jpeg_ghost_depth")
    tampered = depth["r1c1"].value
    clean = [depth[r].value for r in depth if r not in ("r1c1", "global")]
    assert tampered > max(clean), (tampered, max(clean))
    minima = _by_region(signals, "jpeg_ghost_min_quality")
    assert abs(minima["r1c1"].value - 60) <= 10


def test_clean_image_has_low_spread():
    clean = Image.fromarray(_jpeg_roundtrip(_textured(seed=3), 90))
    signals = extract(clean)
    spread = {s.name: s.value for s in signals if s.region == "global"}
    tampered_signals = extract_tampered_reference()
    tampered_spread = {s.name: s.value for s in tampered_signals if s.region == "global"}
    assert spread["jpeg_ghost_depth_range"] < tampered_spread["jpeg_ghost_depth_range"]


def extract_tampered_reference():
    clean = _jpeg_roundtrip(_textured(seed=3), 90)
    patch = _jpeg_roundtrip(clean[128:256, 128:256], 60)
    clean[128:256, 128:256] = patch
    return extract(Image.fromarray(_jpeg_roundtrip(clean, 90)))


def test_signals_follow_the_contract(tampered_image):
    signals = extract(tampered_image)
    assert all(isinstance(s, Signal) for s in signals)
    assert {s.region for s in signals} <= set(ALL_REGIONS)
    assert {s.name for s in signals} == {
        "jpeg_ghost_min_quality",
        "jpeg_ghost_depth",
        "jpeg_ghost_quality_spread",
        "jpeg_ghost_depth_range",
    }
    directions = {s.name: s.direction for s in signals}
    assert directions["jpeg_ghost_depth"] is Direction.HIGHER_SUSPICIOUS
    assert directions["jpeg_ghost_min_quality"] is Direction.CONTEXT


def test_deterministic(tampered_image):
    assert extract(tampered_image) == extract(tampered_image)


def test_runtime_is_bounded_on_large_input():
    large = Image.fromarray(_textured(seed=5, size=2048))
    start = time.perf_counter()
    extract(large)
    elapsed = time.perf_counter() - start
    assert elapsed < 60, f"ghost extraction took {elapsed:.1f}s"
