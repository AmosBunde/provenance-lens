"""Blocking-grid tests: phase tracking under crops, misalignment detection."""

import io

import numpy as np
from PIL import Image

from provenance_lens.forensics.blocking_grid import PERIOD, extract, region_grid_stats
from provenance_lens.forensics.signals import ALL_REGIONS, Signal


def _textured(seed=0, size=512):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 110 + 50 * np.sin(xx / 17.0) * np.cos(yy / 29.0)
    return np.clip(base + rng.normal(0, 10, (size, size)), 0, 255).astype(np.uint8)


def _jpeg(array, quality=60):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer)).copy()


def _phases(gray):
    stats = region_grid_stats(gray.astype(np.float64))
    return {region: (h, v) for region, (_, h, v) in stats.items()}


def test_crop_shifts_detected_phase_consistently():
    compressed = _jpeg(_textured())
    base_phases = _phases(compressed)
    shift = 3
    cropped = compressed[:, shift:]
    cropped_phases = _phases(cropped)
    moved = sum(
        1
        for region in base_phases
        if (base_phases[region][1] - cropped_phases[region][1]) % PERIOD == shift
    )
    assert moved >= 7, f"only {moved}/9 regions tracked the {shift}px crop"


def test_misaligned_paste_raises_misalignment():
    compressed = _jpeg(_textured(seed=1))
    clean_signals = {
        s.name: s.value for s in extract(Image.fromarray(compressed)) if s.region == "global"
    }
    tampered = compressed.copy()
    # paste the center region shifted by half a block: same content, moved lattice
    region = compressed[192:320, 192:320]
    tampered[192:320, 188:316] = region
    tampered_signals = {
        s.name: s.value for s in extract(Image.fromarray(tampered)) if s.region == "global"
    }
    assert (
        tampered_signals["blocking_phase_misalignment"]
        > clean_signals["blocking_phase_misalignment"]
    )


def test_clean_image_has_uniform_phase():
    compressed = _jpeg(_textured(seed=2))
    signals = {
        s.name: s.value for s in extract(Image.fromarray(compressed)) if s.region == "global"
    }
    assert signals["blocking_phase_misalignment"] < 0.15


def test_contract_and_determinism():
    image = Image.fromarray(_jpeg(_textured(seed=4)))
    signals = extract(image)
    assert all(isinstance(s, Signal) for s in signals)
    assert {s.region for s in signals} <= set(ALL_REGIONS)
    assert extract(image) == signals
