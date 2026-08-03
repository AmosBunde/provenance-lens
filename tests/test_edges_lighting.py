"""Edge and lighting tests on synthetic lit scenes and composites."""

import numpy as np
from PIL import Image

from provenance_lens.forensics.edges_lighting import extract, render_debug
from provenance_lens.forensics.signals import ALL_REGIONS, Signal


def _lit_scene(azimuth_deg, size=384, seed=0):
    """Smooth scene lit from a given azimuth (0 = light from the right)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    ramp = np.cos(np.radians(azimuth_deg)) * xx - np.sin(np.radians(azimuth_deg)) * yy
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    texture = 12 * np.sin(xx / 13.0) * np.cos(yy / 11.0)
    noise = rng.normal(0, 2, (size, size))
    return np.clip(40 + 170 * ramp + texture + noise, 0, 255).astype(np.uint8)


def _by_region(signals, name):
    return {s.region: s.value for s in signals if s.name == name}


def _angular_distance(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def test_known_light_direction_is_recovered():
    for azimuth in (0, 90, 225):
        signals = extract(Image.fromarray(_lit_scene(azimuth)))
        directions = _by_region(signals, "lighting_direction_deg")
        errors = [_angular_distance(v, azimuth) for v in directions.values()]
        assert np.median(errors) < 25, (azimuth, sorted(round(e) for e in errors))


def test_opposed_patch_raises_disagreement():
    scene = _lit_scene(0, seed=1)
    opposite = _lit_scene(180, seed=1)
    scene[128:256, 128:256] = opposite[128:256, 128:256]
    signals = extract(Image.fromarray(scene))
    disagreement = _by_region(signals, "lighting_disagreement_deg")
    center = disagreement.pop("r1c1")
    assert center > max(disagreement.values())
    global_signals = {s.name: s.value for s in signals if s.region == "global"}
    clean_global = {
        s.name: s.value
        for s in extract(Image.fromarray(_lit_scene(0, seed=1)))
        if s.region == "global"
    }
    assert global_signals["lighting_inconsistency_deg"] > clean_global["lighting_inconsistency_deg"]


def test_pasted_texture_changes_edge_statistics():
    rng = np.random.default_rng(2)
    smooth = np.clip(
        120 + 30 * np.mgrid[0:384, 0:384][1] / 384 + rng.normal(0, 1.5, (384, 384)),
        0,
        255,
    ).astype(np.uint8)
    signals_clean = extract(Image.fromarray(smooth))
    tampered = smooth.copy()
    tampered[128:256, 128:256] = np.clip(128 + rng.normal(0, 55, (128, 128)), 0, 255).astype(
        np.uint8
    )
    signals_tampered = extract(Image.fromarray(tampered))
    density_clean = _by_region(signals_clean, "edge_density")["r1c1"]
    density_tampered = _by_region(signals_tampered, "edge_density")["r1c1"]
    assert density_tampered > 3 * density_clean
    seam_clean = {s.name: s.value for s in signals_clean if s.region == "global"}
    seam_tampered = {s.name: s.value for s in signals_tampered if s.region == "global"}
    assert seam_tampered["edge_seam_max"] > seam_clean["edge_seam_max"]


def test_debug_overlay_renders(tmp_path):
    out = tmp_path / "overlay.png"
    render_debug(Image.fromarray(_lit_scene(45)), out)
    assert out.exists() and out.stat().st_size > 10_000


def test_contract_and_determinism():
    image = Image.fromarray(_lit_scene(30, seed=3))
    signals = extract(image)
    assert all(isinstance(s, Signal) for s in signals)
    assert {s.region for s in signals} <= set(ALL_REGIONS)
    assert extract(image) == signals
