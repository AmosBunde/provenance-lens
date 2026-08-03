"""Noise residual tests on synthetic composites."""

import numpy as np
from PIL import Image

from provenance_lens.forensics.noise_residuals import _correlation, extract, residual
from provenance_lens.forensics.signals import ALL_REGIONS, Signal


def _noisy(seed=0, size=384, sigma=12.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 120 + 40 * np.sin(xx / 19.0) * np.cos(yy / 23.0)
    return np.clip(base + rng.normal(0, sigma, (size, size)), 0, 255).astype(np.uint8)


def _by_region(signals, name):
    return {s.region: s.value for s in signals if s.name == name}


def test_disjoint_regions_of_homogeneous_noise_decorrelate():
    gray = np.asarray(Image.fromarray(_noisy()).convert("L"), float) / 255.0
    res = residual(gray)
    a = res[0:128, 0:128]
    b = res[256:384, 256:384]
    assert abs(_correlation(a, b)) < 0.05
    assert _correlation(a, a) > 0.99


def test_noise_mismatch_fires_in_noisier_patch():
    image = _noisy(seed=1)
    rng = np.random.default_rng(2)
    patch = image[128:256, 128:256].astype(float)
    image[128:256, 128:256] = np.clip(patch + rng.normal(0, 25, patch.shape), 0, 255)
    signals = extract(Image.fromarray(image))
    mismatch = _by_region(signals, "noise_region_mismatch")
    center = mismatch.pop("r1c1")
    assert center > max(mismatch.values())


def test_cloned_region_correlates():
    image = _noisy(seed=3)
    # copy the top-left third onto the bottom-right third: cloned residual
    image[256:384, 256:384] = image[0:128, 0:128]
    signals = extract(Image.fromarray(image))
    corr = _by_region(signals, "noise_residual_correlation_max")
    clean = _noisy(seed=3)
    clean_corr = _by_region(extract(Image.fromarray(clean)), "noise_residual_correlation_max")
    assert corr["r2c2"] > 0.5
    assert corr["r0c0"] > 0.5
    assert max(clean_corr.values()) < 0.2
    global_pair = {s.name: s.value for s in signals if s.region == "global"}
    assert global_pair["noise_max_pair_correlation"] > 0.5


def test_contract_and_determinism():
    image = Image.fromarray(_noisy(seed=4))
    signals = extract(image)
    assert all(isinstance(s, Signal) for s in signals)
    assert {s.region for s in signals} <= set(ALL_REGIONS)
    assert extract(image) == signals
