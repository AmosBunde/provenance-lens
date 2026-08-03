"""Manifest tests: labeling, hashing stability, dedupe, determinism."""

import json

import pandas as pd
import pytest
from PIL import Image

from provenance_lens.data.manifest import (
    build_manifest,
    classify,
    collapse,
    hash_and_measure,
    scan_assets,
)


def _write_image(path, seed, size=(64, 64)):
    """Seeded noise image: distinct seeds give distant perceptual hashes,
    unlike flat colors, which all share one pHash."""
    import random

    rng = random.Random(seed)
    image = Image.new("L", size)
    image.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


@pytest.fixture()
def raw_root(tmp_path):
    root = tmp_path / "raw"
    micc = root / "micc_f220"
    _write_image(micc / "MICC-F220" / "CRW_4809_scale.jpg", seed=1)
    _write_image(micc / "MICC-F220" / "CRW_4809tamp1.jpg", seed=2)
    (micc / "source_record.json").write_text(
        json.dumps({"source": "micc_f220", "license": "MICC research-only"})
    )
    cifake = root / "cifake"
    _write_image(cifake / "real" / "img_000001.png", seed=3, size=(32, 32))
    _write_image(cifake / "fake" / "img_000002.png", seed=4, size=(32, 32))
    (cifake / "source_record.json").write_text(json.dumps({"source": "cifake", "license": "MIT"}))
    return root


def test_labels_follow_source_conventions(tmp_path):
    from pathlib import Path

    assert classify("micc_f220", Path("CRW_4809tamp1.jpg")) == (
        "manipulated",
        "copy_move",
        "micc_f220:CRW_4809",
    )
    assert classify("micc_f220", Path("CRW_4809_scale.jpg")) == (
        "authentic",
        "none",
        "micc_f220:CRW_4809",
    )
    label, manipulation, _ = classify("cifake", Path("fake/x.png"))
    assert (label, manipulation) == ("manipulated", "ai_generated")


def test_hashing_is_stable(tmp_path):
    image = tmp_path / "a.png"
    _write_image(image, seed=5)
    assert hash_and_measure(image) == hash_and_measure(image)


def test_exact_duplicates_collapse(raw_root):
    duplicate = raw_root / "micc_f220" / "MICC-F220" / "CRW_9999_scale.jpg"
    original = raw_root / "micc_f220" / "MICC-F220" / "CRW_4809_scale.jpg"
    duplicate.write_bytes(original.read_bytes())
    kept, report = collapse(scan_assets(raw_root))
    assert report["exact_removed"] == 1
    assert report["output"] == 4


def test_near_duplicates_collapse_within_label_only(raw_root):
    base = raw_root / "cifake" / "real" / "img_000001.png"
    with Image.open(base) as image:
        near_real = image.copy()
        near_real.putpixel((0, 0), (11, 11, 201))
        near_real.save(raw_root / "cifake" / "real" / "img_000003.png")
        near_fake = image.copy()
        near_fake.putpixel((1, 1), (99, 11, 201))
        near_fake.save(raw_root / "cifake" / "fake" / "img_000004.png")
    kept, report = collapse(scan_assets(raw_root))
    # the near-identical real image collapses into the original; the fake
    # near-twin survives because a tampered asset must never hide behind
    # its source image
    assert report["near_removed"] == 1
    paths = {r.path for r in kept}
    assert "cifake/fake/img_000004.png" in paths


def test_byte_identical_label_conflict_is_quarantined(raw_root):
    original = raw_root / "cifake" / "real" / "img_000001.png"
    twin = raw_root / "cifake" / "fake" / "img_000009.png"
    twin.write_bytes(original.read_bytes())
    kept, report = collapse(scan_assets(raw_root))
    assert report["label_conflicts"] == 2
    assert not any("img_000001" in r.path or "img_000009" in r.path for r in kept)


def test_manifest_is_deterministic(raw_root, tmp_path):
    out1 = tmp_path / "m1.parquet"
    out2 = tmp_path / "m2.parquet"
    build_manifest(raw_root, out1)
    build_manifest(raw_root, out2)
    assert out1.read_bytes() == out2.read_bytes()
    frame = pd.read_parquet(out1)
    assert set(frame.columns) >= {
        "sha256",
        "source",
        "license",
        "label",
        "manipulation",
        "scene",
        "width",
        "height",
        "phash",
        "path",
    }
    assert frame.license.str.len().min() > 0


def test_same_scene_variants_are_kept(raw_root):
    base = raw_root / "micc_f220" / "MICC-F220" / "CRW_4809tamp1.jpg"
    with Image.open(base) as image:
        variant = image.copy()
        variant.putpixel((5, 5), (250, 250, 250))
        variant.save(raw_root / "micc_f220" / "MICC-F220" / "CRW_4809tamp2.jpg")
    kept, report = collapse(scan_assets(raw_root))
    # two manipulations of one base scene are distinct samples, not dupes
    names = {r.path.rsplit("/", 1)[-1] for r in kept}
    assert {"CRW_4809tamp1.jpg", "CRW_4809tamp2.jpg"} <= names


def test_distance_four_pair_with_spread_bits_is_found():
    from provenance_lens.data.manifest import AssetRow, collapse

    def row(phash, scene, sha):
        return AssetRow(
            sha256=sha,
            source="cifake",
            license="MIT",
            label="authentic",
            manipulation="none",
            scene=scene,
            width=32,
            height=32,
            phash=phash,
            path=f"cifake/real/{sha}.png",
        )

    base = 0
    # four differing bits spread across four distinct 13-bit bands: the
    # adversarial case a four-band index would miss
    spread = (1 << 0) | (1 << 13) | (1 << 26) | (1 << 39)
    kept, report = collapse(
        [row(f"{base:016x}", "s1", "a" * 64), row(f"{spread:016x}", "s2", "b" * 64)]
    )
    assert report["near_removed"] == 1
