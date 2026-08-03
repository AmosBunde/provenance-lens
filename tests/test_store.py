"""Feature store tests: registry, resumability, version invalidation."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from provenance_lens.forensics import store as store_module
from provenance_lens.forensics.store import (
    EXTRACTORS,
    build_store,
    load_features,
    shard_path,
    signal_vocabulary,
)


@pytest.fixture()
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        array = np.clip(rng.normal(128, 30, (96, 96)), 0, 255).astype(np.uint8)
        path = raw / f"img_{i}.png"
        Image.fromarray(array).save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"sha256": sha, "path": f"img_{i}.png"})
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest, raw, tmp_path / "features", rows


def test_registry_covers_all_four_families():
    assert set(EXTRACTORS) == {
        "compression_ghosts",
        "blocking_grid",
        "noise_residuals",
        "edges_lighting",
    }


def test_store_builds_and_loads(corpus):
    manifest, raw, features, rows = corpus
    report = build_store(manifest, raw, features, workers=2)
    assert all(report["computed"][e] == 3 for e in EXTRACTORS)
    frame = load_features(feature_dir=features)
    assert set(frame.sha256.unique()) == {r["sha256"] for r in rows}
    assert set(frame.extractor.unique()) == set(EXTRACTORS)
    vocabulary = signal_vocabulary(rows[0]["sha256"], features)
    assert ("jpeg_ghost_depth", "r1c1") in vocabulary
    assert ("noise_residual_std", "global") not in vocabulary


def test_second_run_skips_everything(corpus):
    manifest, raw, features, rows = corpus
    build_store(manifest, raw, features, workers=2)
    report = build_store(manifest, raw, features, workers=2)
    assert all(report["computed"][e] == 0 for e in EXTRACTORS)
    assert all(report["skipped"][e] == 3 for e in EXTRACTORS)


def test_version_bump_invalidates_one_extractor(corpus, monkeypatch):
    manifest, raw, features, rows = corpus
    build_store(manifest, raw, features, workers=2)
    extract, _ = EXTRACTORS["blocking_grid"]
    monkeypatch.setitem(store_module.EXTRACTORS, "blocking_grid", (extract, "2"))
    report = build_store(manifest, raw, features, workers=2)
    assert report["computed"]["blocking_grid"] == 3
    assert report["computed"]["compression_ghosts"] == 0
    assert shard_path("blocking_grid", features).name.endswith("v2.parquet")


def test_new_asset_is_added_incrementally(corpus):
    manifest, raw, features, rows = corpus
    build_store(manifest, raw, features, workers=2)
    array = np.clip(np.random.default_rng(9).normal(120, 25, (96, 96)), 0, 255).astype(np.uint8)
    path = raw / "img_new.png"
    Image.fromarray(array).save(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_parquet(manifest)
    frame = pd.concat(
        [frame, pd.DataFrame([{"sha256": sha, "path": "img_new.png"}])],
        ignore_index=True,
    )
    frame.to_parquet(manifest, index=False)
    report = build_store(manifest, raw, features, workers=2)
    assert all(report["computed"][e] == 1 for e in EXTRACTORS)
    assert sha in set(load_features(feature_dir=features).sha256.unique())
