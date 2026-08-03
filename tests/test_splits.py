"""Split tests: purity, cluster integrity, leakage impossibility, stratification."""

import random

import pandas as pd

from provenance_lens.data.splits import (
    RATIOS,
    assign_splits,
    build_splits,
    canonical_test_bytes,
    cluster_scenes,
)


def _manifest(rows):
    return pd.DataFrame(
        rows,
        columns=["sha256", "source", "license", "label", "manipulation", "scene", "phash"],
    )


def _row(i, scene, phash, label="authentic", manipulation="none"):
    return (f"{i:064x}", "src", "lic", label, manipulation, scene, f"{phash:016x}")


def _random_manifest(n=600, seed=7):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        manipulation = rng.choice(["none", "copy_move", "ai_generated"])
        label = "authentic" if manipulation == "none" else "manipulated"
        rows.append(_row(i, f"scene{i}", rng.getrandbits(64), label, manipulation))
    return _manifest(rows)


def test_assignment_is_pure():
    frame = _random_manifest()
    a = assign_splits(frame)
    b = assign_splits(frame)
    assert a.equals(b)


def test_near_duplicate_pair_lands_in_one_split():
    # cross-label, cross-source style twins at pHash distance 1: the
    # authentic original and its copy-move manipulation must not straddle
    rows = [_row(i, f"scene{i}", (i + 1) << 32) for i in range(200)]
    rows.append(_row(900, "sceneA", 0b0110, "authentic", "none"))
    rows.append(_row(901, "sceneB", 0b0111, "manipulated", "copy_move"))
    frame = _manifest(rows)
    splits = assign_splits(frame)
    pair = splits[frame.scene.isin(["sceneA", "sceneB"])]
    assert pair.nunique() == 1


def test_explicit_scene_stays_together():
    rows = [_row(i, f"scene{i % 50}", (i + 1) * 12345678901) for i in range(300)]
    frame = _manifest(rows)
    frame["split"] = assign_splits(frame)
    assert (frame.groupby("scene")["split"].nunique() == 1).all()


def test_cluster_ids_are_stable_min_sha():
    rows = [
        _row(0, "sceneX", 0b1),
        _row(1, "sceneX", 1 << 40),
        _row(2, "sceneY", 0b11),  # near-dup of row 0 at distance 1
    ]
    frame = _manifest(rows)
    clusters = cluster_scenes(frame)
    assert clusters.nunique() == 1
    assert (clusters == f"{0:064x}").all()


def test_ratios_and_stratification_within_tolerance():
    frame = _random_manifest(n=3000)
    frame["split"] = assign_splits(frame)
    counts = frame.split.value_counts(normalize=True)
    for split, ratio in RATIOS.items():
        assert abs(counts[split] - ratio) < 0.05
    for manipulation, group in frame.groupby("manipulation"):
        shares = group.split.value_counts(normalize=True)
        for split, ratio in RATIOS.items():
            assert abs(shares.get(split, 0.0) - ratio) < 0.06, (manipulation, split)


def test_build_writes_committed_hash_and_is_deterministic(tmp_path):
    frame = _random_manifest(n=400)
    frame["width"] = 64
    frame["height"] = 64
    frame["path"] = [f"src/{i}.png" for i in range(len(frame))]
    manifest_path = tmp_path / "manifest.parquet"
    frame.to_parquet(manifest_path, index=False)
    s1 = build_splits(manifest_path, tmp_path / "s1")
    s2 = build_splits(manifest_path, tmp_path / "s2")
    assert s1["test_sha256"] == s2["test_sha256"]
    committed = (tmp_path / "s1" / "test_manifest.sha256").read_text().strip()
    assert committed == s1["test_sha256"]
    test_frame = pd.read_parquet(tmp_path / "s1" / "test_manifest.parquet")
    assert canonical_test_bytes(test_frame) == canonical_test_bytes(
        pd.read_parquet(tmp_path / "s2" / "test_manifest.parquet")
    )


def test_changed_manifest_changes_test_hash(tmp_path):
    frame = _random_manifest(n=400)
    frame["width"] = 64
    frame["height"] = 64
    frame["path"] = [f"src/{i}.png" for i in range(len(frame))]
    p1 = tmp_path / "m1.parquet"
    frame.to_parquet(p1, index=False)
    s1 = build_splits(p1, tmp_path / "a")
    test_frame = pd.read_parquet(tmp_path / "a" / "test_manifest.parquet")
    removed = test_frame.sha256.iloc[0]
    frame2 = frame[frame.sha256 != removed]
    p2 = tmp_path / "m2.parquet"
    frame2.to_parquet(p2, index=False)
    s2 = build_splits(p2, tmp_path / "b")
    assert s1["test_sha256"] != s2["test_sha256"]
