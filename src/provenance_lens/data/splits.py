"""Frozen, scene-aware train/validation/test splits.

Split assignment is a pure function of the manifest and a committed seed.
Rows are first clustered into scenes with a union-find over two edge types:
shared explicit scene ids, and label-agnostic perceptual-hash proximity
(Hamming distance at most ``NEAR_DUP_DISTANCE``). The second edge type is
load bearing: authentic images sit at pHash distance 0 from their copy-move
tampered versions, and MICC-F220 overlaps MICC-F2000 across source
boundaries, so neither filenames nor explicit scenes are enough on their own.

Whole clusters are then assigned to train, validation, and test (70/15/15 by
asset count), stratified by the cluster's dominant manipulation type, in a
deterministic seeded order. The test manifest is serialized canonically (the
sorted asset hashes, one per line) and its SHA-256 is committed at
``data/splits/test_manifest.sha256``; ``--verify`` recomputes and compares.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from provenance_lens.data.manifest import BAND_SLICES, NEAR_DUP_DISTANCE, _hamming

SPLIT_SEED = 20260803
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
STRATIFY_TOLERANCE = 0.03  # max abs deviation of a split's type share


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _near_dup_edges(phashes: list[int]) -> list[tuple[int, int]]:
    """All index pairs within the near-duplicate distance, via banded lookup."""
    edges = []
    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, value in enumerate(phashes):
        seen: set[int] = set()
        for band, (offset, width) in enumerate(BAND_SLICES):
            key = (band, (value >> offset) & ((1 << width) - 1))
            for j in band_index[key]:
                if j not in seen and _hamming(value, phashes[j]) <= NEAR_DUP_DISTANCE:
                    edges.append((i, j))
                seen.add(j)
            band_index[key].append(i)
    return edges


def cluster_scenes(frame: pd.DataFrame) -> pd.Series:
    """Return a cluster id per row: min sha256 of the connected component."""
    uf = _UnionFind(len(frame))
    by_scene: dict[str, int] = {}
    for i, scene in enumerate(frame["scene"]):
        if scene in by_scene:
            uf.union(i, by_scene[scene])
        else:
            by_scene[scene] = i
    phashes = [int(p, 16) for p in frame["phash"]]
    for a, b in _near_dup_edges(phashes):
        uf.union(a, b)
    shas = frame["sha256"].tolist()
    root_min: dict[int, str] = {}
    for i in range(len(frame)):
        root = uf.find(i)
        current = root_min.get(root)
        if current is None or shas[i] < current:
            root_min[root] = shas[i]
    return pd.Series([root_min[uf.find(i)] for i in range(len(frame))], index=frame.index)


def _order_key(cluster_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{cluster_id}".encode()).hexdigest()


def assign_splits(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Assign every row to train/val/test; whole clusters move together."""
    work = frame.copy()
    work["cluster"] = cluster_scenes(work)
    clusters = []
    for cluster_id, group in work.groupby("cluster"):
        dominant = group["manipulation"].value_counts().idxmax()
        clusters.append((cluster_id, dominant, len(group)))

    assigned: dict[str, str] = {}
    fill: dict[tuple[str, str], float] = defaultdict(float)  # (type, split) -> count
    by_type: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cluster_id, dominant, size in clusters:
        by_type[dominant].append((cluster_id, size))
    for dominant, members in sorted(by_type.items()):
        members.sort(key=lambda m: _order_key(m[0], seed))
        total = sum(size for _, size in members)
        for cluster_id, size in members:
            deficits = {split: RATIOS[split] * total - fill[(dominant, split)] for split in RATIOS}
            target = max(sorted(deficits), key=lambda s: deficits[s])
            assigned[cluster_id] = target
            fill[(dominant, target)] += size
    return work["cluster"].map(assigned)


def canonical_test_bytes(frame: pd.DataFrame) -> bytes:
    return ("\n".join(sorted(frame["sha256"])) + "\n").encode()


def build_splits(manifest_path: Path, out_dir: Path, seed: int = SPLIT_SEED) -> dict:
    frame = pd.read_parquet(manifest_path)
    frame = frame.sort_values("sha256").reset_index(drop=True)
    frame["split"] = assign_splits(frame, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"seed": seed, "counts": {}, "type_shares": {}}
    for split in RATIOS:
        part = frame[frame["split"] == split].drop(columns=["split"])
        part.to_parquet(out_dir / f"{split}_manifest.parquet", index=False)
        summary["counts"][split] = len(part)
        shares = (part["manipulation"].value_counts() / len(part)).round(4)
        summary["type_shares"][split] = shares.to_dict()
    test = frame[frame["split"] == "test"]
    digest = hashlib.sha256(canonical_test_bytes(test)).hexdigest()
    (out_dir / "test_manifest.sha256").write_text(digest + "\n")
    summary["test_sha256"] = digest
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify(manifest_path: Path, out_dir: Path, seed: int = SPLIT_SEED) -> bool:
    committed = (out_dir / "test_manifest.sha256").read_text().strip()
    frame = pd.read_parquet(manifest_path).sort_values("sha256").reset_index(drop=True)
    frame["split"] = assign_splits(frame, seed)
    digest = hashlib.sha256(canonical_test_bytes(frame[frame["split"] == "test"])).hexdigest()
    ok = digest == committed
    print(f"committed {committed}\nrecomputed {digest}\n{'MATCH' if ok else 'MISMATCH'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    manifest_path = Path("data/manifest.parquet")
    out_dir = Path("data/splits")
    if "--verify" in args:
        return 0 if verify(manifest_path, out_dir) else 1
    marker = out_dir / "split_summary.json"
    if marker.exists() and marker.stat().st_mtime >= manifest_path.stat().st_mtime:
        print("splits are newer than the manifest, skipping rebuild")
        return 0
    summary = build_splits(manifest_path, out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
