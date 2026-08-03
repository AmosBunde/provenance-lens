"""Canonical asset manifest: hashing, labeling, and near-duplicate collapse.

Walks ``data/raw/<source>/`` for every source that has a ``source_record.json``,
derives per-asset labels from source-specific conventions, computes the SHA-256
byte hash (the project-wide asset key), a perceptual hash, and the resolution,
collapses exact and near duplicates, and writes ``data/manifest.parquet``
deterministically together with a dedupe report.

Label conventions per source:

- ``micc_f220`` and ``micc_f2000``: filenames containing ``tamp`` are copy-move
  tampered, everything else is authentic (verified against the shipped ground
  truth list, which agrees exactly). The scene id is the filename stem before
  any ``tamp`` or ``_`` suffix, so splits can keep whole scenes together.
- ``cifake``: the ``real`` directory is authentic, ``fake`` is AI generated.
  There is no scene structure; the perceptual hash is the only grouping key.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PHASH_SIZE = 8  # 64-bit perceptual hash
NEAR_DUP_DISTANCE = 4  # max Hamming distance treated as a near duplicate
# Five bands guarantee that any pair within Hamming distance 4 shares at
# least one untouched band (pigeonhole: 4 differing bits cannot touch all 5
# bands), so banded candidate lookup has full recall at the threshold.
BAND_SLICES = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]

_MICC_SCENE = re.compile(r"^(?P<scene>[A-Za-z]*_?\d+)")


@dataclass
class AssetRow:
    sha256: str
    source: str
    license: str
    label: str
    manipulation: str
    scene: str
    width: int
    height: int
    phash: str
    path: str


def _micc_label(name: str) -> tuple[str, str]:
    if "tamp" in name.lower():
        return "manipulated", "copy_move"
    return "authentic", "none"


def _micc_scene(name: str) -> str:
    match = _MICC_SCENE.match(name)
    return match.group("scene") if match else name


def classify(source: str, path: Path) -> tuple[str, str, str]:
    """Return (label, manipulation, scene) for one asset path."""
    name = path.name
    if source.startswith("micc"):
        label, manipulation = _micc_label(name)
        return label, manipulation, f"{source}:{_micc_scene(name)}"
    if source == "cifake":
        if path.parent.name == "fake":
            return "manipulated", "ai_generated", f"cifake:{path.stem}"
        return "authentic", "none", f"cifake:{path.stem}"
    raise ValueError(f"no label convention for source {source}")


def hash_and_measure(path: Path) -> tuple[str, str, int, int]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        width, height = image.size
        phash = str(imagehash.phash(image, hash_size=PHASH_SIZE))
    return sha, phash, width, height


def scan_assets(raw_root: Path) -> list[AssetRow]:
    rows = []
    for record_path in sorted(raw_root.glob("*/source_record.json")):
        source_dir = record_path.parent
        record = json.loads(record_path.read_text())
        source, license_ = record["source"], record["license"]
        for path in sorted(source_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            label, manipulation, scene = classify(source, path)
            sha, phash, width, height = hash_and_measure(path)
            rows.append(
                AssetRow(
                    sha256=sha,
                    source=source,
                    license=license_,
                    label=label,
                    manipulation=manipulation,
                    scene=scene,
                    width=width,
                    height=height,
                    phash=phash,
                    path=str(path.relative_to(raw_root)),
                )
            )
    return rows


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def collapse(rows: list[AssetRow]) -> tuple[list[AssetRow], dict]:
    """Collapse exact byte duplicates, then near duplicates by perceptual hash.

    Near-duplicate candidates are found by splitting the 64-bit hash into
    five bands; any pair within Hamming distance ``NEAR_DUP_DISTANCE`` = 4
    differs in at most 4 bits, so at least one of the five bands is identical
    and the pair always meets in a candidate bucket. Candidates are verified
    with the exact distance before collapsing. Two guards limit the merge:
    only assets with the same label collapse (a tampered image can never hide
    behind its original), and only assets from different scenes collapse.
    Same-scene near variants (multiple manipulations of one base image) are
    intentional distinct samples, and the scene-aware splits already keep
    them in one split; cross-scene near duplicates are true duplicates such
    as re-encodes and burst shots, which are exactly the split-leakage risk.
    """
    rows = sorted(rows, key=lambda r: r.sha256)
    report = {"input": len(rows), "exact_removed": 0, "near_removed": 0, "label_conflicts": 0}

    by_sha: dict[str, list[AssetRow]] = defaultdict(list)
    for row in rows:
        by_sha[row.sha256].append(row)
    unique: list[AssetRow] = []
    for group in by_sha.values():
        labels = {r.label for r in group}
        if len(labels) > 1:
            # byte-identical files with conflicting labels are a data
            # integrity error; quarantine them rather than pick a side
            report["label_conflicts"] += len(group)
            continue
        report["exact_removed"] += len(group) - 1
        unique.append(group[0])

    kept: list[AssetRow] = []
    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept_hashes: list[int] = []

    def band_keys(value: int):
        for band, (offset, width) in enumerate(BAND_SLICES):
            yield band, (value >> offset) & ((1 << width) - 1)

    for row in unique:
        value = int(row.phash, 16)
        candidate_ids = set()
        for key in band_keys(value):
            candidate_ids.update(band_index[key])
        duplicate = any(
            _hamming(value, kept_hashes[i]) <= NEAR_DUP_DISTANCE
            and kept[i].label == row.label
            and kept[i].scene != row.scene
            for i in candidate_ids
        )
        if duplicate:
            report["near_removed"] += 1
            continue
        index = len(kept)
        kept.append(row)
        kept_hashes.append(value)
        for key in band_keys(value):
            band_index[key].append(index)
    report["output"] = len(kept)
    return kept, report


def build_manifest(raw_root: Path, out_path: Path) -> dict:
    rows = scan_assets(raw_root)
    kept, report = collapse(rows)
    frame = pd.DataFrame([asdict(r) for r in kept]).sort_values("sha256").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    report["by_source"] = frame.groupby("source").size().to_dict()
    report["by_label"] = frame.groupby("label").size().to_dict()
    (out_path.parent / "dedupe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    raw_root = Path("data/raw")
    out_path = Path("data/manifest.parquet")
    records = list(raw_root.glob("*/source_record.json"))
    if out_path.exists() and records:
        newest_input = max(p.stat().st_mtime for p in records)
        if out_path.stat().st_mtime >= newest_input:
            print("manifest is newer than every source record, skipping rebuild")
            return 0
    report = build_manifest(raw_root, out_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
