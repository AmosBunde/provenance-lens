"""The parquet feature store: every forensic signal for every asset.

The store is the single vocabulary of the project. The baseline reads
values from it, the prompt template renders it, and the grounding check
validates that every cited ``(signal, region)`` pair exists here for the
asset in question. Rows are keyed by asset SHA-256 and carry the signal
name, region, value, direction of suspicion, extractor name, and extractor
version.

Layout: one parquet shard per extractor version at
``data/features/<extractor>-v<version>.parquet``. Runs are resumable at
asset granularity (a shard's existing assets are skipped) and a version
bump changes the shard filename, which invalidates exactly that
extractor's signals and nothing else. Extraction parallelizes across
processes; output row order is canonicalized before writing so the store
is byte-deterministic regardless of scheduling.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from provenance_lens.forensics import (
    blocking_grid,
    compression_ghosts,
    edges_lighting,
    noise_residuals,
)

EXTRACTORS: dict[str, tuple] = {
    "compression_ghosts": (compression_ghosts.extract, "1"),
    "blocking_grid": (blocking_grid.extract, "1"),
    "noise_residuals": (noise_residuals.extract, "1"),
    "edges_lighting": (edges_lighting.extract, "1"),
}

FEATURE_DIR = Path("data/features")
RAW_ROOT = Path("data/raw")
COLUMNS = ["sha256", "signal", "region", "value", "direction", "extractor", "version"]


def shard_path(extractor: str, feature_dir: Path = FEATURE_DIR) -> Path:
    version = EXTRACTORS[extractor][1]
    return feature_dir / f"{extractor}-v{version}.parquet"


def _extract_one(args: tuple[str, str, str, str]) -> list[tuple]:
    sha, path, extractor, raw_root = args
    extract = EXTRACTORS[extractor][0]
    version = EXTRACTORS[extractor][1]
    with Image.open(Path(raw_root) / path) as image:
        image.load()
        signals = extract(image)
    return [(sha, s.name, s.region, s.value, str(s.direction), extractor, version) for s in signals]


def build_store(
    manifest_path: Path,
    raw_root: Path = RAW_ROOT,
    feature_dir: Path = FEATURE_DIR,
    workers: int | None = None,
) -> dict:
    manifest = pd.read_parquet(manifest_path, columns=["sha256", "path"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"assets": len(manifest), "computed": {}, "skipped": {}}
    for extractor in sorted(EXTRACTORS):
        shard = shard_path(extractor, feature_dir)
        if shard.exists():
            existing = pd.read_parquet(shard, columns=["sha256"])
            done = set(existing.sha256.unique())
        else:
            done = set()
        todo = manifest[~manifest.sha256.isin(done)]
        report["skipped"][extractor] = len(manifest) - len(todo)
        report["computed"][extractor] = len(todo)
        if todo.empty:
            continue
        jobs = [
            (row.sha256, row.path, extractor, str(raw_root)) for row in todo.itertuples(index=False)
        ]
        rows: list[tuple] = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for i, result in enumerate(pool.map(_extract_one, jobs, chunksize=64)):
                rows.extend(result)
                if (i + 1) % 5000 == 0:
                    print(f"  {extractor}: {i + 1}/{len(jobs)}", flush=True)
        new_frame = pd.DataFrame(rows, columns=COLUMNS)
        if shard.exists():
            new_frame = pd.concat([pd.read_parquet(shard), new_frame], ignore_index=True)
        new_frame = new_frame.sort_values(["sha256", "signal", "region"]).reset_index(drop=True)
        new_frame.to_parquet(shard, index=False)
        print(f"{extractor}: shard now {len(new_frame)} rows", flush=True)
    return report


def load_features(shas: list[str] | None = None, feature_dir: Path = FEATURE_DIR) -> pd.DataFrame:
    frames = []
    for extractor in sorted(EXTRACTORS):
        shard = shard_path(extractor, feature_dir)
        if shard.exists():
            frame = pd.read_parquet(shard)
            if shas is not None:
                frame = frame[frame.sha256.isin(shas)]
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def signal_vocabulary(sha: str, feature_dir: Path = FEATURE_DIR) -> set[tuple[str, str]]:
    """The set of (signal, region) pairs that exist for an asset; the
    grounding check validates citations against exactly this set."""
    features = load_features([sha], feature_dir)
    return set(zip(features.signal, features.region, strict=True))


def main() -> int:
    report = build_store(Path("data/manifest.parquet"))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
