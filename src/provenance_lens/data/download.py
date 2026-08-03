"""Verified dataset download and extraction.

Every asset enters the project through ``configs/data_sources.yaml``. For each
enabled source this module fetches the configured files, verifies their SHA-256
against the pinned value, extracts them under ``data/raw/<source>/``, and writes
a ``source_record.json`` that attributes every extracted asset to its source,
license, and archive checksum. Verified archives are skipped on re-runs, which
makes the stage idempotent.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CHUNK = 1 << 20


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match its pinned checksum."""

    def __init__(self, url: str, expected: str, actual: str):
        super().__init__(
            f"checksum mismatch for {url}: expected {expected}, got {actual}; "
            "the file is corrupt or the upstream content changed"
        )
        self.expected = expected
        self.actual = actual


@dataclass
class SourceFile:
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass
class Source:
    name: str
    kind: str
    license: str
    classes: list[str]
    enabled: bool
    files: list[SourceFile] = field(default_factory=list)
    notes: str = ""


def load_sources(config_path: Path) -> list[Source]:
    raw = yaml.safe_load(config_path.read_text())
    sources = []
    for entry in raw["sources"]:
        sources.append(
            Source(
                name=entry["name"],
                kind=entry["kind"],
                license=entry["license"],
                classes=list(entry["classes"]),
                enabled=bool(entry["enabled"]),
                files=[SourceFile(u["url"], u["sha256"]) for u in entry.get("urls", [])],
                notes=entry.get("notes", ""),
            )
        )
    return sources


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class TruncatedDownloadError(RuntimeError):
    """Raised when a server closes the connection before Content-Length bytes."""

    def __init__(self, url: str, expected: int, actual: int):
        super().__init__(
            f"truncated download for {url}: got {actual} of {expected} bytes; "
            "the connection was cut before the file completed"
        )


def fetch(url: str, dest: Path) -> None:
    """Stream a URL to disk. Supports http(s) and file:// for tests and
    locally generated sources. A response shorter than its declared
    Content-Length raises instead of leaving a silently truncated file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
        declared = response.headers.get("Content-Length") if response.headers else None
        shutil.copyfileobj(response, out, CHUNK)
    if declared is not None:
        actual = tmp.stat().st_size
        if actual != int(declared):
            tmp.unlink()
            raise TruncatedDownloadError(url, int(declared), actual)
    tmp.replace(dest)


def _marker(archive_dir: Path, source_file: SourceFile) -> Path:
    return archive_dir / f"{source_file.filename}.verified"


def ensure_file(source_file: SourceFile, archive_dir: Path) -> Path:
    """Download and verify one file, skipping work already proven done."""
    dest = archive_dir / source_file.filename
    marker = _marker(archive_dir, source_file)
    if marker.exists() and marker.read_text().strip() == source_file.sha256:
        return dest
    if not dest.exists():
        print(f"  fetching {source_file.url}", flush=True)
        fetch(source_file.url, dest)
    actual = sha256_of(dest)
    if actual != source_file.sha256:
        dest.unlink()
        raise ChecksumMismatchError(source_file.url, source_file.sha256, actual)
    marker.write_text(source_file.sha256 + "\n")
    return dest


def extract_zip(archive: Path, out_dir: Path) -> int:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
        return sum(1 for i in zf.infolist() if not i.is_dir())


def materialize_parquet(parquet: Path, out_dir: Path) -> int:
    """Write parquet-embedded images out as PNG files, one directory per label."""
    import pandas as pd
    from PIL import Image

    frame = pd.read_parquet(parquet)
    label_names = {0: "real", 1: "fake"}
    count = 0
    stem = parquet.stem
    for idx, row in enumerate(frame.itertuples(index=False)):
        image = row.image
        payload = image["bytes"] if isinstance(image, dict) else image
        label = label_names.get(row.label, str(row.label))
        target = out_dir / label / f"{stem}_{idx:06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            Image.open(io.BytesIO(payload)).save(target)
        count += 1
    return count


def process_source(source: Source, raw_root: Path) -> dict:
    archive_dir = raw_root / "_archives"
    out_dir = raw_root / source.name
    record_path = out_dir / "source_record.json"
    expected_files = [{"url": f.url, "sha256": f.sha256} for f in source.files]
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("files") == expected_files:
            print(f"  {source.name} already extracted, skipping", flush=True)
            return previous
    extracted = 0
    for source_file in source.files:
        local = ensure_file(source_file, archive_dir)
        if source.kind == "archive":
            extracted += extract_zip(local, out_dir)
        elif source.kind == "hf_parquet":
            extracted += materialize_parquet(local, out_dir)
        else:
            raise ValueError(f"unknown source kind: {source.kind}")
    record = {
        "source": source.name,
        "license": source.license,
        "classes": source.classes,
        "files": [{"url": f.url, "sha256": f.sha256} for f in source.files],
        "extracted_files": extracted,
        "notes": source.notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(config_path: Path, raw_root: Path) -> list[dict]:
    records = []
    for source in load_sources(config_path):
        if not source.enabled:
            print(f"skipping {source.name} (disabled: credential gated)", flush=True)
            continue
        print(f"source {source.name}", flush=True)
        records.append(process_source(source, raw_root))
    return records


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    config = Path(args[0]) if args else Path("configs/data_sources.yaml")
    raw_root = Path(args[1]) if len(args) > 1 else Path("data/raw")
    records = run(config, raw_root)
    total = sum(r["extracted_files"] for r in records)
    print(f"done: {len(records)} sources, {total} files extracted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
