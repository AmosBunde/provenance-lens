"""Batch runner tests: resumability, cost accounting, failure recording."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from provenance_lens.forensics.store import COLUMNS
from provenance_lens.reasoner.batch import run_split
from provenance_lens.reasoner.client import MockBackend

GOOD = (
    '{"label": "manipulated", "confidence": 0.9, "evidence": '
    '[{"signal": "jpeg_ghost_depth", "region": "r1c1", "direction": "supports_manipulated"}]}'
)
BAD = "this is not json"


@pytest.fixture()
def split_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    rows, store_rows = [], []
    for i in range(4):
        array = np.clip(rng.normal(128, 30, (48, 48, 3)), 0, 255).astype(np.uint8)
        path = raw / f"img_{i}.png"
        Image.fromarray(array).save(path)
        sha = f"{i:064x}"
        rows.append({"sha256": sha, "path": f"img_{i}.png", "label": "authentic"})
        store_rows.append(
            (
                sha,
                "jpeg_ghost_depth",
                "r1c1",
                0.4,
                "higher_is_suspicious",
                "compression_ghosts",
                "1",
            )
        )
    features = tmp_path / "features"
    features.mkdir()
    pd.DataFrame(store_rows, columns=COLUMNS).to_parquet(
        features / "compression_ghosts-v1.parquet", index=False
    )
    frame = pd.DataFrame(rows)
    return frame, raw, features, tmp_path


def test_batch_runs_and_accounts_costs(split_fixture):
    frame, raw, features, tmp = split_fixture
    backend = MockBackend(replies=[GOOD, BAD, GOOD, GOOD])
    out = tmp / "verdicts.jsonl"
    report = run_split(
        "val", out_path=out, backend=backend, frame=frame, raw_root=raw, feature_dir=features
    )
    assert report["calls"] == 4
    assert report["parse_failures"] == 1
    assert report["input_tokens"] == 40 and report["output_tokens"] == 20
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 4
    failed = [r for r in records if not r["ok"]]
    assert failed[0]["failure"] == "not_json"
    assert out.with_suffix(".report.json").exists()


def test_batch_resumes_without_repeat_calls(split_fixture):
    frame, raw, features, tmp = split_fixture
    out = tmp / "verdicts.jsonl"
    first = MockBackend(replies=[GOOD])
    run_split(
        "val", out_path=out, backend=first, frame=frame.head(2), raw_root=raw, feature_dir=features
    )
    assert len(first.calls) == 2
    second = MockBackend(replies=[GOOD])
    report = run_split(
        "val", out_path=out, backend=second, frame=frame, raw_root=raw, feature_dir=features
    )
    assert report["skipped"] == 2
    assert report["calls"] == 2
    assert len(second.calls) == 2
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len({r["sha256"] for r in records}) == 4


def test_direct_test_split_is_refused():
    import pytest as _pytest

    from provenance_lens.reasoner.batch import run_split as _run

    with _pytest.raises(ValueError) as excinfo:
        _run("test")
    assert "harness" in str(excinfo.value)
