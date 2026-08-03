"""The leakage-impossibility test: runtime guard plus static reference check."""

from pathlib import Path

import pandas as pd
import pytest

from provenance_lens.data.access import (
    HarnessToken,
    TestSplitAccessError,
    load_split,
)
from provenance_lens.eval import harness

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "provenance_lens"
FORBIDDEN = ("test_manifest", "load_test_split", "HarnessToken")


@pytest.fixture()
def split_dir(tmp_path):
    frame = pd.DataFrame({"sha256": ["a" * 64], "label": ["authentic"], "manipulation": ["none"]})
    for name in ("train", "val", "test"):
        frame.to_parquet(tmp_path / f"{name}_manifest.parquet", index=False)
    return tmp_path


def test_train_and_val_load_freely(split_dir):
    assert len(load_split("train", split_dir=split_dir)) == 1
    assert len(load_split("val", split_dir=split_dir)) == 1


def test_test_split_raises_without_token(split_dir):
    with pytest.raises(TestSplitAccessError):
        load_split("test", split_dir=split_dir)


def test_token_cannot_be_constructed_outside_harness():
    with pytest.raises(TestSplitAccessError) as excinfo:
        HarnessToken()
    assert "tests.test_leakage" in str(excinfo.value) or "test_leakage" in str(excinfo.value)


def test_fake_token_object_is_rejected(split_dir):
    class Impostor:
        pass

    with pytest.raises(TestSplitAccessError):
        load_split("test", token=Impostor(), split_dir=split_dir)


def test_harness_loads_test_split(split_dir):
    frame = harness.load_test_split(split_dir=split_dir)
    assert len(frame) == 1


def find_forbidden_references(root: Path) -> list[str]:
    """Modules outside eval/ that mention the test manifest or the harness
    loader. Two producers are allowed: the access guard itself, and the
    split builder, which creates the artifacts and cannot avoid naming
    them; the rule exists for consumers."""
    allowed = {Path("data/access.py"), Path("data/splits.py")}
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] == "eval" or relative in allowed:
            continue
        content = path.read_text()
        for needle in FORBIDDEN:
            if needle in content:
                offenders.append(f"{relative}:{needle}")
    return offenders


def test_no_module_outside_eval_references_the_test_split():
    assert find_forbidden_references(SRC_ROOT) == []


def test_reference_checker_catches_a_violation(tmp_path):
    tree = tmp_path / "pkg"
    (tree / "eval").mkdir(parents=True)
    (tree / "baseline").mkdir()
    (tree / "eval" / "harness.py").write_text("test_manifest = 'allowed here'\n")
    offender = tree / "baseline" / "sneaky.py"
    offender.write_text("frame = read('data/splits/test_manifest.parquet')\n")
    assert find_forbidden_references(tree) == ["baseline/sneaky.py:test_manifest"]
