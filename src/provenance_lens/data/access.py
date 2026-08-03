"""Split loading with the test split gated behind a harness-only capability.

``load_split("train")`` and ``load_split("val")`` are free. ``load_split("test")``
requires a :class:`HarnessToken`, and the token constructor verifies via the
call stack that it is being constructed inside ``provenance_lens.eval.harness``.
Everything else raises :class:`TestSplitAccessError` before any file is opened.

The guard stops honest mistakes and makes dishonest access loud and greppable;
a determined caller could forge frame globals, and the static reference check
in the leakage test exists to catch exactly that kind of code in review: no
module outside ``eval/`` may mention the test manifest or the harness loader.

This module never spells the test manifest filename literally; paths are
derived from the split name, so the static check applies to this file too.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

DEFAULT_SPLIT_DIR = Path("data/splits")
_HARNESS_MODULE = "provenance_lens.eval.harness"
SPLITS = ("train", "val", "test")


class TestSplitAccessError(RuntimeError):
    """Raised when anything but the eval harness touches the test split."""

    __test__ = False  # the Test prefix is domain naming, not a pytest class


class HarnessToken:
    """Capability proving the caller is the eval harness module."""

    def __init__(self) -> None:
        caller = inspect.stack()[1].frame.f_globals.get("__name__", "")
        if caller != _HARNESS_MODULE:
            raise TestSplitAccessError(
                f"HarnessToken can only be constructed inside {_HARNESS_MODULE}; "
                f"attempted from {caller or 'unknown module'}"
            )


def load_split(
    name: str,
    token: HarnessToken | None = None,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> pd.DataFrame:
    """Load one split manifest. The test split requires a harness token."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    if name == "test" and not isinstance(token, HarnessToken):
        raise TestSplitAccessError(
            "the test split is frozen; only the eval harness may read it "
            "(see README, data rules)"
        )
    return pd.read_parquet(split_dir / f"{name}_manifest.parquet")
