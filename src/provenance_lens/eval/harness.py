"""Evaluation harness: the sole reader of the frozen test split.

Scoring, calibration, and reporting land at M5; this module exists from M1
because it is the only place allowed to construct the test-split capability.
Every test-set evaluation in the project enters through here, which is what
makes the single-scoring rule for the baseline and the reasoner auditable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from provenance_lens.data.access import DEFAULT_SPLIT_DIR, HarnessToken, load_split


def load_test_split(split_dir: Path = DEFAULT_SPLIT_DIR) -> pd.DataFrame:
    """Load the frozen test split. Callable only through the harness module."""
    return load_split("test", token=HarnessToken(), split_dir=split_dir)


def reason_on_test(**kwargs) -> dict:
    """Run reasoner batch inference over the test split. This is the sole
    entry point for test-split inference; the batch runner refuses the
    test split unless the frame arrives from here."""
    from provenance_lens.reasoner.batch import run_split

    return run_split("test", frame=load_test_split(), **kwargs)
