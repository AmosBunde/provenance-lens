"""Prompt rendering golden tests."""

from pathlib import Path

import pandas as pd
import pytest

from provenance_lens.forensics.store import COLUMNS
from provenance_lens.reasoner.prompt import build_prompt, estimate_tokens

GOLDEN = Path(__file__).parent / "golden" / "prompt_two_signals.txt"


def _features(sha="a" * 64):
    rows = [
        (
            sha,
            "noise_region_mismatch",
            "r1c1",
            0.91234,
            "higher_is_suspicious",
            "noise_residuals",
            "1",
        ),
        (
            sha,
            "jpeg_ghost_depth",
            "r0c2",
            0.4567,
            "higher_is_suspicious",
            "compression_ghosts",
            "1",
        ),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_prompt_matches_golden_file():
    prompt = build_prompt(_features())
    assert prompt == GOLDEN.read_text()


def test_rendering_is_sorted_and_verbatim():
    prompt = build_prompt(_features())
    assert "jpeg_ghost_depth [r0c2] = 0.4567" in prompt
    assert "noise_region_mismatch [r1c1] = 0.9123" in prompt
    assert prompt.index("Compression") < prompt.index("Noise residuals")


def test_empty_features_refuse():
    with pytest.raises(ValueError):
        build_prompt(_features().iloc[0:0])


def test_token_estimate_is_positive():
    assert estimate_tokens(build_prompt(_features())) > 100
