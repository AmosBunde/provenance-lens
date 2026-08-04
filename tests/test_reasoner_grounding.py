"""Grounding tests: existence in the store decides citation validity."""

import numpy as np
import pandas as pd
from PIL import Image

from provenance_lens.forensics.store import COLUMNS
from provenance_lens.reasoner.client import MockBackend
from provenance_lens.reasoner.grounding import ground, reason_about
from provenance_lens.reasoner.parser import ParseFailure, parse_verdict


def _fixture_store(tmp_path, sha):
    rows = [
        (sha, "jpeg_ghost_depth", "r1c1", 0.4, "higher_is_suspicious", "compression_ghosts", "1"),
        (sha, "noise_region_mismatch", "r1c1", 0.9, "higher_is_suspicious", "noise_residuals", "1"),
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    (tmp_path / "features").mkdir()
    frame.to_parquet(tmp_path / "features" / "compression_ghosts-v1.parquet", index=False)
    return tmp_path / "features"


def _image():
    return Image.fromarray(
        np.clip(np.random.default_rng(1).normal(128, 30, (64, 64, 3)), 0, 255).astype("uint8")
    )


GOOD = (
    '{"label": "manipulated", "confidence": 0.9, "evidence": '
    '[{"signal": "jpeg_ghost_depth", "region": "r1c1", "direction": "supports_manipulated"}]}'
)
INVENTED = (
    '{"label": "manipulated", "confidence": 0.9, "evidence": '
    '[{"signal": "made_up_signal", "region": "r1c1", "direction": "supports_manipulated"}]}'
)
WRONG_REGION = (
    '{"label": "manipulated", "confidence": 0.9, "evidence": '
    '[{"signal": "jpeg_ghost_depth", "region": "r2c2", "direction": "supports_manipulated"}]}'
)


def test_grounded_citation_passes(tmp_path):
    sha = "a" * 64
    features = _fixture_store(tmp_path, sha)
    result = reason_about(sha, _image(), MockBackend(replies=[GOOD]), features)
    assert result.ok
    assert result.grounding_rate == 1.0


def test_invented_signal_fails_whole_response(tmp_path):
    sha = "a" * 64
    features = _fixture_store(tmp_path, sha)
    result = reason_about(sha, _image(), MockBackend(replies=[INVENTED]), features)
    assert not result.ok
    assert result.parse.failure is ParseFailure.UNGROUNDED_EVIDENCE
    assert "made_up_signal" in result.parse.detail


def test_existing_signal_wrong_region_fails(tmp_path):
    sha = "a" * 64
    features = _fixture_store(tmp_path, sha)
    result = reason_about(sha, _image(), MockBackend(replies=[WRONG_REGION]), features)
    assert not result.ok
    assert result.parse.failure is ParseFailure.UNGROUNDED_EVIDENCE


def test_empty_evidence_is_grounded_but_rate_zero(tmp_path):
    sha = "a" * 64
    features = _fixture_store(tmp_path, sha)
    empty = '{"label": "authentic", "confidence": 0.6, "evidence": []}'
    result = reason_about(sha, _image(), MockBackend(replies=[empty]), features)
    assert result.ok
    assert result.grounding_rate == 0.0


def test_ground_function_passthrough_on_parse_failure():
    bad = parse_verdict("not json at all")
    grounded, missing = ground(bad, {("a", "b")})
    assert grounded.failure is bad.failure
    assert missing == []
