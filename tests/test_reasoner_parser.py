"""Parser taxonomy tests: one fixture per failure mode."""

import pytest

from provenance_lens.reasoner.parser import ParseFailure, parse_verdict

VALID = (
    '{"label": "manipulated", "confidence": 0.82, "evidence": '
    '[{"signal": "jpeg_ghost_depth", "region": "r1c2", '
    '"direction": "supports_manipulated"}]}'
)


def test_valid_verdict_parses():
    result = parse_verdict(VALID)
    assert result.ok
    assert result.verdict.label == "manipulated"
    assert result.verdict.confidence == 0.82
    assert result.verdict.evidence[0].signal == "jpeg_ghost_depth"


def test_single_fenced_block_is_stripped():
    result = parse_verdict(f"```json\n{VALID}\n```")
    assert result.ok


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("the image looks fake to me", ParseFailure.NOT_JSON),
        ('{"label": "manipulated"}', ParseFailure.SCHEMA_VIOLATION),
        (
            '{"label": "fake", "confidence": 0.5, "evidence": []}',
            ParseFailure.LABEL_INVALID,
        ),
        (
            '{"label": "authentic", "confidence": 1.7, "evidence": []}',
            ParseFailure.CONFIDENCE_OUT_OF_RANGE,
        ),
        (
            '{"label": "authentic", "confidence": true, "evidence": []}',
            ParseFailure.CONFIDENCE_OUT_OF_RANGE,
        ),
        (
            '{"label": "authentic", "confidence": 0.5, "evidence": [{"signal": "x"}]}',
            ParseFailure.EVIDENCE_MALFORMED,
        ),
        (
            '{"label": "authentic", "confidence": 0.5, "evidence": '
            '[{"signal": "x", "region": "r0c0", "direction": "maybe"}]}',
            ParseFailure.EVIDENCE_MALFORMED,
        ),
        (f"Sure! Here is my analysis:\n{VALID}", ParseFailure.NOT_JSON),
        (f"{VALID}\ntrailing commentary", ParseFailure.NOT_JSON),
    ],
)
def test_failure_taxonomy(raw, expected):
    result = parse_verdict(raw)
    assert not result.ok
    assert result.failure is expected
