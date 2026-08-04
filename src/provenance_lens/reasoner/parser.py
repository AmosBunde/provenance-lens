"""Strict verdict parser with a failure taxonomy.

Protocol: anything that is not exactly one JSON object conforming to the
contract is a parse failure. The only tolerated wrapper is a single fenced
code block. Failures carry a taxonomy label so the M5 report can break down
failure modes rather than reporting one opaque rate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum


class ParseFailure(StrEnum):
    NOT_JSON = "not_json"
    SCHEMA_VIOLATION = "schema_violation"
    LABEL_INVALID = "label_invalid"
    CONFIDENCE_OUT_OF_RANGE = "confidence_out_of_range"
    EVIDENCE_MALFORMED = "evidence_malformed"
    UNGROUNDED_EVIDENCE = "ungrounded_evidence"


@dataclass
class Evidence:
    signal: str
    region: str
    direction: str


@dataclass
class Verdict:
    label: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class ParseResult:
    verdict: Verdict | None = None
    failure: ParseFailure | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None and self.failure is None


_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(.*?)\n\s*```\s*\Z", re.S)
_ALLOWED_KEYS = {"label", "confidence", "evidence"}
_EVIDENCE_KEYS = {"signal", "region", "direction"}
_DIRECTIONS = {"supports_manipulated", "supports_authentic"}


def parse_verdict(raw: str) -> ParseResult:
    text = raw.strip()
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        return ParseResult(failure=ParseFailure.NOT_JSON, detail=str(error))
    if not isinstance(payload, dict) or set(payload) != _ALLOWED_KEYS:
        shape = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        return ParseResult(failure=ParseFailure.SCHEMA_VIOLATION, detail=f"keys: {shape}")
    label = payload["label"]
    if label not in ("authentic", "manipulated"):
        return ParseResult(failure=ParseFailure.LABEL_INVALID, detail=repr(label))
    confidence = payload["confidence"]
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        return ParseResult(failure=ParseFailure.CONFIDENCE_OUT_OF_RANGE, detail=repr(confidence))
    if not 0.0 <= float(confidence) <= 1.0:
        return ParseResult(failure=ParseFailure.CONFIDENCE_OUT_OF_RANGE, detail=repr(confidence))
    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list):
        return ParseResult(
            failure=ParseFailure.EVIDENCE_MALFORMED, detail=type(raw_evidence).__name__
        )
    evidence = []
    for item in raw_evidence:
        if (
            not isinstance(item, dict)
            or set(item) != _EVIDENCE_KEYS
            or not all(isinstance(item[k], str) for k in _EVIDENCE_KEYS)
            or item["direction"] not in _DIRECTIONS
        ):
            return ParseResult(failure=ParseFailure.EVIDENCE_MALFORMED, detail=repr(item))
        evidence.append(Evidence(item["signal"], item["region"], item["direction"]))
    return ParseResult(
        verdict=Verdict(label=label, confidence=float(confidence), evidence=evidence)
    )
