"""Evidence grounding: every citation must exist in the store, or the whole
response fails.

``reason_about`` is the single public path from an image to a verdict; it
chains prompt construction, the backend, strict parsing, and grounding so no
consumer can skip the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from provenance_lens.forensics.store import FEATURE_DIR, load_features
from provenance_lens.reasoner.client import VlmBackend
from provenance_lens.reasoner.parser import (
    ParseFailure,
    ParseResult,
    Verdict,
    parse_verdict,
)
from provenance_lens.reasoner.prompt import build_prompt, estimate_tokens


@dataclass
class GroundedResult:
    sha256: str
    parse: ParseResult
    grounding_rate: float = 0.0
    ungrounded: list[tuple[str, str]] = field(default_factory=list)
    prompt_tokens_estimate: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.parse.ok

    @property
    def verdict(self) -> Verdict | None:
        return self.parse.verdict


def ground(
    parse: ParseResult, vocabulary: set[tuple[str, str]]
) -> tuple[ParseResult, list[tuple[str, str]]]:
    """Convert a parsed verdict into a failure if any citation is ungrounded;
    also return the offending pairs."""
    if not parse.ok:
        return parse, []
    missing = [
        (e.signal, e.region)
        for e in parse.verdict.evidence
        if (e.signal, e.region) not in vocabulary
    ]
    if missing:
        return (
            ParseResult(
                failure=ParseFailure.UNGROUNDED_EVIDENCE,
                detail=", ".join(f"{s}[{r}]" for s, r in missing),
            ),
            missing,
        )
    return parse, []


def reason_about(
    sha256: str,
    image: Image.Image,
    backend: VlmBackend,
    feature_dir: Path = FEATURE_DIR,
) -> GroundedResult:
    features = load_features([sha256], feature_dir)
    prompt = build_prompt(features)
    vocabulary = set(zip(features.signal, features.region, strict=True))
    reply = backend.generate(image, prompt)
    parse = parse_verdict(reply.text)
    grounded, missing = ground(parse, vocabulary)
    cited = len(parse.verdict.evidence) if parse.ok else 0
    rate = (cited - len(missing)) / cited if cited else 0.0
    return GroundedResult(
        sha256=sha256,
        parse=grounded,
        grounding_rate=rate,
        ungrounded=missing,
        prompt_tokens_estimate=estimate_tokens(prompt),
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )
