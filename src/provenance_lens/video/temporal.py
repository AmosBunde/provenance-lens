"""Clip-level aggregation of per-frame verdicts with temporal diagnostics.

A manipulated clip should look manipulated consistently; verdict flips
between adjacent sampled frames and drift in cited-signal overlap are
evidence about reliability, so they are surfaced as diagnostics next to the
aggregated verdict rather than hidden inside it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameVerdict:
    frame_index: int
    ok: bool
    label: str | None
    confidence: float
    cited: frozenset[tuple[str, str]] = frozenset()


@dataclass
class ClipVerdict:
    label: str
    confidence: float
    flip_rate: float
    signal_drift: float
    parsed_frames: int
    failed_frames: int


def aggregate(frames: list[FrameVerdict]) -> ClipVerdict:
    ordered = sorted(frames, key=lambda f: f.frame_index)
    parsed = [f for f in ordered if f.ok and f.label is not None]
    failed = len(ordered) - len(parsed)
    if not parsed:
        return ClipVerdict("manipulated", 0.5, 0.0, 0.0, 0, failed)

    weight = {"authentic": 0.0, "manipulated": 0.0}
    for frame in parsed:
        weight[frame.label] += frame.confidence
    total = weight["authentic"] + weight["manipulated"]
    label = "manipulated" if weight["manipulated"] >= weight["authentic"] else "authentic"
    confidence = weight[label] / total if total else 0.5

    flips = sum(1 for a, b in zip(parsed, parsed[1:], strict=False) if a.label != b.label)
    flip_rate = flips / (len(parsed) - 1) if len(parsed) > 1 else 0.0
    # penalize confidence by observed instability
    confidence = confidence * (1.0 - flip_rate / 2.0)

    drifts = []
    for a, b in zip(parsed, parsed[1:], strict=False):
        union = a.cited | b.cited
        if union:
            drifts.append(1.0 - len(a.cited & b.cited) / len(union))
    signal_drift = sum(drifts) / len(drifts) if drifts else 0.0
    return ClipVerdict(
        label=label,
        confidence=float(confidence),
        flip_rate=float(flip_rate),
        signal_drift=float(signal_drift),
        parsed_frames=len(parsed),
        failed_frames=failed,
    )
