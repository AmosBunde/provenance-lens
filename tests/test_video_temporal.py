"""Temporal aggregation tests: majority, flips, drift, failure handling."""

from provenance_lens.video.temporal import ClipVerdict, FrameVerdict, aggregate


def _frame(i, label, confidence=0.9, ok=True, cited=()):
    return FrameVerdict(
        frame_index=i,
        ok=ok,
        label=label if ok else None,
        confidence=confidence if ok else 0.0,
        cited=frozenset(cited),
    )


def test_stable_frames_aggregate_with_high_confidence():
    frames = [_frame(i, "manipulated") for i in range(6)]
    clip = aggregate(frames)
    assert clip.label == "manipulated"
    assert clip.confidence > 0.85
    assert clip.flip_rate == 0.0
    assert clip.parsed_frames == 6 and clip.failed_frames == 0


def test_alternating_verdicts_raise_flip_rate_and_cut_confidence():
    frames = [_frame(i, "manipulated" if i % 2 else "authentic") for i in range(6)]
    clip = aggregate(frames)
    assert clip.flip_rate == 1.0
    stable = aggregate([_frame(i, clip.label) for i in range(6)])
    assert clip.confidence < stable.confidence


def test_signal_drift_measures_cited_overlap():
    same = frozenset({("jpeg_ghost_depth", "r1c1")})
    other = frozenset({("noise_region_mismatch", "r2c2")})
    stable = aggregate([_frame(i, "manipulated", cited=same) for i in range(4)])
    drifting = aggregate(
        [_frame(i, "manipulated", cited=same if i % 2 else other) for i in range(4)]
    )
    assert stable.signal_drift == 0.0
    assert drifting.signal_drift == 1.0


def test_confidence_weighted_majority():
    frames = [
        _frame(0, "authentic", 0.95),
        _frame(1, "authentic", 0.9),
        _frame(2, "manipulated", 0.6),
    ]
    clip = aggregate(frames)
    assert clip.label == "authentic"


def test_all_failed_frames_default_conservatively():
    frames = [_frame(i, None, ok=False) for i in range(3)]
    clip = aggregate(frames)
    assert isinstance(clip, ClipVerdict)
    assert clip.parsed_frames == 0 and clip.failed_frames == 3
    assert clip.confidence == 0.5


def test_failures_are_excluded_from_flip_math():
    frames = [
        _frame(0, "manipulated"),
        _frame(1, None, ok=False),
        _frame(2, "manipulated"),
    ]
    clip = aggregate(frames)
    assert clip.flip_rate == 0.0
    assert clip.failed_frames == 1
