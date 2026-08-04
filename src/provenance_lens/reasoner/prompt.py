"""Prompt template: forensic measurements rendered as structured context.

Every signal line is rendered verbatim from the feature store for the
asset; the template cannot invent names, which is what makes citation
grounding meaningful. Rendering is deterministic (sorted by family, signal,
region) and pinned by golden-file tests.
"""

from __future__ import annotations

import pandas as pd

FAMILIES = {
    "jpeg_ghost": "Compression: JPEG ghost analysis",
    "blocking": "Compression: blocking grid",
    "noise": "Noise residuals",
    "edge": "Edge structure",
    "lighting": "Lighting direction",
}

INSTRUCTION = """\
You are a forensic image analyst. Decide whether the image is authentic or
manipulated, using the image itself and the measurements above. Measurements
are named signals over a 3x3 region grid (r0c0 top-left through r2c2
bottom-right) plus global; each carries its direction of suspicion.

Answer with exactly one JSON object and nothing else:

{"label": "authentic" or "manipulated",
 "confidence": a number between 0.0 and 1.0,
 "evidence": [{"signal": "<signal name from the measurements>",
               "region": "<its region>",
               "direction": "supports_manipulated" or "supports_authentic"}]}

Cite only signals and regions that appear in the measurements. Any cited
signal that does not appear above invalidates the whole answer."""


def _family(signal_name: str) -> str:
    for prefix, title in FAMILIES.items():
        if signal_name.startswith(prefix):
            return title
    return "Other signals"


def render_measurements(features: pd.DataFrame) -> str:
    """Render store rows (one asset) as grouped, sorted measurement lines."""
    lines: dict[str, list[str]] = {}
    frame = features.sort_values(["signal", "region"])
    for row in frame.itertuples(index=False):
        family = _family(row.signal)
        lines.setdefault(family, []).append(
            f"  {row.signal} [{row.region}] = {row.value:.4f} ({row.direction})"
        )
    blocks = []
    for family in sorted(lines):
        blocks.append(family + ":\n" + "\n".join(lines[family]))
    return "\n\n".join(blocks)


def build_prompt(features: pd.DataFrame) -> str:
    if features.empty:
        raise ValueError("no measurements for this asset; refuse to prompt blind")
    return (
        "FORENSIC MEASUREMENTS\n\n" + render_measurements(features) + "\n\nTASK\n\n" + INSTRUCTION
    )


def estimate_tokens(prompt: str) -> int:
    """Rough token estimate for cost planning (about 4 characters per token)."""
    return len(prompt) // 4
