"""Per-manipulation-type breakdown and confusion matrices.

Types are nested in sources in this corpus (ai_generated exists only in
cifake, copy_move only in MICC), so per-type numbers are really per-source
numbers; every artifact this module emits carries that confound note
verbatim rather than leaving it to a reader's memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_lens.eval.scorer import Track, bootstrap_metrics

CONFOUND_NOTE = (
    "Manipulation types are nested in sources (ai_generated only in cifake, "
    "copy_move only in MICC); per-type performance is inseparable from "
    "per-source performance in this corpus."
)


def per_type_metrics(labels: np.ndarray, types: pd.Series, track: Track) -> dict:
    out = {"note": CONFOUND_NOTE, "types": {}}
    for type_name in sorted(types.unique()):
        mask = (types == type_name).to_numpy()
        sliced = Track(
            name=track.name,
            predictions=track.predictions[mask],
            scores=track.scores[mask],
            failure_mask=track.failure_mask[mask],
            failure_taxonomy={},
        )
        metrics = bootstrap_metrics(labels[mask], sliced)
        metrics["n"] = int(mask.sum())
        out["types"][type_name] = metrics
    return out


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "true_authentic_pred_authentic": int(((labels == 0) & (predictions == 0)).sum()),
        "true_authentic_pred_manipulated": int(((labels == 0) & (predictions == 1)).sum()),
        "true_manipulated_pred_authentic": int(((labels == 1) & (predictions == 0)).sum()),
        "true_manipulated_pred_manipulated": int(((labels == 1) & (predictions == 1)).sum()),
    }


def plot_confusion(matrix: dict, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array(
        [
            [matrix["true_authentic_pred_authentic"], matrix["true_authentic_pred_manipulated"]],
            [
                matrix["true_manipulated_pred_authentic"],
                matrix["true_manipulated_pred_manipulated"],
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(grid, cmap="Blues")
    for (i, j), value in np.ndenumerate(grid):
        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            color="black" if value < grid.max() * 0.6 else "white",
        )
    ax.set_xticks([0, 1], ["pred authentic", "pred manipulated"])
    ax.set_yticks([0, 1], ["true authentic", "true manipulated"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_breakdown(
    labels: np.ndarray,
    types: pd.Series,
    track: Track,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    breakdown = per_type_metrics(labels, types, track)
    breakdown["confusion"] = confusion_matrix(labels, track.predictions)
    plot_confusion(
        breakdown["confusion"],
        f"{track.name}: test confusion",
        out_dir / f"confusion_{track.name}.png",
    )
    (out_dir / f"breakdown_{track.name}.json").write_text(json.dumps(breakdown, indent=2) + "\n")
    return breakdown
