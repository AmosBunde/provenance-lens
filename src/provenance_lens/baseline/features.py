"""Per-asset baseline feature vectors: embedding plus structured signals.

The vector is the frozen backbone embedding concatenated with the pivoted
forensic signals from the feature store. Standardization statistics are fit
on the train split only and applied everywhere else; fitting them on
anything wider would leak validation or test statistics into training.

Two EDA-mandated exclusions are enforced structurally rather than by
filtering: file size and JPEG quality factor are simply never part of the
store vocabulary, and a unit test asserts they cannot appear in the
assembled columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_lens.baseline.embeddings import load_embeddings
from provenance_lens.forensics.store import load_features

FORBIDDEN_FEATURES = ("file_bytes", "file_size", "quality_factor", "qf")


def structured_matrix(shas: list[str], feature_dir=None) -> pd.DataFrame:
    """Pivot store rows into one row per asset, one column per signal_region."""
    kwargs = {} if feature_dir is None else {"feature_dir": feature_dir}
    features = load_features(shas, **kwargs)
    features["column"] = features.signal + "__" + features.region
    matrix = features.pivot_table(index="sha256", columns="column", values="value", aggfunc="first")
    for needle in FORBIDDEN_FEATURES:
        bad = [c for c in matrix.columns if needle in c]
        if bad:
            raise ValueError(f"forbidden artifact features present: {bad}")
    return matrix.sort_index(axis=1)


def assemble(shas: list[str], embed_root=None, feature_dir=None) -> pd.DataFrame:
    kwargs = {} if embed_root is None else {"root": embed_root}
    embeddings = load_embeddings(shas, **kwargs).set_index("sha256")
    structured = structured_matrix(shas, feature_dir)
    joined = embeddings.join(structured, how="inner")
    missing = set(shas) - set(joined.index)
    if missing:
        raise ValueError(f"{len(missing)} assets lack embeddings or features")
    vectors = np.stack(
        [
            np.concatenate([np.asarray(e, dtype=np.float32), row.astype(np.float32)])
            for e, row in zip(
                joined.embedding, joined.drop(columns="embedding").to_numpy(), strict=True
            )
        ]
    )
    frame = pd.DataFrame(vectors, index=joined.index)
    frame.columns = [f"f{i}" for i in range(vectors.shape[1])]
    return frame


class Scaler:
    """Train-split standardization, persisted so every consumer shares it."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Scaler:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}) + "\n")

    @classmethod
    def load(cls, path: Path) -> Scaler:
        payload = json.loads(path.read_text())
        return cls(np.asarray(payload["mean"]), np.asarray(payload["std"]))
