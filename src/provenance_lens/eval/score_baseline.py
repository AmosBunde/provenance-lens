"""Single-shot baseline test scoring. Lives in eval/ because only the
harness may read the test split, and this is the one sanctioned invocation.

The protocol allows the frozen baseline checkpoint exactly one test-set
evaluation. This module loads the checkpoint and scaler frozen by the sweep,
scores the test split through the harness token, writes the predictions and
the results record, and marks the test set closed to the baseline. Running
it again with an existing record refuses unless forced, so the single-shot
rule is enforced by the artifact, not by memory.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from provenance_lens.baseline.features import Scaler, assemble
from provenance_lens.baseline.train import CONFIG, Head, macro_f1
from provenance_lens.eval.harness import load_test_split

FROZEN_DIR = Path("data/baseline")
RESULT_PATH = Path("docs/report/baseline_test.json")
PREDICTIONS_PATH = Path("data/baseline/test_predictions.parquet")


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels.astype(bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score_once(force: bool = False) -> dict:
    if RESULT_PATH.exists() and not force:
        raise RuntimeError(
            f"{RESULT_PATH} exists: the baseline has already used its single "
            "test evaluation and the test set is closed to it"
        )
    config = yaml.safe_load(CONFIG.read_text())
    checkpoint_path = FROZEN_DIR / "best_checkpoint.pt"
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    frozen_meta = json.loads((FROZEN_DIR / "best_checkpoint.json").read_text())
    if frozen_meta["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("checkpoint on disk does not match the frozen record")

    test = load_test_split().sort_values("sha256")
    labels = (test.label == "manipulated").to_numpy().astype(int)
    matrix = assemble(test.sha256.tolist()).loc[test.sha256].to_numpy()
    scaler = Scaler.load(FROZEN_DIR / "scaler.json")
    inputs = torch.from_numpy(scaler.transform(matrix)).float()

    head = Head(inputs.shape[1], config["head"]["hidden"], config["head"]["dropout"])
    head.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    head.eval()
    with torch.no_grad():
        scores = torch.sigmoid(head(inputs)).numpy()
    predictions = (scores > 0.5).astype(int)

    result = {
        "scored_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "checkpoint_sha256": checkpoint_sha,
        "winning_config": frozen_meta["winner"],
        "test_assets": int(len(test)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1(labels, predictions),
        "auroc": rank_auc(scores, labels),
        "protocol_note": (
            "The test set is now closed to the baseline. This record is the "
            "number to beat; any further baseline evaluation on test would "
            "violate protocol."
        ),
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sha256": test.sha256.to_numpy(), "score": scores}).to_parquet(
        PREDICTIONS_PATH, index=False
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    result = score_once(force="--force" in args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
