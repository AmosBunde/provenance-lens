"""Learning rate sweep with early stopping; freezes the best checkpoint.

Every configuration trains with the shared loop from ``baseline.train``;
the winner by validation macro F1 is copied to
``data/baseline/best_checkpoint.pt`` with its SHA-256 and provenance in
``data/baseline/best_checkpoint.json``. The test split plays no part here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from provenance_lens.baseline.train import CONFIG, RUNS, train_head

FROZEN_DIR = Path("data/baseline")


def run_sweep(limit: int | None = None, max_epochs: int | None = None) -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    stamp = int(time.time())
    results = []
    for learning_rate in config["sweep"]["learning_rates"]:
        run_name = f"sweep-{stamp}-lr{learning_rate:g}"
        best = train_head(learning_rate, config, run_name, limit=limit, max_epochs=max_epochs)
        results.append({"learning_rate": learning_rate, "run": run_name, **best})
        print(json.dumps(results[-1]), flush=True)
    winner = max(results, key=lambda r: r["val_macro_f1"])
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    source = RUNS / winner["run"]
    shutil.copy(source / "best.pt", FROZEN_DIR / "best_checkpoint.pt")
    shutil.copy(source / "scaler.json", FROZEN_DIR / "scaler.json")
    checkpoint_sha = hashlib.sha256((FROZEN_DIR / "best_checkpoint.pt").read_bytes()).hexdigest()
    summary = {
        "winner": winner,
        "all": results,
        "checkpoint_sha256": checkpoint_sha,
        "selection_rule": "max validation macro F1 across the learning rate grid",
    }
    (FROZEN_DIR / "best_checkpoint.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    summary = run_sweep()
    print(json.dumps(summary["winner"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
