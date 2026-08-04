"""Detection head and training loop.

An MLP over the standardized baseline vector, trained on the train split
with per-epoch validation metrics logged to ``runs/<name>/``. Data flows
exclusively through the gated loader; the test split is unreachable from
here by construction (#23).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from provenance_lens.baseline.features import Scaler, assemble
from provenance_lens.data.access import load_split

RUNS = Path("runs")
CONFIG = Path("configs/baseline.yaml")


class Head(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers += [
                torch.nn.Linear(previous, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.net(batch).squeeze(-1)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for positive in (0, 1):
        tp = ((predictions == positive) & (labels == positive)).sum()
        fp = ((predictions == positive) & (labels != positive)).sum()
        fn = ((predictions != positive) & (labels == positive)).sum()
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _split_tensors(split: str, scaler: Scaler | None, limit: int | None, **paths):
    frame = load_split(split)
    if limit:
        frame = frame.sort_values("sha256").head(limit)
    labels = (frame.label == "manipulated").to_numpy().astype(np.float32)
    matrix = assemble(frame.sha256.tolist(), **paths).loc[frame.sha256].to_numpy()
    if scaler is None:
        scaler = Scaler.fit(matrix)
    return torch.from_numpy(scaler.transform(matrix)).float(), torch.from_numpy(labels), scaler


def train_head(
    learning_rate: float,
    config: dict,
    run_name: str,
    limit: int | None = None,
    max_epochs: int | None = None,
    **paths,
) -> dict:
    torch.manual_seed(config["train"]["seed"])
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, scaler = _split_tensors("train", None, limit, **paths)
    x_val, y_val, _ = _split_tensors("val", scaler, limit and max(1, limit // 4), **paths)
    scaler.save(run_dir / "scaler.json")

    head = Head(x_train.shape[1], config["head"]["hidden"], config["head"]["dropout"])
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["train"]["seed"]),
    )
    epochs = max_epochs or config["train"]["max_epochs"]
    patience = config["train"]["patience"]
    best = {"val_macro_f1": -1.0, "epoch": -1}
    history = []
    stale = 0
    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(batch_x)
        head.eval()
        with torch.no_grad():
            val_logits = head(x_val).numpy()
        val_pred = (val_logits > 0).astype(int)
        val_labels = y_val.numpy().astype(int)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(x_train),
            "val_accuracy": float((val_pred == val_labels).mean()),
            "val_macro_f1": macro_f1(val_labels, val_pred),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["val_macro_f1"] > best["val_macro_f1"]:
            best = record
            stale = 0
            torch.save(head.state_dict(), run_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    (run_dir / "config.json").write_text(
        json.dumps({"learning_rate": learning_rate, "config": config, "limit": limit}, indent=2)
        + "\n"
    )
    (run_dir / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n")
    (run_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    return best


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    config = yaml.safe_load(CONFIG.read_text())
    smoke = "--smoke" in args
    limit = 2000 if smoke else None
    epochs = 2 if smoke else None
    name = f"smoke-{int(time.time())}" if smoke else f"train-{int(time.time())}"
    best = train_head(
        config["sweep"]["learning_rates"][0], config, name, limit=limit, max_epochs=epochs
    )
    print(json.dumps({"run": name, "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
