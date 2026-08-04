"""Resumable batch inference with per-call cost accounting.

Verdicts persist incrementally as JSONL keyed by asset hash; a restarted
run skips answered assets, so API failures and interruptions cost nothing.
The test split flows only through the harness; every other split comes
from the gated loader directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from PIL import Image

from provenance_lens.data.access import load_split
from provenance_lens.reasoner.client import CONFIG, load_backend
from provenance_lens.reasoner.grounding import reason_about

VERDICT_DIR = Path("data/verdicts")
RAW_ROOT = Path("data/raw")
# USD per million tokens; keyed by model, input and output
PRICES = {
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}
MAX_RETRIES = 4


def _price(model: str, input_tokens: int, output_tokens: int) -> float:
    table = PRICES.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * table["input"] + output_tokens * table["output"]) / 1e6


def _answered(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {
        json.loads(line)["sha256"] for line in out_path.read_text().splitlines() if line.strip()
    }


def run_split(
    split: str,
    out_path: Path | None = None,
    limit: int | None = None,
    backend=None,
    frame=None,
    raw_root: Path = RAW_ROOT,
    feature_dir=None,
) -> dict:
    if frame is None:
        if split == "test":
            raise ValueError(
                "test-split inference enters through the eval harness "
                "(provenance_lens.eval.harness.reason_on_test), never directly"
            )
        frame = load_split(split)
    frame = frame.sort_values("sha256")
    if limit:
        frame = frame.head(limit)
    out_path = out_path or (VERDICT_DIR / f"{split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = backend or load_backend()
    config = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
    model = config.get("anthropic", {}).get("model", "unknown")
    done = _answered(out_path)
    report = {
        "split": split,
        "total": len(frame),
        "skipped": len(done),
        "calls": 0,
        "parse_failures": 0,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    with out_path.open("a") as sink:
        for row in frame.itertuples(index=False):
            if row.sha256 in done:
                continue
            with Image.open(raw_root / row.path) as image:
                image.load()
                result = _call_with_retries(row.sha256, image, backend, feature_dir)
            record = {
                "sha256": row.sha256,
                "ok": result.ok,
                "label": result.verdict.label if result.ok else None,
                "confidence": result.verdict.confidence if result.ok else None,
                "evidence": ([e.__dict__ for e in result.verdict.evidence] if result.ok else []),
                "failure": str(result.parse.failure) if result.parse.failure else None,
                "grounding_rate": result.grounding_rate,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": _price(model, result.input_tokens, result.output_tokens),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            report["calls"] += 1
            report["parse_failures"] += 0 if result.ok else 1
            report["cost_usd"] += record["cost_usd"]
            report["input_tokens"] += record["input_tokens"]
            report["output_tokens"] += record["output_tokens"]
            if report["calls"] % 100 == 0:
                print(json.dumps(report), flush=True)
    (out_path.with_suffix(".report.json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _call_with_retries(sha, image, backend, feature_dir):
    kwargs = {} if feature_dir is None else {"feature_dir": feature_dir}
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return reason_about(sha, image, backend, **kwargs)
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"retry {attempt + 1} for {sha[:12]}: {error}", flush=True)
            time.sleep(delay)
            delay *= 2


def main(argv=None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    split = args[0] if args else "val"
    limit = int(args[1]) if len(args) > 1 else None
    report = run_split(split, limit=limit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
