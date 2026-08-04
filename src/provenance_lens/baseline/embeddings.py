"""Frozen backbone embeddings, computed once and cached by asset hash.

The baseline head trains over cached vectors, never over pixels, so the
backbone runs exactly once per asset. The backbone is loaded frozen (eval
mode, gradients disabled) and its identity and weight version are recorded
in the cache; changing either changes the cache directory, which invalidates
everything derived from it.

Cache layout: ``data/embeddings/<backbone>-<weights>/part-<n>.parquet``,
each part holding ``sha256`` plus a fixed-length float32 ``embedding`` list
column. Parts are append-only, so resuming never rewrites existing work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

BACKBONE = "resnet50"
WEIGHTS = "IMAGENET1K_V2"
EMBED_DIM = 2048
BATCH = 64
EMBED_ROOT = Path("data/embeddings")
RAW_ROOT = Path("data/raw")


def build_backbone() -> tuple[torch.nn.Module, object]:
    """The frozen backbone and its preprocessing transform."""
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights[WEIGHTS]
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def cache_dir(root: Path = EMBED_ROOT) -> Path:
    return root / f"{BACKBONE}-{WEIGHTS}"


def _existing_shas(directory: Path) -> set[str]:
    shas: set[str] = set()
    for part in sorted(directory.glob("part-*.parquet")):
        shas.update(pd.read_parquet(part, columns=["sha256"]).sha256)
    return shas


@torch.no_grad()
def embed_batch(model, transform, images: list[Image.Image]) -> np.ndarray:
    tensors = torch.stack([transform(image.convert("RGB")) for image in images])
    return model(tensors).cpu().numpy().astype(np.float32)


def build_cache(
    manifest_path: Path,
    raw_root: Path = RAW_ROOT,
    root: Path = EMBED_ROOT,
    model_and_transform: tuple | None = None,
    batch_size: int = BATCH,
) -> dict:
    manifest = pd.read_parquet(manifest_path, columns=["sha256", "path"])
    directory = cache_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    done = _existing_shas(directory)
    todo = manifest[~manifest.sha256.isin(done)].reset_index(drop=True)
    report = {"assets": len(manifest), "skipped": len(done), "computed": len(todo)}
    if todo.empty:
        (directory / "meta.json").write_text(
            json.dumps({"backbone": BACKBONE, "weights": WEIGHTS, "dim": EMBED_DIM}) + "\n"
        )
        return report
    model, transform = model_and_transform or build_backbone()
    part_index = len(list(directory.glob("part-*.parquet")))
    buffer_shas: list[str] = []
    buffer_vectors: list[np.ndarray] = []
    torch.set_num_threads(max(1, (torch.get_num_threads())))

    def flush():
        nonlocal part_index, buffer_shas, buffer_vectors
        if not buffer_shas:
            return
        frame = pd.DataFrame(
            {
                "sha256": buffer_shas,
                "embedding": [vector.tolist() for vector in np.concatenate(buffer_vectors)],
            }
        )
        frame.to_parquet(directory / f"part-{part_index:04d}.parquet", index=False)
        part_index += 1
        buffer_shas, buffer_vectors = [], []

    for start in range(0, len(todo), batch_size):
        chunk = todo.iloc[start : start + batch_size]
        images = [Image.open(raw_root / p) for p in chunk.path]
        vectors = embed_batch(model, transform, images)
        for image in images:
            image.close()
        buffer_shas.extend(chunk.sha256.tolist())
        buffer_vectors.append(vectors)
        if len(buffer_shas) >= 8192:
            flush()
        if (start // batch_size) % 50 == 0:
            print(f"  embeddings: {start + len(chunk)}/{len(todo)}", flush=True)
    flush()
    (directory / "meta.json").write_text(
        json.dumps({"backbone": BACKBONE, "weights": WEIGHTS, "dim": EMBED_DIM}) + "\n"
    )
    return report


def load_embeddings(shas: list[str] | None = None, root: Path = EMBED_ROOT) -> pd.DataFrame:
    directory = cache_dir(root)
    frames = []
    for part in sorted(directory.glob("part-*.parquet")):
        frame = pd.read_parquet(part)
        if shas is not None:
            frame = frame[frame.sha256.isin(shas)]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["sha256", "embedding"])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    report = build_cache(Path("data/manifest.parquet"))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
