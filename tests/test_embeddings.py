"""Embedding cache tests with a stub backbone (no weight downloads in CI)."""

import hashlib

import numpy as np
import pandas as pd
import torch
from PIL import Image

from provenance_lens.baseline.embeddings import build_cache, load_embeddings


class _StubBackbone(torch.nn.Module):
    """Deterministic tiny stand-in: mean-pools pixels into a fixed vector."""

    def forward(self, batch):
        pooled = batch.mean(dim=(2, 3))
        return pooled.repeat(1, 8)[:, :16]


def _stub_transform(image):
    array = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _corpus(tmp_path, count=5):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(count):
        array = np.clip(rng.normal(128, 40, (32, 32, 3)), 0, 255).astype(np.uint8)
        path = raw / f"img_{i}.png"
        Image.fromarray(array).save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"sha256": sha, "path": f"img_{i}.png"})
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest, raw, rows


def test_cache_builds_and_resumes(tmp_path):
    manifest, raw, rows = _corpus(tmp_path)
    stub = (_StubBackbone(), _stub_transform)
    root = tmp_path / "emb"
    report = build_cache(manifest, raw, root, model_and_transform=stub, batch_size=2)
    assert report["computed"] == 5
    again = build_cache(manifest, raw, root, model_and_transform=stub, batch_size=2)
    assert again["computed"] == 0 and again["skipped"] == 5
    frame = load_embeddings(root=root)
    assert len(frame) == 5
    assert len(frame.embedding.iloc[0]) == 16


def test_embeddings_are_deterministic(tmp_path):
    manifest, raw, rows = _corpus(tmp_path)
    stub = (_StubBackbone(), _stub_transform)
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_cache(manifest, raw, a, model_and_transform=stub, batch_size=3)
    build_cache(manifest, raw, b, model_and_transform=stub, batch_size=3)
    fa = load_embeddings(root=a).sort_values("sha256").reset_index(drop=True)
    fb = load_embeddings(root=b).sort_values("sha256").reset_index(drop=True)
    assert all(np.allclose(x, y) for x, y in zip(fa.embedding, fb.embedding, strict=True))


def test_incremental_asset_lands_in_new_part(tmp_path):
    manifest, raw, rows = _corpus(tmp_path)
    stub = (_StubBackbone(), _stub_transform)
    root = tmp_path / "emb"
    build_cache(manifest, raw, root, model_and_transform=stub, batch_size=2)
    array = np.clip(np.random.default_rng(9).normal(100, 30, (32, 32, 3)), 0, 255).astype(np.uint8)
    path = raw / "img_new.png"
    Image.fromarray(array).save(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_parquet(manifest)
    frame = pd.concat(
        [frame, pd.DataFrame([{"sha256": sha, "path": "img_new.png"}])], ignore_index=True
    )
    frame.to_parquet(manifest, index=False)
    report = build_cache(manifest, raw, root, model_and_transform=stub, batch_size=2)
    assert report["computed"] == 1
    assert sha in set(load_embeddings(root=root).sha256)
