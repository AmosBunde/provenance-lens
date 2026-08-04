"""Head training tests on synthetic tensors and fixture stores."""

import numpy as np
import torch

from provenance_lens.baseline.features import FORBIDDEN_FEATURES, Scaler
from provenance_lens.baseline.train import Head, macro_f1


def test_scaler_fits_on_given_matrix_only():
    train = np.array([[0.0, 10.0], [2.0, 30.0]])
    scaler = Scaler.fit(train)
    other = np.array([[4.0, 50.0]])
    transformed = scaler.transform(other)
    assert np.allclose(scaler.mean, [1.0, 20.0])
    assert transformed[0][0] > 2.9  # standardized against train stats, not its own


def test_scaler_roundtrip(tmp_path):
    scaler = Scaler.fit(np.random.default_rng(0).normal(3, 2, (50, 4)))
    scaler.save(tmp_path / "s.json")
    loaded = Scaler.load(tmp_path / "s.json")
    assert np.allclose(loaded.mean, scaler.mean)
    assert np.allclose(loaded.std, scaler.std)


def test_forbidden_features_are_named():
    assert "file_bytes" in FORBIDDEN_FEATURES
    assert "quality_factor" in FORBIDDEN_FEATURES


def test_head_learns_a_separable_problem():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (512, 8)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    head = Head(8, [16], dropout=0.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(head(tx), ty)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = ((head(tx) > 0).float() == ty).float().mean()
    assert accuracy > 0.95


def test_macro_f1_matches_manual_case():
    labels = np.array([0, 0, 1, 1, 1])
    predictions = np.array([0, 1, 1, 1, 0])
    # class 0: tp=1 fp=1 fn=1 -> f1=0.5; class 1: tp=2 fp=1 fn=1 -> f1=2/3
    assert abs(macro_f1(labels, predictions) - (0.5 + 2 / 3) / 2) < 1e-9
