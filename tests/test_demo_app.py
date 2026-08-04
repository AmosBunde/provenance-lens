"""Demo endpoint tests with the mock backend via the test client."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from provenance_lens.demo import app as demo_module
from provenance_lens.reasoner.client import MockBackend

GOOD = (
    '{"label": "manipulated", "confidence": 0.83, "evidence": '
    '[{"signal": "noise_region_mismatch", "region": "r1c1", '
    '"direction": "supports_manipulated"}]}'
)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(demo_module, "load_backend", lambda: MockBackend(replies=[GOOD]))
    return TestClient(demo_module.app)


def _upload_bytes():
    rng = np.random.default_rng(0)
    image = Image.fromarray(np.clip(rng.normal(128, 40, (96, 96, 3)), 0, 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_verdict_endpoint_returns_grounded_json(client):
    response = client.post("/verdict", files={"file": ("upload.png", _upload_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["label"] == "manipulated"
    assert body["confidence"] == 0.83
    assert body["evidence"][0]["region"] == "r1c1"
    assert body["grounding_rate"] == 1.0
    assert len(body["sha256"]) == 64


def test_ungrounded_citation_reports_failure(client, monkeypatch):
    invented = (
        '{"label": "manipulated", "confidence": 0.9, "evidence": '
        '[{"signal": "invented_signal", "region": "r0c0", '
        '"direction": "supports_manipulated"}]}'
    )
    monkeypatch.setattr(demo_module, "load_backend", lambda: MockBackend(replies=[invented]))
    response = client.post("/verdict", files={"file": ("upload.png", _upload_bytes(), "image/png")})
    body = response.json()
    assert body["ok"] is False
    assert body["failure"] == "ungrounded_evidence"


def test_index_serves_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Provenance Lens" in response.text
    assert "drop an image" in response.text
