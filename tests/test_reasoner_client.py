"""Backend abstraction tests; no network anywhere."""

import numpy as np
import pytest
from PIL import Image

from provenance_lens.reasoner.client import (
    AnthropicBackend,
    MockBackend,
    load_backend,
)


def _image():
    return Image.fromarray(
        np.clip(np.random.default_rng(0).normal(128, 30, (64, 64, 3)), 0, 255).astype("uint8")
    )


def test_factory_returns_mock(tmp_path):
    config = tmp_path / "reasoner.yaml"
    config.write_text('backend: mock\nmock:\n  replies:\n    - "{\\"label\\": \\"authentic\\"}"\n')
    backend = load_backend(config)
    assert isinstance(backend, MockBackend)


def test_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        AnthropicBackend("claude-sonnet-5", 1024, 0.0)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_anthropic_request_construction():
    backend = AnthropicBackend("claude-sonnet-5", 512, 0.0, client=object())
    request = backend.build_request(_image(), "analyze this")
    assert request["model"] == "claude-sonnet-5"
    assert request["max_tokens"] == 512
    content = request["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1] == {"type": "text", "text": "analyze this"}


def test_mock_scripts_replies_in_order():
    backend = MockBackend(replies=["a", "b"])
    image = _image()
    assert backend.generate(image, "p1").text == "a"
    assert backend.generate(image, "p2").text == "b"
    assert backend.generate(image, "p3").text == "b"
    assert backend.calls == ["p1", "p2", "p3"]
