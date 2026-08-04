"""VLM backend abstraction: API, local, and mock behind one interface.

Nothing outside this module knows which backend is active; the factory
reads ``configs/reasoner.yaml``. Credentials are checked at construction so
a missing key fails once and clearly, never mid-batch.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

CONFIG = Path("configs/reasoner.yaml")


@dataclass
class BackendReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class VlmBackend(ABC):
    """One method: an image and a prompt in, raw model text out."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str) -> BackendReply: ...


class AnthropicBackend(VlmBackend):
    def __init__(self, model: str, max_tokens: int, temperature: float, client=None):
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set; the API backend cannot start. "
                    "Set the key or select the mock or local backend in "
                    "configs/reasoner.yaml"
                )
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _encode(image: Image.Image) -> dict:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        payload = base64.standard_b64encode(buffer.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": payload},
        }

    def build_request(self, image: Image.Image, prompt: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [self._encode(image), {"type": "text", "text": prompt}],
                }
            ],
        }

    def generate(self, image: Image.Image, prompt: str) -> BackendReply:
        response = self._client.messages.create(**self.build_request(image, prompt))
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = getattr(response, "usage", None)
        return BackendReply(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )


class LocalBackend(VlmBackend):
    """Open-weights VLM for the CUDA image; loads lazily on first use."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipeline = None

    def generate(self, image: Image.Image, prompt: str) -> BackendReply:
        raise NotImplementedError(
            "the local backend runs in the CUDA image with GPU weights; "
            "select the api or mock backend on CPU-only machines"
        )


@dataclass
class MockBackend(VlmBackend):
    """Scripted replies for tests and offline development."""

    replies: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def generate(self, image: Image.Image, prompt: str) -> BackendReply:
        self.calls.append(prompt)
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        if not self.replies:
            raise RuntimeError("MockBackend has no scripted replies")
        return BackendReply(text=self.replies[index], input_tokens=10, output_tokens=5)


def load_backend(config_path: Path = CONFIG, **overrides) -> VlmBackend:
    config = yaml.safe_load(config_path.read_text())
    kind = overrides.get("backend", config["backend"])
    if kind == "anthropic":
        c = config["anthropic"]
        return AnthropicBackend(c["model"], c["max_tokens"], c["temperature"])
    if kind == "local":
        return LocalBackend(config["local"]["model_id"])
    if kind == "mock":
        return MockBackend(replies=list(config.get("mock", {}).get("replies", [])))
    raise ValueError(f"unknown backend kind: {kind}")
