"""Single-asset verdict service: upload an image, get grounded evidence.

The endpoint runs the four extractors on the uploaded image in memory,
builds the prompt from those fresh signals, calls the configured backend,
and returns the parsed, grounded verdict. The page renders the verdict and
highlights cited regions over the 3x3 grid.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from provenance_lens.forensics import (
    blocking_grid,
    compression_ghosts,
    edges_lighting,
    noise_residuals,
)
from provenance_lens.forensics.store import COLUMNS
from provenance_lens.reasoner.client import load_backend
from provenance_lens.reasoner.grounding import reason_about

app = FastAPI(title="Provenance Lens")
_EXTRACTORS = {
    "compression_ghosts": compression_ghosts.extract,
    "blocking_grid": blocking_grid.extract,
    "noise_residuals": noise_residuals.extract,
    "edges_lighting": edges_lighting.extract,
}
PAGE = Path(__file__).parent / "page.html"


def _ephemeral_store(sha: str, image: Image.Image) -> Path:
    """Extract signals for one uploaded image into a temporary store whose
    shard names match the registry convention load_features scans."""
    directory = Path(tempfile.mkdtemp(prefix="pl-demo-"))
    for name, extract in _EXTRACTORS.items():
        rows = [
            (sha, s.name, s.region, s.value, str(s.direction), name, "1") for s in extract(image)
        ]
        pd.DataFrame(rows, columns=COLUMNS).to_parquet(
            directory / f"{name}-v1.parquet", index=False
        )
    return directory


@app.post("/verdict")
async def verdict(file: Annotated[UploadFile, File()]) -> JSONResponse:
    payload = await file.read()
    sha = hashlib.sha256(payload).hexdigest()
    image = Image.open(io.BytesIO(payload))
    image.load()
    feature_dir = _ephemeral_store(sha, image)
    backend = load_backend()
    result = reason_about(sha, image, backend, feature_dir)
    body = {
        "sha256": sha,
        "ok": result.ok,
        "label": result.verdict.label if result.ok else None,
        "confidence": result.verdict.confidence if result.ok else None,
        "evidence": [e.__dict__ for e in result.verdict.evidence] if result.ok else [],
        "failure": str(result.parse.failure) if result.parse.failure else None,
        "grounding_rate": result.grounding_rate,
    }
    return JSONResponse(body)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE.read_text()
