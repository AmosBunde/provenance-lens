"""The signal naming contract shared by every forensic extractor.

Signals are the vocabulary of the whole project: the feature store persists
them, the baseline consumes their values, the prompt template renders them,
and the grounding check validates that every citation names a signal and
region that actually exist for the asset. An extractor therefore emits only
:class:`Signal` instances with a stable snake_case name, a region from the
fixed grid, a float value, and a documented direction of suspicion.

Regions are the cells of a fixed 3x3 grid, ``r0c0`` through ``r2c2`` (row
first, top left origin), plus ``global`` for whole-image measurements, so
every evidence citation is a checkable string.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

GRID = 3
REGIONS = tuple(f"r{r}c{c}" for r in range(GRID) for c in range(GRID))
GLOBAL = "global"
ALL_REGIONS = (*REGIONS, GLOBAL)


class Direction(StrEnum):
    """How a value relates to suspicion of manipulation."""

    HIGHER_SUSPICIOUS = "higher_is_suspicious"
    LOWER_SUSPICIOUS = "lower_is_suspicious"
    CONTEXT = "context"


@dataclass(frozen=True)
class Signal:
    name: str
    region: str
    value: float
    direction: Direction

    def __post_init__(self):
        if self.region not in ALL_REGIONS:
            raise ValueError(f"unknown region {self.region!r}")
        if not self.name.replace("_", "").isalnum() or self.name != self.name.lower():
            raise ValueError(f"signal name must be snake_case: {self.name!r}")


def region_boxes(height: int, width: int) -> dict[str, tuple[slice, slice]]:
    """Map each grid region to its (rows, cols) slices for an image shape."""
    boxes = {}
    row_edges = [round(i * height / GRID) for i in range(GRID + 1)]
    col_edges = [round(i * width / GRID) for i in range(GRID + 1)]
    for r in range(GRID):
        for c in range(GRID):
            boxes[f"r{r}c{c}"] = (
                slice(row_edges[r], row_edges[r + 1]),
                slice(col_edges[c], col_edges[c + 1]),
            )
    return boxes
