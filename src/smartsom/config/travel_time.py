"""Resolve a complete matrix from factory geometry or an explicit manual table."""

from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from smartsom.config.codec import primitive, read_model
from smartsom.config.models import StrictModel
from smartsom.domain.factory_design import occupied_cells
from smartsom.domain.travel_time import TravelTimeMatrix


class MatrixFile(StrictModel):
    schema_id: Literal["smartsom.travel-time-matrix/v1"] = Field(alias="schema")
    source: Literal["auto", "manual"] = "auto"
    default_ticks: int | None = Field(default=None, ge=0)
    overrides: dict[str, dict[str, int | None]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_entries(self):
        if self.source == "auto" and self.default_ticks is not None:
            raise ValueError("default_ticks applies only to a manual matrix")
        if any(
            v is not None and v < 0
            for row in self.overrides.values()
            for v in row.values()
        ):
            raise ValueError("travel times cannot be negative")
        return self


class TransportSettings(StrictModel):
    mode: Literal["grid", "travel_time_matrix"] = "grid"
    matrix: str | None = None
    resolved: MatrixFile | None = None

    @model_validator(mode="after")
    def mode_config(self):
        if self.mode == "grid" and (self.matrix or self.resolved):
            raise ValueError("grid transport cannot specify a travel matrix")
        if self.mode == "travel_time_matrix" and not (self.matrix or self.resolved):
            raise ValueError("matrix transport requires a matrix file")
        return self


def freeze_transport(settings, scenario_path):
    """Read a reference once; materialize and resume consume the inline recipe."""
    transport = settings.transport
    if transport.matrix and transport.resolved is None:
        matrix, _ = read_model(
            Path(scenario_path).parent / transport.matrix, MatrixFile
        )
        transport = transport.model_copy(update={"resolved": matrix})
        settings = settings.model_copy(update={"transport": transport})
    return settings


def materialize_matrix(factory, settings):
    if settings.mode == "grid":
        return None
    spec = settings.resolved
    if spec is None:
        raise ValueError("travel matrix reference was not frozen")
    points = {p.port_id: (p.cell.x, p.cell.y) for p in factory.ports}
    points.update(
        {
            "initial:" + a.agv_id: (a.initial_cell.x, a.initial_cell.y)
            for a in factory.agvs
        }
    )
    if any(
        s not in points or any(t not in points for t in row)
        for s, row in spec.overrides.items()
    ):
        raise ValueError("travel matrix overrides reference an unknown point")
    solids = {(c.x, c.y) for c in factory.grid.blocked_cells}
    for field in (
        "machines",
        "buffers",
        "inspection_stations",
        "scrap_bins",
        "chargers",
    ):
        for resource in getattr(factory, field):
            solids.update((c.x, c.y) for c in occupied_cells(resource.footprint))
    edges = []
    for source, start in sorted(points.items()):
        distances = {start: 0}
        queue = deque([start])
        if spec.source == "auto":
            while queue:
                x, y = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cell = (x + dx, y + dy)
                    if (
                        0 <= cell[0] < factory.grid.width
                        and 0 <= cell[1] < factory.grid.height
                        and cell not in solids
                        and cell not in distances
                    ):
                        distances[cell] = distances[(x, y)] + 1
                        queue.append(cell)
        for target, cell in sorted(points.items()):
            row = spec.overrides.get(source, {})
            if target in row:
                value = row[target]
            elif source == target:
                value = 0
            elif spec.source == "auto":
                value = distances.get(cell)
            elif spec.default_ticks is not None:
                value = spec.default_ticks
            else:
                raise ValueError(f"missing travel time {source} -> {target}")
            edges.append((source, target, value))
    return TravelTimeMatrix(
        tuple((p, *cell) for p, cell in sorted(points.items())),
        tuple(edges),
        spec.source,
    )


def matrix_summary(matrix):
    if matrix is None:
        return {"mode": "grid"}
    from smartsom.config.codec import digest

    finite = [v for _, _, v in matrix.times if v is not None]
    return {
        "mode": "travel_time_matrix",
        "source": matrix.source,
        "points": len(matrix.points),
        "sha256": digest(primitive(matrix)),
        "max_ticks": max(finite),
        "unreachable_pairs": len(matrix.times) - len(finite),
    }
