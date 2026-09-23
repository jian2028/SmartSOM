"""Factory grid layout and path utilities.

The default layout reproduces the "Current Factory Layout" (Jian, SOM slides
2026-09-15, slide 11) and the obstacle / interaction-point maps on slide 28:
a 12 x 7 grid, 8 machines (M1..M8) on the top and bottom rows with a PRE
buffer on the left and a POST buffer on the right, input/output buffers,
two inspection stations, two disposal stations and four chargers.

Coordinates are ``[x, y]`` with ``x`` = column, ``y`` = row, origin top-left.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

Cell = tuple[int, int]

GRID_WIDTH = 12
GRID_HEIGHT = 7

# Top row machines M1, M3, M5, M7; bottom row M2, M4, M6, M8. The machine
# pair in one column serves the same operation (slide 35).
_MACHINE_COLUMNS = (1, 4, 7, 10)
_OPERATIONS = ("O1", "O2", "O3", "O4")

MODES: dict[str, dict[str, float]] = {
    "SLOW": {"time_scale": 1.25, "defect_rate": 0.05},
    "NORMAL": {"time_scale": 1.00, "defect_rate": 0.10},
    "FAST": {"time_scale": 0.80, "defect_rate": 0.15},
}


def _station(
    sid: str,
    stype: str,
    cell: Cell,
    interaction_points: Iterable[Cell],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": sid,
        "type": stype,
        "cell": list(cell),
        "interaction_points": [list(p) for p in interaction_points],
        **extra,
    }


def default_layout(
    buffer_capacity: int = 4, inspection_slots: int = 4
) -> dict[str, Any]:
    """Return the slide-11 factory layout as a JSON-serialisable dict."""
    stations: list[dict[str, Any]] = []
    for idx, col in enumerate(_MACHINE_COLUMNS):
        op = _OPERATIONS[idx]
        for row, ip_row, num in (
            (0, 1, 2 * idx + 1),
            (GRID_HEIGHT - 1, GRID_HEIGHT - 2, 2 * idx + 2),
        ):
            mid = f"M{num}"
            stations.append(
                _station(
                    f"{mid}.PRE",
                    "pre_buffer",
                    (col - 1, row),
                    [(col - 1, ip_row)],
                    machine=mid,
                    capacity=buffer_capacity,
                )
            )
            stations.append(
                _station(
                    mid,
                    "machine",
                    (col, row),
                    [],
                    operation=op,
                    pre_buffer=f"{mid}.PRE",
                    post_buffer=f"{mid}.POST",
                )
            )
            stations.append(
                _station(
                    f"{mid}.POST",
                    "post_buffer",
                    (col + 1, row),
                    [(col + 1, ip_row)],
                    machine=mid,
                    capacity=buffer_capacity,
                )
            )

    mid_row = GRID_HEIGHT // 2
    stations += [
        _station("IN", "input", (0, mid_row), [(1, mid_row)], capacity=8),
        _station(
            "OUT",
            "output",
            (GRID_WIDTH - 1, mid_row),
            [(GRID_WIDTH - 2, mid_row)],
            capacity=None,
        ),
        _station(
            "D1",
            "disposal",
            (3, mid_row),
            [(3, mid_row - 1), (3, mid_row + 1)],
            capacity=None,
        ),
        _station(
            "Q1",
            "inspection",
            (4, mid_row),
            [(4, mid_row - 1), (4, mid_row + 1)],
            capacity=inspection_slots,
        ),
        _station(
            "Q2",
            "inspection",
            (7, mid_row),
            [(7, mid_row - 1), (7, mid_row + 1)],
            capacity=inspection_slots,
        ),
        _station(
            "D2",
            "disposal",
            (8, mid_row),
            [(8, mid_row - 1), (8, mid_row + 1)],
            capacity=None,
        ),
        _station("C1", "charger", (0, mid_row - 1), [(1, mid_row - 1)]),
        _station("C2", "charger", (0, mid_row + 1), [(1, mid_row + 1)]),
        _station(
            "C3",
            "charger",
            (GRID_WIDTH - 1, mid_row - 1),
            [(GRID_WIDTH - 2, mid_row - 1)],
        ),
        _station(
            "C4",
            "charger",
            (GRID_WIDTH - 1, mid_row + 1),
            [(GRID_WIDTH - 2, mid_row + 1)],
        ),
    ]

    obstacles = sorted({tuple(s["cell"]) for s in stations})
    return {
        "width": GRID_WIDTH,
        "height": GRID_HEIGHT,
        "obstacles": [list(c) for c in obstacles],
        "stations": stations,
        "operations": list(_OPERATIONS),
        "modes": MODES,
    }


class Grid:
    """Road-network helper built from a layout dict."""

    def __init__(self, layout: dict[str, Any]) -> None:
        self.width = int(layout["width"])
        self.height = int(layout["height"])
        self.obstacles: set[Cell] = {tuple(c) for c in layout["obstacles"]}  # type: ignore[misc]
        self._dist_cache: dict[Cell, dict[Cell, int]] = {}

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_road(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        out = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [c for c in out if self.is_road(c)]

    def distances_from(self, source: Cell) -> dict[Cell, int]:
        """BFS distances over road cells, cached per source."""
        cached = self._dist_cache.get(source)
        if cached is not None:
            return cached
        dist = {source: 0}
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for nxt in self.neighbors(cur):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        self._dist_cache[source] = dist
        return dist

    def distance(self, a: Cell, b: Cell) -> int | None:
        return self.distances_from(b).get(a)
