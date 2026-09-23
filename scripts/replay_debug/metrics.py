"""Episode metrics derived from a replay trace (slide 4 "Old Test Result" table).

- Total / passed / defect throughput: jobs delivered to OUTPUT, split by true
  (hidden) defect status. Passing rate = passed / total.
- Total AGV conflicts = movement conflicts (AGV-AGV + AGV-environment)
  + pickup/drop-off conflicts (``service_conflict`` events).
- Loaded-path ratio: mean over loaded trips (pickup -> dropoff) of
  actual cells moved / shortest road distance (slide 10).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .layout import Grid


def loaded_trips(
    manifest: dict[str, Any], frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return one record per completed loaded trip."""
    grid = Grid(manifest["layout"])
    open_trips: dict[str, dict[str, Any]] = {}
    trips: list[dict[str, Any]] = []
    prev_pos: dict[str, tuple[int, int]] = {}

    for frame in frames:
        pos = {a["id"]: tuple(a["pos"]) for a in frame["agvs"]}
        for agv_id, trip in open_trips.items():
            if agv_id in prev_pos and pos.get(agv_id) != prev_pos[agv_id]:
                trip["moves"] += 1
        for event in frame["events"]:
            if event["type"] == "pickup":
                open_trips[event["agv"]] = {
                    "agv": event["agv"],
                    "job": event["job"],
                    "start_tick": frame["tick"],
                    "start": pos[event["agv"]],
                    "moves": 0,
                }
            elif event["type"] == "dropoff" and event["agv"] in open_trips:
                trip = open_trips.pop(event["agv"])
                end = pos[event["agv"]]
                shortest = grid.distance(trip["start"], end)
                trip.update(end_tick=frame["tick"], end=end, shortest=shortest)
                trip["ratio"] = (trip["moves"] / shortest) if shortest else None
                trip["start"], trip["end"] = list(trip["start"]), list(end)
                trips.append(trip)
        prev_pos = pos
    return trips


def compute_metrics(
    manifest: dict[str, Any], frames: list[dict[str, Any]]
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    delivered: list[dict[str, Any]] = []
    shared_return = 0.0
    local_by_type: dict[str, float] = defaultdict(float)
    agent_type = {a["id"]: a["type"] for a in manifest["agents"]}

    for frame in frames:
        shared_return += float(frame["rewards"].get("shared", 0.0))
        for agent_id, value in frame["rewards"].get("local", {}).items():
            local_by_type[agent_type.get(agent_id, "unknown")] += float(value)
        for event in frame["events"]:
            etype = event["type"]
            counts[etype] += 1
            if etype == "movement_conflict":
                counts[f"movement_conflict_{event['kind']}"] += 1
            elif etype == "delivered":
                delivered.append(event)

    total = len(delivered)
    passed = sum(1 for e in delivered if not e["defective"])
    trips = loaded_trips(manifest, frames)
    ratios = [t["ratio"] for t in trips if t["ratio"] is not None]
    movement = counts["movement_conflict"]
    service = counts["service_conflict"]

    return {
        "ticks": len(frames),
        "total_throughput": total,
        "passed_throughput": passed,
        "defect_throughput": total - passed,
        "passing_rate": (passed / total) if total else None,
        "scrapped": counts["scrapped"],
        "on_time_rate": (sum(1 for e in delivered if e["on_time"]) / total)
        if total
        else None,
        "mean_lateness": (sum(e["lateness"] for e in delivered) / total)
        if total
        else None,
        "delivered_unknown_quality": sum(
            1 for e in delivered if e["observed_quality"] == "unknown"
        ),
        "total_agv_conflicts": movement + service,
        "movement_conflicts": movement,
        "movement_conflicts_agv": counts["movement_conflict_agv"],
        "movement_conflicts_env": counts["movement_conflict_env"],
        "pickup_dropoff_conflicts": service,
        "loaded_path_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "loaded_trips": len(trips),
        "inspection_batches": counts["inspection_start"],
        "shared_return": shared_return,
        "local_return_by_type": dict(local_by_type),
    }
