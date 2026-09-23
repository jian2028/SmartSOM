"""Tests for the SO-MARL replay debug prototype (scripts/replay_debug)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# pytest puts only src/ on the path; the prototype lives under scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.replay_debug.layout import Grid, default_layout  # noqa: E402
from scripts.replay_debug.metrics import compute_metrics  # noqa: E402
from scripts.replay_debug.mock_episode import generate_episode  # noqa: E402
from scripts.replay_debug.render import PLACEHOLDER, render_html  # noqa: E402
from scripts.replay_debug.schema import (
    SCHEMA_VERSION,
    load_trace,
    validate_trace,
    write_trace,
)  # noqa: E402


# --------------------------------------------------------------------- layout
def test_default_layout_matches_slide_11() -> None:
    layout = default_layout()
    assert (layout["width"], layout["height"]) == (12, 7)
    types = [s["type"] for s in layout["stations"]]
    assert types.count("machine") == 8
    assert types.count("pre_buffer") == types.count("post_buffer") == 8
    assert (
        types.count("inspection") == 2
        and types.count("disposal") == 2
        and types.count("charger") == 4
    )
    machines = {s["id"]: s for s in layout["stations"] if s["type"] == "machine"}
    # Column pairs share an operation (slide 35): M1/M2 -> O1 ... M7/M8 -> O4.
    assert [machines[f"M{i}"]["operation"] for i in range(1, 9)] == [
        "O1",
        "O1",
        "O2",
        "O2",
        "O3",
        "O3",
        "O4",
        "O4",
    ]


def test_every_interaction_point_is_road_and_connected() -> None:
    layout = default_layout()
    grid = Grid(layout)
    ips = [tuple(p) for s in layout["stations"] for p in s["interaction_points"]]
    assert all(grid.is_road(p) for p in ips)
    reachable = grid.distances_from(ips[0])
    assert all(p in reachable for p in ips)


# ----------------------------------------------------------------- mock trace
@pytest.mark.parametrize("controller", ["rule", "marl"])
def test_mock_episode_is_valid_and_deterministic(controller: str) -> None:
    manifest, frames = generate_episode(controller, seed=3, horizon=300)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert validate_trace(manifest, frames) == []
    _, frames_again = generate_episode(controller, seed=3, horizon=300)
    assert json.dumps(frames) == json.dumps(frames_again)


def test_rule_controller_never_conflicts_but_marl_does() -> None:
    rule = compute_metrics(*generate_episode("rule", seed=1, horizon=500))
    marl = compute_metrics(*generate_episode("marl", seed=1, horizon=500))
    assert rule["total_agv_conflicts"] == 0
    assert rule["total_throughput"] > 0
    assert marl["total_agv_conflicts"] > 0


def test_write_and_load_roundtrip(tmp_path) -> None:
    manifest, frames = generate_episode("rule", seed=0, horizon=50)
    write_trace(tmp_path / "trace", manifest, frames)
    loaded_manifest, loaded_frames = load_trace(tmp_path / "trace")
    assert loaded_manifest == manifest
    assert loaded_frames == frames


# ----------------------------------------------------------- hand-made trace
def _tiny_trace() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """3x3 open grid, one AGV: pickup at (0,0), detour, dropoff at (2,0)."""
    layout = {
        "width": 3,
        "height": 3,
        "obstacles": [],
        "stations": [
            {
                "id": "IN",
                "type": "input",
                "cell": [0, 0],
                "interaction_points": [[0, 0]],
                "capacity": 2,
            },
            {
                "id": "OUT",
                "type": "output",
                "cell": [2, 0],
                "interaction_points": [[2, 0]],
                "capacity": None,
            },
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "tiny",
        "controller": "test",
        "seed": 0,
        "horizon": 6,
        "layout": layout,
        "agents": [{"id": "A1.move", "type": "mover", "entity": "A1"}],
        "jobs": [
            {"id": "J1", "release_tick": 0, "due_tick": 3, "rush": False},
            {"id": "J2", "release_tick": 0, "due_tick": 99, "rush": False},
        ],
    }
    path = [(0, 0), (1, 0), (1, 1), (1, 0), (2, 0), (2, 0)]
    events: list[list[dict[str, Any]]] = [[] for _ in path]
    events[0].append({"type": "pickup", "agv": "A1", "job": "J1", "station": "IN"})
    events[2].append(
        {"type": "movement_conflict", "agv": "A1", "kind": "env", "cell": [1, 2]}
    )
    events[4].append({"type": "dropoff", "agv": "A1", "job": "J1", "station": "OUT"})
    events[4].append(
        {
            "type": "delivered",
            "job": "J1",
            "observed_quality": "unknown",
            "defective": True,
            "lateness": 1,
            "on_time": False,
        }
    )
    events[5].append(
        {
            "type": "delivered",
            "job": "J2",
            "observed_quality": "pass",
            "defective": False,
            "lateness": 0,
            "on_time": True,
        }
    )
    events[5].append(
        {"type": "service_conflict", "agv": "A1", "station": "OUT", "reason": "test"}
    )
    frames = [
        {
            "tick": t,
            "agvs": [
                {
                    "id": "A1",
                    "pos": list(p),
                    "status": "moving",
                    "carrying": None,
                    "goal": None,
                }
            ],
            "machines": [],
            "buffers": [],
            "inspection": [],
            "jobs": {},
            "decisions": [],
            "rewards": {"shared": 1.0, "local": {"A1.move": -0.5}},
            "events": events[t],
        }
        for t, p in enumerate(path)
    ]
    return manifest, frames


def test_metrics_on_hand_made_trace() -> None:
    manifest, frames = _tiny_trace()
    assert validate_trace(manifest, frames) == []
    m = compute_metrics(manifest, frames)
    assert (m["total_throughput"], m["passed_throughput"], m["defect_throughput"]) == (
        2,
        1,
        1,
    )
    assert m["passing_rate"] == pytest.approx(0.5)
    assert m["on_time_rate"] == pytest.approx(0.5)
    assert m["movement_conflicts_env"] == 1 and m["pickup_dropoff_conflicts"] == 1
    assert m["total_agv_conflicts"] == 2
    # 4 moves for a shortest distance of 2.
    assert m["loaded_trips"] == 1
    assert m["loaded_path_ratio"] == pytest.approx(2.0)
    assert m["shared_return"] == pytest.approx(6.0)
    assert m["local_return_by_type"] == {"mover": pytest.approx(-3.0)}


def test_validator_reports_broken_frames() -> None:
    manifest, frames = _tiny_trace()
    frames[1]["tick"] = 7
    frames[2]["agvs"].append({"id": "A2", "pos": [1, 1], "status": "flying"})
    frames[3]["events"].append({"type": "teleport"})
    frames[4]["events"].append({"type": "pickup", "agv": "A1"})
    errors = "\n".join(validate_trace(manifest, frames))
    assert "increase by 1" in errors
    assert "share cell" in errors
    assert "unknown status 'flying'" in errors
    assert "unknown event type 'teleport'" in errors
    assert "pickup event missing" in errors


# --------------------------------------------------------------------- render
def test_render_embeds_trace_safely(tmp_path) -> None:
    manifest, frames = _tiny_trace()
    manifest["note"] = "</script><script>alert(1)</script>"
    html = render_html(manifest, frames, tmp_path / "replay.html").read_text(
        encoding="utf-8"
    )
    assert PLACEHOLDER not in html
    assert "<\\/script><script>alert(1)<\\/script>" in html
    assert html.count("</script>") == 2  # only the viewer's own two script tags
