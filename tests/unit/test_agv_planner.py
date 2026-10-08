"""Regressions for shared port exits and independent station sides."""

import pytest

from smartsom.algorithms.agv_planner import (
    action_destination,
    plan_actions,
    priority_order,
    retarget_scouts,
)


def test_a_vehicle_cannot_fill_the_last_distinct_exit_of_two_servicing_ports():
    view = {
        "tick": 1,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [[1, 1], [2, 0], [2, 2], [4, 0], [4, 2]],
            "ports": {"left": [2, 1], "right": [4, 1], "goal": [0, 0]},
        },
        "agvs": {
            "left": {"cell": [2, 1], "service": {"kind": "pickup"}, "job": None},
            "right": {"cell": [4, 1], "service": {"kind": "pickup"}, "job": None},
            "passing": {
                "cell": [6, 1],
                "service": None,
                "job": "job",
                "target": {"owner": "output", "port": "goal"},
            },
        },
    }
    plan = plan_actions(view, {})
    assert plan["left"] == plan["right"] == "WAIT"
    assert action_destination(view["agvs"], "passing", plan["passing"]) != (5, 1)


def test_opposite_sides_with_independent_exits_can_be_entered_together():
    view = {
        "tick": 1,
        "topology": {
            "width": 5,
            "height": 5,
            "solids": [[2, 2]],
            "ports": {"upper": [2, 1], "lower": [2, 3]},
        },
        "agvs": {
            "upper": {
                "cell": [2, 0],
                "service": None,
                "job": "a",
                "target": {"owner": "station", "port": "upper"},
            },
            "lower": {
                "cell": [2, 4],
                "service": None,
                "job": "b",
                "target": {"owner": "station", "port": "lower"},
            },
        },
    }
    plan = plan_actions(view, {})
    assert plan == {"lower": "UP", "upper": "DOWN"}


@pytest.mark.parametrize("ready", ([], ["available_job"]))
def test_empty_target_intention_yields_before_loaded_vehicle_enters_transit_port(ready):
    view = {
        "tick": 1,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [[2, 0], [2, 2]],
            "ports": {"transit": [2, 1], "delivery": [5, 1]},
        },
        "sources": {"station": {"ready": ready}},
        "agvs": {
            "loaded": {
                "cell": [1, 1],
                "job": "job",
                "target": {"owner": "output", "port": "delivery"},
            },
            "standby": {
                "cell": [3, 1],
                "job": None,
                "target": {"owner": "station", "port": "transit"},
            },
        },
    }
    history = (
        {
            "loaded": {
                "tick": 0,
                "cell": [1, 1],
                "previous": None,
                "task": ["output", "delivery", "job"],
                "blocked_ticks": 0,
                "recent_cells": [[1, 1], [1, 0]] * 4,
            }
        }
        if ready
        else {}
    )
    plan = plan_actions(view, history)
    assert plan["loaded"] == "RIGHT"
    assert action_destination(view["agvs"], "standby", plan["standby"]) != (3, 1)


def test_aged_empty_intention_does_not_outrank_loaded_delivery():
    view = {
        "tick": 1,
        "sources": {"station": {"ready": []}},
        "agvs": {
            "loaded": {"cell": [1, 1], "job": "job"},
            "standby": {
                "cell": [3, 1],
                "job": None,
                "target": {"owner": "station", "port": "port"},
            },
        },
    }
    history = {"loaded": {"blocked_ticks": 0}, "standby": {"blocked_ticks": 999}}
    rankings = {"loaded": ("RIGHT", "WAIT"), "standby": ("WAIT", "UP")}
    assert priority_order(view, history, rankings, set()) == ("loaded", "standby")


def test_empty_intention_parks_when_no_work_exists_and_resumes_when_work_appears():
    view = {
        "tick": 1,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [[2, 0], [2, 2]],
            "ports": {"transit": [2, 1], "pickup": [5, 1]},
        },
        "sources": {"station": {"ready": []}},
        "agvs": {
            "empty": {
                "cell": [1, 1],
                "job": None,
                "target": {"owner": "station", "port": "pickup"},
            },
        },
    }
    history = {}
    assert plan_actions(view, history) == {"empty": "WAIT"}
    view["tick"] = 2
    view["sources"]["station"]["ready"] = ["available"]
    assert plan_actions(view, history) == {"empty": "RIGHT"}


def test_only_uncovered_ready_sources_need_empty_retarget_trips():
    view = {
        "topology": {"ports": {"near": [1, 0], "far": [4, 0], "work": [8, 0]}},
        "sources": {"pending": {"ready": []}, "work": {"ready": ["job"]}},
        "agvs": {
            "nearby": {
                "cell": [1, 1],
                "job": None,
                "target": {"owner": "pending", "port": "near"},
            },
            "distant": {
                "cell": [1, 1],
                "job": None,
                "target": {"owner": "pending", "port": "far"},
            },
        },
    }
    assert retarget_scouts(view) == {"nearby"}
    view["agvs"]["distant"]["target"] = {"owner": "work", "port": "work"}
    assert retarget_scouts(view) == set()


def test_transit_entry_waits_when_its_forward_exit_will_remain_occupied():
    view = {
        "tick": 1,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [[2, 0], [2, 2]],
            "ports": {"transit": [2, 1], "delivery": [5, 1]},
        },
        "agvs": {
            "a": {
                "cell": [1, 1],
                "job": "a",
                "target": {"owner": "output", "port": "delivery"},
            },
            "b": {
                "cell": [3, 1],
                "job": "b",
                "target": {"owner": "output", "port": "delivery"},
            },
        },
    }
    plan = plan_actions(
        view, {}, allowed_actions={"a": ("RIGHT", "WAIT"), "b": ("WAIT",)}
    )
    assert plan == {"a": "WAIT", "b": "WAIT"}


def test_transit_lookahead_does_not_forbid_waiting_on_an_already_occupied_port():
    view = {
        "tick": 1,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [[1, 1], [2, 0], [2, 2]],
            "ports": {"transit": [2, 1], "delivery": [5, 1]},
        },
        "agvs": {
            "a": {
                "cell": [2, 1],
                "job": "a",
                "target": {"owner": "output", "port": "delivery"},
            },
            "b": {
                "cell": [3, 1],
                "job": "b",
                "target": {"owner": "output", "port": "delivery"},
            },
        },
    }
    # With no currently free non-port exit, the core permits WAIT. Entry-only
    # lookahead must not reject it and fall back to a conflicting move into b.
    plan = plan_actions(
        view, {}, allowed_actions={"a": ("RIGHT", "WAIT"), "b": ("WAIT",)}
    )
    assert plan == {"a": "WAIT", "b": "WAIT"}


def test_four_vehicle_rotation_has_distinct_destinations_and_no_swaps():
    cells = {"a": [0, 0], "b": [1, 0], "c": [1, 1], "d": [0, 1]}
    view = {
        "tick": 1,
        "topology": {"width": 2, "height": 2, "solids": [], "ports": {}},
        "agvs": {owner: {"cell": cell, "job": None} for owner, cell in cells.items()},
    }
    expected = {"a": "RIGHT", "b": "DOWN", "c": "LEFT", "d": "UP"}
    plan = plan_actions(
        view,
        {},
        allowed_actions={owner: (action,) for owner, action in expected.items()},
    )
    assert plan == expected
    destinations = {
        owner: action_destination(view["agvs"], owner, action)
        for owner, action in plan.items()
    }
    assert len(set(destinations.values())) == len(cells)
    assert all(
        not (
            destination == tuple(cells[other])
            and destinations[other] == tuple(cells[owner])
        )
        for owner, destination in destinations.items()
        for other in cells
        if other != owner
    )
