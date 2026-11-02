"""Production potential: loaded/empty distance weights and job progress."""

import math

import pytest

from smartsom.learning.extensions import RewardTransition
from smartsom.learning.production_shaping import ProductionPotential

TOPOLOGY = {"width": 6, "height": 1, "solids": [], "ports": {"goal": [5, 0]}}
PARAMETERS = {
    "weight": 4.0,
    "empty_weight": 1.0,
    "gradient": 0.5,
    "cells": 1.0,
    "gamma": 1.0,
    "progress_weight": 3.0,
}


def view(cells=(), *, loaded=(), jobs=None, shipped=()):
    agvs = {
        f"agv_{i}": {
            "cell": [x, 0],
            "target": {"owner": "o", "port": "goal"},
            "job": "j" if i in loaded else None,
        }
        for i, x in enumerate(cells)
    }
    return {
        "agvs": agvs,
        "topology": TOPOLOGY,
        "jobs": jobs or {},
        "shipped": list(shipped),
    }


def shaped(shaping, before, after, reason=None):
    return shaping.transform(RewardTransition(before, after, (), 0.0, 0, reason), 0.0)


def test_loaded_and_empty_vehicles_use_their_own_distance_weights():
    shaping = ProductionPotential(PARAMETERS)
    near = math.exp(-0.5 * 1)
    assert shaping.potential(view([4], loaded=[0])) == pytest.approx(4.0 * near)
    assert shaping.potential(view([4])) == pytest.approx(1.0 * near)
    assert shaped(shaping, view([2], loaded=[0]), view([3], loaded=[0])) > shaped(
        shaping, view([2]), view([3])
    )


def test_completed_operations_and_shipments_raise_the_potential():
    shaping = ProductionPotential(PARAMETERS)
    job = {"demand": "d", "remaining_steps": ["a", "b"]}
    before = view(jobs={"d/1": job})
    after = view(jobs={"d/1": {**job, "remaining_steps": ["b"]}})
    assert shaped(shaping, before, after) == pytest.approx(3.0)
    done = view(jobs={"d/1": {**job, "remaining_steps": []}})
    shipped = view(jobs={"d/1": {**job, "remaining_steps": []}}, shipped=["d"])
    assert shaped(shaping, done, shipped) == pytest.approx(3.0)


def test_episode_shaping_still_telescopes():
    shaping = ProductionPotential({**PARAMETERS, "gamma": 0.9})
    job = {"demand": "d", "remaining_steps": ["a"]}
    path = [
        view([0], jobs={"d/1": job}),
        view([1], loaded=[0], jobs={"d/1": job}),
        view([2], loaded=[0], jobs={"d/1": {**job, "remaining_steps": []}}),
        view([3], jobs={"d/1": {**job, "remaining_steps": []}}, shipped=["d"]),
    ]
    total = sum(
        0.9**t
        * shaped(
            shaping, path[t], path[t + 1], "truncated" if t == len(path) - 2 else None
        )
        for t in range(len(path) - 1)
    )
    assert total == pytest.approx(-shaping.potential(path[0]))


@pytest.mark.parametrize(
    "change", [{"empty_weight": -1.0}, {"progress_weight": -1.0}, {"gamma": 0.0}]
)
def test_invalid_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        ProductionPotential({**PARAMETERS, **change})
    with pytest.raises(ValueError):
        ProductionPotential(
            {k: v for k, v in PARAMETERS.items() if k != "empty_weight"}
        )
