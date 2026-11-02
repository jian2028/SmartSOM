"""Potential-based AGV distance shaping: sign, telescoping and configuration."""

import math

import pytest

from smartsom.learning.extensions import RewardTransition
from smartsom.learning.reward_shaping import AgvDistancePotential

TOPOLOGY = {"width": 6, "height": 1, "solids": [], "ports": {"goal": [5, 0]}}
PARAMETERS = {"weight": 2.0, "gradient": 0.5, "cells": 1.0, "gamma": 1.0}


def view(*cells, target="goal"):
    agvs = {
        f"agv_{i}": {"cell": [x, 0], "target": {"owner": "o", "port": target}}
        for i, x in enumerate(cells)
    }
    return {"agvs": agvs, "topology": TOPOLOGY}


def shaped(shaping, before, after, *, reason=None, reward=0.0):
    transition = RewardTransition(before, after, (), reward, 0, reason)
    return shaping.transform(transition, reward)


def test_potential_is_the_fleet_mean_of_exponential_distance_fields():
    shaping = AgvDistancePotential(PARAMETERS)
    expected = 2.0 * (math.exp(-0.5 * 5) + math.exp(-0.5 * 0)) / 2
    assert shaping.potential(view(0, 5)) == pytest.approx(expected)


def test_approaching_the_target_is_rewarded_and_retreating_penalized():
    shaping = AgvDistancePotential(PARAMETERS)
    assert shaped(shaping, view(2), view(3)) > 0
    assert shaped(shaping, view(3), view(2)) < 0
    assert shaped(shaping, view(3), view(3)) == 0
    assert shaped(shaping, view(2), view(3), reward=10.0) > 10.0


def test_episode_shaping_telescopes_to_the_initial_potential():
    for gamma in (1.0, 0.9):
        shaping = AgvDistancePotential({**PARAMETERS, "gamma": gamma})
        path = [view(x) for x in (0, 1, 1, 2, 1, 3, 4)]
        total = sum(
            gamma**t
            * shaped(
                shaping,
                path[t],
                path[t + 1],
                reason="truncated" if t == len(path) - 2 else None,
            )
            for t in range(len(path) - 1)
        )
        assert total == pytest.approx(-shaping.potential(path[0]))


def test_vehicles_without_targets_or_paths_contribute_nothing():
    shaping = AgvDistancePotential(PARAMETERS)
    idle = view(1)
    idle["agvs"]["agv_0"]["target"] = None
    assert shaping.potential(idle) == 0.0
    walled = {**view(0), "topology": {**TOPOLOGY, "solids": [[3, 0]]}}
    assert AgvDistancePotential(PARAMETERS).potential(walled) == 0.0


@pytest.mark.parametrize(
    "change",
    [{"gamma": 0.0}, {"gamma": 1.5}, {"gradient": 0.0}, {"cells": -1}, {"weight": -1}],
)
def test_invalid_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        AgvDistancePotential({**PARAMETERS, **change})
    with pytest.raises(ValueError):
        AgvDistancePotential({k: v for k, v in PARAMETERS.items() if k != "cells"})
