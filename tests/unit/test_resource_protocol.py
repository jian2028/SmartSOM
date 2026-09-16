"""Current sparse RLlib protocol: proposals share one committed physical tick."""

# ruff: noqa: E402
import importlib
import os
from copy import deepcopy
from dataclasses import replace

import pytest

if os.environ.get("SMARTSOM_REQUIRE_RESOURCE_PROTOCOL") == "1":
    importlib.import_module("ray.rllib")
else:
    pytest.importorskip("ray.rllib")
import numpy as np
from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from test_learning_gymnasium import hand_index
from test_production_runtime import small_scenario

from smartsom.learning.production_env import ProductionEnv
from smartsom.learning.production_ray import ResourceProductionEnv, role_mapping

pytestmark = pytest.mark.learning


def make_env(case=None):
    return ResourceProductionEnv(
        {"scenario": case or small_scenario(), "max_jobs": 8, "gamma": 1.0}
    )


def test_official_multiagent_protocol(monkeypatch):
    env = make_env()
    env.reset()
    for space in env.action_spaces.values():
        monkeypatch.setattr(
            space, "sample", lambda: int(np.flatnonzero(env.adapter.action_masks())[-1])
        )
    check_multiagent_environments(env)


def test_resource_and_central_decode_same_actions_and_reach_hand_result():
    env = make_env()
    central = ProductionEnv(small_scenario(), 8)
    observations, infos = env.reset()
    central.reset()
    actors = set()
    while not central.finished:
        actor = env._actor()
        assert set(observations) == {actor} == set(infos)
        assert role_mapping(actor) == central.role + "_policy"
        assert env.observation_spaces[actor].contains(observations[actor])
        np.testing.assert_array_equal(
            observations[actor]["observations"], central.observation()
        )
        np.testing.assert_array_equal(
            observations[actor]["action_mask"], central.action_masks()
        )
        index = hand_index(central)
        central.step(index)
        observations, rewards, done, truncated, infos = env.step({actor: index})
        actors.add(actor)
        assert env.adapter.sim.snapshot() == central.sim.snapshot()
        assert all(np.isfinite(value) for value in rewards.values())
    assert done["__all__"] and not truncated["__all__"]
    assert set(observations) == actors
    assert env.adapter.sim.tick == 8 and env.adapter.sim.completed == {"demand"}
    assert env.agents == []
    with pytest.raises(ValueError, match="termination"):
        env.step({env._actor(): 0})


@pytest.mark.parametrize(
    "actions",
    [
        {},
        {"missing": 0},
        {"machine:machine": 1},
        {"agv:agv": True},
        {"agv:agv": 999},
        {"agv:agv": 4, "machine:machine": 0},
    ],
)
def test_invalid_owner_or_action_does_not_mutate_resource_state(actions):
    env = make_env()
    env.reset()
    before = deepcopy(
        (
            env.adapter.sim.snapshot(),
            env.adapter.decisions,
            env.starts,
            env.pending_rewards,
            env.seen,
            env.agents,
        )
    )
    with pytest.raises(ValueError):
        env.step(actions)
    assert (
        env.adapter.sim.snapshot(),
        env.adapter.decisions,
        env.starts,
        env.pending_rewards,
        env.seen,
        env.agents,
    ) == before


@pytest.mark.parametrize("seed", [1, 17, 42])
def test_masked_resource_sampling_is_reproducible_over_entire_episode(seed):
    case = small_scenario(mode="dynamic", tick_limit=30)
    a, b = make_env(case), make_env(case)
    a.reset(seed=seed)
    b.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while not a.adapter.finished:
        assert a._actor() == b._actor()
        np.testing.assert_array_equal(
            a.adapter.action_masks(), b.adapter.action_masks()
        )
        index = int(rng.choice(np.flatnonzero(a.adapter.action_masks())))
        ra = a.step({a._actor(): index})
        rb = b.step({b._actor(): index})
        assert ra[1:] == rb[1:]
        assert a.adapter.sim.snapshot() == b.adapter.sim.snapshot()
    assert a.adapter.sim.tick == 30


def test_buffer_ranking_precedes_resource_proposals_without_advancing_time():
    case = small_scenario()
    case = replace(
        case, demands=case.demands + (replace(case.demands[0], demand_id="second"),)
    )
    env = make_env(case)
    observations, _ = env.reset()
    assert set(observations) == {"buffer:input"}
    jobs = env.adapter.current[1]
    observations, _, _, _, _ = env.step({"buffer:input": 1})
    assert env.adapter.sim.tick == 0 and set(observations) == {"agv:agv"}
    env.step({"agv:agv": 4})
    assert env.adapter.sim.agvs["agv"]["job"] == jobs[1]
    assert env.adapter.last_result["rankings"]["input"] == list(reversed(jobs))
