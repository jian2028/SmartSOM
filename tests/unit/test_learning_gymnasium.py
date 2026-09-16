"""Gym protocol and learning/core equivalence for current grid actions."""

# ruff: noqa: E402 -- optional dependency guard precedes adapters.
import os
from copy import deepcopy
from dataclasses import replace

import pytest

if os.environ.get("SMARTSOM_REQUIRE_GYM") == "1":
    import gymnasium as gym
else:
    gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")
from gymnasium.utils.env_checker import check_env
from pydantic import TypeAdapter
from test_production_runtime import small_scenario

from smartsom.api import load_config, prepare
from smartsom.config.codec import canonical_json
from smartsom.domain.production import JointCommand
from smartsom.engine.production import ProductionSimulator
from smartsom.learning.episode import EpisodeLimits
from smartsom.learning.production_env import ProductionEnv

pytestmark = pytest.mark.gym


def hand_index(env):
    if env.role == "machine":
        return 1
    return {0: 4, 1: 3, 2: 4, 3: 5, 4: 5, 5: 4, 6: 3, 7: 4}[env.sim.tick]


def test_official_checker_with_legal_mask_sampler(monkeypatch):
    env = ProductionEnv(small_scenario(), 4)
    env.reset()
    # Gym's checker samples without a mask. Supply a legal sample to the checker;
    # invalid action handling is verified independently below.
    monkeypatch.setattr(
        env.action_space, "sample", lambda: int(np.flatnonzero(env.action_masks())[-1])
    )
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize(
    "name",
    [
        "production_hand",
        "quality_production",
        "dynamic_production",
        "buffers_post_one",
        "transport_hand",
        "machine_events_fixed",
        "processing_fixed",
    ],
)
def test_gym_and_core_exact_semantic_history(name):
    case = prepare(
        load_config(f"configs/runs/{name}.yaml"), training=False
    ).resolved.scenario
    case = replace(case, tick_limit=30)
    env = ProductionEnv(case, 64)
    env.reset()
    core = ProductionSimulator(case)
    rng = np.random.default_rng(101)
    rewards = []
    while not env.finished:
        before = deepcopy(core.snapshot())
        obs, reward, terminated, truncated, info = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks())))
        )
        rewards.append(reward)
        assert env.observation_space.contains(obs)
        if info["physical_dt"]:
            row = env.last_result
            command = TypeAdapter(JointCommand).validate_json(
                canonical_json(row["actions"])
            )
            direct = core.step(command)
            assert direct["state"] == row["state"]
            assert direct["events"] == row["events"]
            assert direct["rejections"] == row["rejections"]
            assert direct["reward"] == reward
        else:
            assert core.snapshot() == before == env.sim.snapshot()
            assert reward == 0
    assert terminated != truncated
    assert sum(rewards) == pytest.approx(core.total_reward)
    assert core.snapshot() == env.sim.snapshot()


def test_hand_task_raw_rewards_are_charged_only_once_and_completion_is_real():
    env = ProductionEnv(small_scenario(), 4)
    env.reset()
    total, elapsed = 0, 0
    while not env.finished:
        _, reward, terminated, truncated, info = env.step(hand_index(env))
        total += reward
        elapsed += info["physical_dt"]
        if info["physical_dt"] == 0:
            assert reward == 0
    assert terminated and not truncated
    assert elapsed == env.sim.tick == 8
    assert env.sim.completed == {"demand"}
    assert total == pytest.approx(env.sim.total_reward)


@pytest.mark.parametrize(
    "bad", [-1, 999, True, False, np.bool_(True), 1.0, None, "3", 0]
)
def test_invalid_or_masked_actions_reject_before_any_mutation(bad):
    env = ProductionEnv(small_scenario(), 4)
    env.reset()
    before = (
        deepcopy(env.sim.snapshot()),
        deepcopy(env.decision_log),
        env.current,
        env.decisions,
    )
    with pytest.raises(ValueError):
        env.step(bad)
    assert (env.sim.snapshot(), env.decision_log, env.current, env.decisions) == before
    env.step(4)
    assert env.sim.tick == 1


def test_observation_and_mask_have_no_mutable_aliases():
    env = ProductionEnv(small_scenario(), 4)
    obs, _ = env.reset()
    expected = env.observation().copy()
    mask = env.action_masks()
    obs[:] = -99
    mask[:] = False
    np.testing.assert_array_equal(env.observation(), expected)
    assert env.action_masks().any()


def test_episode_source_owns_seed_sequence_and_rejects_changed_factory():
    case = small_scenario()
    calls = []

    def source(index):
        calls.append(index)
        return case

    env = ProductionEnv(case, 4, episode_source=source)
    a, _ = env.reset(seed=1)
    b, _ = env.reset(seed=999)
    np.testing.assert_array_equal(a, b)
    assert calls == [0, 1]
    before = env.sim, env.episode_index
    changed = replace(
        case,
        factory=replace(
            case.factory,
            machines=(replace(case.factory.machines[0], machine_id="renamed"),),
        ),
    )
    env.episode_source = lambda index: changed
    with pytest.raises(ValueError, match="frozen"):
        env.reset()
    assert (env.sim, env.episode_index) == before


@pytest.mark.parametrize("limit", [1, 3, 7])
def test_tick_budget_keeps_whole_commits_and_never_claims_completion(limit):
    env = ProductionEnv(small_scenario(), 4, limits=EpisodeLimits(max_ticks=limit))
    env.reset()
    while not env.finished:
        _, _, done, truncated, info = env.step(hand_index(env))
    assert not done and truncated and info["passed"] == 0
    assert env.sim.tick == limit and env.reason == "budget_exhausted"
    with pytest.raises(ValueError, match="termination"):
        env.step(5)
