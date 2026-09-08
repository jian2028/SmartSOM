"""Optional Gym protocol, reward accounting and semantic replay acceptance."""

# ruff: noqa: E402 -- optional dependency guard must precede adapter imports.

import os
from dataclasses import replace

import pytest

if os.environ.get("SMARTSOM_REQUIRE_GYM") == "1":
    import gymnasium as gym
else:
    gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env
from test_fjsp_engine import flexible_case
from test_learning_projection import COMBINATIONS, SPEC, module_case
from test_static_engine import op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.domain import ArrivalPlan, Job, JobArrival
from smartsom.engine import Simulator, replay
from smartsom.learning.episode import EpisodeInput, EpisodeLimits
from smartsom.learning.gymnasium import SchedulingEnv

pytestmark = pytest.mark.gym


@pytest.mark.parametrize("kind", ["plain", "masked"])
def test_official_checker(kind):
    f, w = flexible_case()
    env = SchedulingEnv(EpisodeInput(f, w), SPEC, observation_kind=kind)
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("flags", COMBINATIONS)
def test_gym_and_core_exact_semantic_history(flags):
    inputs = module_case(*flags)
    env = SchedulingEnv(inputs, SPEC)
    env.reset(seed=101)
    core = Simulator(inputs.factory, inputs.workload, **inputs.options())
    policy = SPTPolicy()
    rewards = []
    while core.current_decision is not None:
        action = policy.select_action(core.current_decision)
        index = env.projected.actions.index(action)
        obs, reward, terminated, truncated, info = env.step(index)
        result = core.step(action)
        rewards.append(reward)
        assert env.simulator.trace == core.trace
        assert env.observation_space.contains(obs)
        assert not truncated
    assert terminated and info["is_success"]
    assert env.result == result
    assert sum(rewards) == -result.makespan
    assert (
        replay(inputs.factory, inputs.workload, result.actions, **inputs.options())
        == result
    )


def test_initial_autoadvance_is_charged_once_and_seed_cannot_change_fixed_input():
    f, w = problem(Job("J", (op("J1", "M1", 3),)))
    inp = EpisodeInput(f, w, arrivals=ArrivalPlan((JobArrival("J", 5, 5),)))
    env = SchedulingEnv(inp, SPEC)
    a, _ = env.reset(seed=1)
    index = next(i for i, a in enumerate(env.projected.actions) if a is not None)
    _, reward, done, truncated, info = env.step(index)
    assert reward == -8 and done and not truncated and info["makespan"] == 8
    b, _ = env.reset(seed=999)
    np.testing.assert_array_equal(a, b)


def test_invalid_action_is_atomic_and_fatal_for_training():
    f, w = flexible_case()
    env = SchedulingEnv(EpisodeInput(f, w), SPEC)
    env.reset()
    trace = env.simulator.trace
    invalid = next(i for i, x in enumerate(env.projected.actions) if x is None)
    _, reward, done, truncated, info = env.step(invalid)
    assert done and not truncated and reward == -10001
    assert info["end_reason"] == "invalid_action" and info["makespan"] is None
    assert env.simulator.trace == trace
    env.strict_actions = True
    env.reset()
    with pytest.raises(ValueError):
        env.step(invalid)
    assert env.simulator.trace == trace


def test_budget_does_not_cut_atomic_physical_transition():
    f, w = problem(Job("J", (op("J1", "M1", 20),)))
    env = SchedulingEnv(EpisodeInput(f, w), SPEC, limits=EpisodeLimits(3, 10))
    env.reset()
    index = next(i for i, x in enumerate(env.projected.actions) if x is not None)
    _, reward, terminated, truncated, info = env.step(index)
    assert not terminated and truncated and reward == -20
    assert info["simulation_time"] == 20 and info["makespan"] is None
    assert env.result.makespan == 20  # Physics finished, the budgeted episode did not.


def test_decision_budget_failure_cost_includes_elapsed_time():
    f, w = flexible_case()
    env = SchedulingEnv(EpisodeInput(f, w), SPEC, limits=EpisodeLimits(1, 10))
    env.reset()
    index = next(i for i, x in enumerate(env.projected.actions) if x is not None)
    _, reward, terminated, truncated, info = env.step(index)
    assert not terminated and truncated and reward == -11 and info["makespan"] is None


def test_backend_views_have_identical_mask_and_decode_and_no_mutable_aliases():
    inp = module_case(True, True, True, True, True, True, True)
    a = SchedulingEnv(inp, SPEC)
    b = SchedulingEnv(inp, SPEC, observation_kind="masked")
    obs_a, _ = a.reset(seed=1)
    obs_b, _ = b.reset(seed=2)
    np.testing.assert_array_equal(obs_a, obs_b["observations"])
    np.testing.assert_array_equal(a.action_masks(), obs_b["action_mask"])
    assert a.projected == b.projected
    obs_a[:] = -99
    mask = a.action_masks()
    mask[:] = False
    assert any(a.projected.action_mask) and min(a.projected.observations) >= 0


def test_episode_source_owns_scientific_sequence():
    inp = module_case(True, False, False, False, False, False, False)
    calls = []

    def source(i):
        calls.append(i)
        return inp

    env = SchedulingEnv(inp, SPEC, episode_source=source)
    env.reset(seed=100)
    env.reset(seed=100)
    assert calls == [0, 1]
    env.episode_source = lambda i: replace(inp, workload=flexible_case()[1])
    with pytest.raises(ValueError, match="frozen"):
        env.reset()


def test_real_deadlock_is_failed_terminal_and_retains_physical_trace():
    from test_buffers import dispatch, move, vehicle_case

    from smartsom.domain import MachineBuffers

    f, w, _ = vehicle_case()
    f = replace(
        f,
        buffers=(MachineBuffers("M1", 0, 0),),
        transport=replace(f.transport, agvs=f.transport.agvs[:1]),
    )
    env = SchedulingEnv(
        EpisodeInput(f, w, buffers_enabled=True, transport_enabled=True), SPEC
    )
    env.reset()
    for action in (move("A", "M1", "V1"), dispatch("A"), move("B", "M1", "V1")):
        _, _, terminated, truncated, info = env.step(
            env.projected.actions.index(action)
        )
    assert terminated and not truncated and info["end_reason"] == "deadlock"
    assert env.total_reward == -10001 and info["makespan"] is None
    assert any(r.kind == "wait_for_unload" for r in env.simulator.trace)


def test_policy_stalled_is_distinct_from_hidden_future_event():
    from test_holding import holding_factory, move

    from smartsom.dispatch import Dispatch

    f, w = problem(
        Job("A", (op("A1", "M1", 1), op("A2", "M2", 1, "A1"))),
        Job("B", (op("B1", "M2", 1),)),
        Job("C", (op("C1", "M2", 1),)),
    )
    f = holding_factory(f, capacity=0, zero=True)
    inp = EpisodeInput(
        f,
        w,
        transport_enabled=True,
        holding_buffer_enabled=True,
        arrivals=ArrivalPlan(
            (JobArrival("A", 0, 0), JobArrival("B", 5, 5), JobArrival("C", 10, 10))
        ),
        decision_trigger="arrival_event",
    )
    env = SchedulingEnv(inp, SPEC)
    env.reset()
    for action in (move("A", "M1"), Dispatch("A1", "standard"), move("A", "H")):
        _, _, terminated, truncated, info = env.step(
            env.projected.actions.index(action)
        )
    assert terminated and not truncated and info["end_reason"] == "policy_stalled"
    assert info["simulation_time"] == 5 and info["makespan"] is None
    assert not any(env.action_masks())
    assert tuple(j.job_id for j in env.simulator.current_decision.jobs) == ("A", "B")


def test_zero_time_rerouting_is_bounded_without_changing_kernel_candidates():
    from test_holding import holding_factory, move

    from smartsom.domain import Operation, ProcessingMode

    f, w = problem(
        Job(
            "A",
            (
                Operation(
                    "A1", (ProcessingMode("a", "M1", 1), ProcessingMode("b", "M2", 2))
                ),
            ),
        )
    )
    f = holding_factory(f, zero=True)
    env = SchedulingEnv(
        EpisodeInput(f, w, transport_enabled=True), SPEC, limits=EpisodeLimits(4, 10)
    )
    env.reset()
    for destination in ("M1", "M2", "M1", "M2"):
        action = move("A", destination)
        _, _, terminated, truncated, info = env.step(
            env.projected.actions.index(action)
        )
    assert not terminated and truncated and info["simulation_time"] == 0
    assert env.total_reward == -11
    assert env.simulator.current_decision.candidates
