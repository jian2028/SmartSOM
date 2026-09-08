"""Actual Parallel API, team rewards, and independent core equivalence."""
# ruff: noqa: E402

import os

import pytest

if os.environ.get("SMARTSOM_REQUIRE_PETTINGZOO") == "1":
    import pettingzoo
else:
    pettingzoo = pytest.importorskip("pettingzoo")

import gymnasium as gym
import numpy as np
from pettingzoo.test import parallel_api_test, parallel_seed_test
from test_fjsp_engine import flexible_case
from test_learning_projection import COMBINATIONS, module_case
from test_resource_projection import SPEC, indices
from test_static_engine import op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.domain import ArrivalPlan, Job, JobArrival
from smartsom.engine import Simulator
from smartsom.learning.episode import EpisodeInput, EpisodeLimits
from smartsom.learning.pettingzoo import SmartSOMParallelEnv

pytestmark = pytest.mark.pettingzoo


@pytest.mark.parametrize("flags", COMBINATIONS)
def test_parallel_matches_core_all_module_combinations(flags):
    inp = module_case(*flags)
    env = SmartSOMParallelEnv(inp, SPEC)
    env.reset(seed=101)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    policy = SPTPolicy()
    total = {a: 0 for a in env.possible_agents}
    while env.agents:
        action = policy.select_action(sim.current_decision)
        obs, rewards, done, truncated, info = env.step(indices(env.projected, action))
        result = sim.step(action)
        assert env.simulator.trace == sim.trace
        for a in total:
            total[a] += rewards[a]
            assert env.observation_space(a).contains(obs[a])
            assert not truncated[a]
    assert all(done.values()) and env.result == result
    assert set(total.values()) == {-result.makespan}
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_official_api_and_seed_checks():
    f, w = flexible_case()

    def factory():
        return SmartSOMParallelEnv(EpisodeInput(f, w), SPEC)

    parallel_api_test(factory(), num_cycles=100)

    # The official seed test samples without masks (unlike its API test).
    # Adapt its sampler only, never actions already selected or env.step().
    def seeded_factory():
        env = factory()

        class MaskedSampler(gym.spaces.Discrete):
            def __init__(self, agent):
                self.agent = agent
                super().__init__(env.action_space(agent).n)

            def sample(self, mask=None, probability=None):
                return super().sample(
                    mask=np.asarray(
                        env.projected.for_agent(self.agent).action_mask, dtype=np.int8
                    )
                )

        env.action_spaces = {a: MaskedSampler(a) for a in env.possible_agents}
        return env

    parallel_seed_test(seeded_factory, num_cycles=100)
    # PettingZoo 1.27's seed helper stops after the first step because it calls
    # any() on dictionary keys. Independently check the ENTIRE episode as well.
    for seed in range(10):
        a, b = seeded_factory(), seeded_factory()
        a.reset(seed=seed)
        b.reset(seed=seed)
        while a.agents:
            aa = {agent: a.action_space(agent).sample() for agent in a.agents}
            bb = {agent: b.action_space(agent).sample() for agent in b.agents}
            assert aa == bb
            a.step(aa)
            b.step(bb)
            assert a.steps == b.steps and a.simulator.trace == b.simulator.trace
            assert a.reason == b.reason and a.agents == b.agents


def test_initial_advance_and_failure_costs_and_budget():
    f, w = problem(Job("J", (op("J1", "M1", 3),)))
    inp = EpisodeInput(f, w, arrivals=ArrivalPlan((JobArrival("J", 5, 5),)))
    env = SmartSOMParallelEnv(inp, SPEC)
    env.reset()
    obs, rewards, done, trunc, _ = env.step(
        indices(
            env.projected, SPTPolicy().select_action(env.simulator.current_decision)
        )
    )
    assert (
        set(rewards.values()) == {-8} and all(done.values()) and not any(trunc.values())
    )
    env.reset()
    _, rewards, done, trunc, _ = env.step(indices(env.projected))
    assert (
        all(done.values())
        and env.reason == "policy_stalled"
        and env.total_reward == -10001
    )
    env = SmartSOMParallelEnv(inp, SPEC, limits=EpisodeLimits(max_ticks=6))
    env.reset()
    _, rewards, done, trunc, info = env.step(
        indices(
            env.projected, SPTPolicy().select_action(env.simulator.current_decision)
        )
    )
    assert all(trunc.values()) and not any(done.values())
    assert set(rewards.values()) == {-8} and env.rewarded_tick == 8


def test_invalid_joint_action_does_not_mutate_any_resource():
    f, w = flexible_case()
    env = SmartSOMParallelEnv(EpisodeInput(f, w), SPEC)
    env.reset()
    before = env.simulator.trace
    actions = {"machine:M1": 1, "machine:M2": True}
    with pytest.raises(ValueError):
        env.step(actions)
    assert env.simulator.trace == before and not env.steps and not env.finished
