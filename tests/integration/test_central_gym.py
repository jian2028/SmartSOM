"""Optional custom-space Gymnasium contract, including zero-time decisions."""

from gymnasium.utils.env_checker import check_env

from smartsom.config.base_diagnostic import generate_scenario
from smartsom.engine.central_control import ADVANCE
from smartsom.learning.central_gym import CentralGym
from tests.unit.test_current_runtime import inputs, with_central


def test_central_gym_contract():
    factory, workload, options = inputs(control="centralized")
    env = CentralGym(
        lambda seed: generate_scenario(factory, workload, seed, "normal", 100), options
    )
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=123)
    assert env.observation_space.contains(observation)
    candidate = next(c for c in info["candidates"] if c.kind == "transport")
    _, reward, terminated, truncated, info = env.step(candidate.identity)
    assert reward == info["physical_dt"] == 0
    assert not terminated and not truncated
    _, _, _, _, info = env.step(info["advance_action"])
    assert info["physical_dt"] > 0


def test_central_gym_contract_when_advance_only_when_idle():
    factory, workload, options = inputs(control="centralized")
    options = with_central(options, advance_only_when_idle=True)
    env = CentralGym(
        lambda seed: generate_scenario(factory, workload, seed, "normal", 100), options
    )
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=123)
    assert ADVANCE not in info["candidates"] and info["advance_action"] is None
    for _ in range(50):
        action = env.action_space.sample()
        assert action in {c.identity for c in info["candidates"]}
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
