"""Optional custom-space Gymnasium contract, including zero-time decisions."""

from gymnasium.utils.env_checker import check_env

from smartsom.config.base_diagnostic import generate_scenario
from smartsom.learning.central_gym import CentralGym
from tests.unit.test_current_runtime import inputs


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
