"""Physical-time credit and policy projections for the grid runtime."""

# ruff: noqa: E402 -- optional dependency guards precede adapter imports.
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
from smartsom.api import load_config, prepare
from smartsom.learning.production import physical_advantages
from smartsom.learning.production_env import ProductionEnv


def scenario():
    return prepare(
        load_config("configs/runs/quality_production.yaml"), training=False
    ).resolved.scenario


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_scale", 10.0),
        ("count_scale", 10.0),
        ("max_jobs", 9),
        ("hidden_sizes", (16,)),
    ],
)
def test_initialization_rejects_changed_input_or_network_contract(field, value):
    from smartsom.config.codec import primitive
    from smartsom.config.production import AlgorithmConfig
    from smartsom.learning.production import validate_model_contract
    from smartsom.learning.production_env import ACTION_CONTRACT, OBSERVATION_CONTRACT
    from smartsom.trace.production import state_hash

    case = scenario()
    original = AlgorithmConfig(provider="sb3.maskable_ppo", max_jobs=8)
    manifest = {
        "schema": "smartsom.production-checkpoint/v1",
        "factory_hash": state_hash(primitive(case.factory)),
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "algorithm": primitive(original),
    }
    # Optimization settings may change when starting a new experiment from weights.
    validate_model_contract(
        manifest, case, original.model_copy(update={"learning_rate": 0.001})
    )
    with pytest.raises(ValueError, match=field):
        validate_model_contract(
            manifest, case, original.model_copy(update={field: value})
        )
    del manifest["action_contract"]
    with pytest.raises(ValueError, match="retrain"):
        validate_model_contract(manifest, case, original)


def test_zero_duration_decisions_do_not_discount_physical_rewards():
    advantages, returns = physical_advantages(
        np.array([[0.0], [1.0], [2.0]]),
        np.zeros((3, 1)),
        np.array([[1.0], [0.0], [0.0]]),
        np.array([[0.0], [1.0], [0.0]]),
        np.array([4.0]),
        np.array([False]),
        0.5,
        1.0,
    )
    np.testing.assert_array_equal(advantages[:, 0], [4.0, 4.0, 6.0])
    np.testing.assert_array_equal(returns, advantages)


def test_resource_metrics_count_committed_commands_not_joint_ticks():
    from smartsom.learning.production_sampling import committed_actions

    row = {
        "actions": {
            "agvs": (("a", "RIGHT"), ("b", "LEFT"), ("c", "WAIT")),
            "machines": (
                ("m", {"job_id": "job", "mode_id": "normal"}),
                ("idle", {"job_id": None, "mode_id": None}),
            ),
            "quality": (("q", "INSPECT"),),
            "rankings": (("buffer", ("job",)),),
        },
        "rejections": {"agv:b": "conflict"},
    }
    assert [(role, resource) for role, resource, _ in committed_actions(row)] == [
        ("agv", "a"),
        ("machine", "m"),
        ("quality", "q"),
    ]


def test_decision_limit_stops_before_uncommitted_physics():
    from smartsom.learning.episode import EpisodeLimits

    env = ProductionEnv(
        scenario(), 8, limits=EpisodeLimits(max_decisions=1, max_ticks=10)
    )
    env.reset()
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated and env.finished
    assert env.reason == "budget_exhausted" and env.sim.tick == 0
    assert env.last_result is None and not env.sim.done
    with pytest.raises(ValueError, match="termination"):
        env.step(0)


def test_tick_limit_does_not_execute_an_extra_tick():
    from smartsom.learning.episode import EpisodeLimits

    env = ProductionEnv(
        scenario(), 8, limits=EpisodeLimits(max_decisions=100, max_ticks=2)
    )
    env.reset()
    while not env.finished:
        env.step(int(np.flatnonzero(env.action_masks())[-1]))
    assert env.sim.tick == 2 and env.reason == "budget_exhausted"


def test_episode_boundary_stops_advantage_propagation():
    advantages, _ = physical_advantages(
        np.array([[1.0], [2.0], [100.0]]),
        np.zeros((3, 1)),
        np.array([[1.0], [0.0], [1.0]]),
        np.ones((3, 1)),
        np.array([0.0]),
        np.array([True]),
        0.5,
        1.0,
    )
    np.testing.assert_array_equal(advantages[:, 0], [2.0, 2.0, 100.0])


def test_buffer_ranks_before_physics_and_keeps_semantic_candidates():
    env = ProductionEnv(scenario(), 8)
    obs, info = env.reset()
    assert info["actor"] == "buffer:input"
    assert env.sim.tick == 0
    original = env.current[1]
    _, _, _, _, info = env.step(2)
    assert info["physical_dt"] == 0
    assert env.current[1] == original
    assert not env.action_masks()[2]
    env.step(1)
    assert env.sim.tick == 0
    assert env.rankings["input"] == [original[2], original[1], original[0]]
    assert env.role == "agv"
    row = env.step(4)
    assert row[-1]["physical_dt"] == 1
    assert env.sim.agvs["agv"]["job"] == original[2]
    assert env.last_result["rankings"]["input"] == list(reversed(original))


def test_hidden_defect_and_unfinished_actual_time_do_not_change_actor_input():
    env = ProductionEnv(scenario(), 8)
    env.reset()
    before = env.observation().copy()
    job = next(iter(env.sim.jobs))
    env.sim.jobs[job]["defective"] = True
    env.sim.machine_state["machine"]["remaining"] = 12345
    np.testing.assert_array_equal(before, env.observation())


def test_overflow_is_explicit_not_silent_candidate_truncation():
    with pytest.raises(ValueError, match="max_jobs"):
        ProductionEnv(scenario(), 2).reset()


def test_recorded_adapter_actions_reproduce_core():
    import json

    from pydantic import TypeAdapter

    from smartsom.domain.production import JointCommand
    from smartsom.engine.production import ProductionSimulator

    env = ProductionEnv(replace(scenario(), tick_limit=30), 8)
    env.reset()
    independent = ProductionSimulator(env.scenario)
    rng = np.random.default_rng(87)
    while not env.sim.done:
        _, _, _, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        if info["physical_dt"]:
            command = TypeAdapter(JointCommand).validate_json(
                json.dumps(env.last_result["actions"])
            )
            assert independent.step(command)["state"] == env.last_result["state"]


@pytest.mark.learning
def test_sb3_resume_preserves_optimizer_sampler_and_random_state(tmp_path):
    pytest.importorskip("sb3_contrib")
    import torch
    from sb3_contrib import MaskablePPO

    from smartsom.config.production import AlgorithmConfig
    from smartsom.learning.production import train_sb3

    case = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved.scenario
    algo = AlgorithmConfig(provider="sb3.maskable_ppo", hidden_sizes=(8,), max_jobs=4)
    full = train_sb3(case, algo, tmp_path / "full", total_steps=128, rollout_steps=64)
    first = train_sb3(case, algo, tmp_path / "first", total_steps=64, rollout_steps=64)
    resumed = train_sb3(
        case,
        algo,
        tmp_path / "resumed",
        total_steps=64,
        rollout_steps=64,
        resume_from=first,
    )
    a = MaskablePPO.load(full / "model.zip")
    b = MaskablePPO.load(resumed / "model.zip")
    assert a.num_timesteps == b.num_timesteps == 128
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, b.policy.state_dict()[key]), key


@pytest.mark.learning
def test_ray_bootstrap_rows_are_removed_after_physical_gae():
    pytest.importorskip("ray")
    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.evaluation.postprocessing import Postprocessing

    from smartsom.learning.production_ray import PhysicalGAE

    class Module:
        def compute_values(self, data):
            return torch.tensor([0.0, 0.0, 4.0])

    batch = {
        "role": {
            Columns.OBS: {
                "physical_tick": torch.tensor([[0.0], [1.0], [2.0]]),
                "observations": torch.tensor([[10.0], [20.0], [30.0]]),
            },
            Columns.REWARDS: torch.tensor([1.0, 2.0, 0.0]),
            Columns.TERMINATEDS: torch.tensor([False, False, False]),
            Columns.TRUNCATEDS: torch.tensor([False, True, True]),
            Columns.LOSS_MASK: torch.tensor([True, True, False]),
        }
    }
    result = PhysicalGAE(0.5, 1)(rl_module={"role": Module()}, batch=batch)["role"]
    assert result[Columns.LOSS_MASK].tolist() == [True, True]
    assert result[Columns.OBS]["observations"].flatten().tolist() == [10.0, 20.0]
    assert result[Postprocessing.VALUE_TARGETS].tolist() == [3.0, 4.0]


def test_inference_cleanup_preserves_the_loading_error(monkeypatch):
    from smartsom.learning.production import LearnedProductionDriver

    primary = ValueError("invalid model weights")

    class BrokenEnvironment:
        def close(self):
            raise RuntimeError("environment close failed")

    def load(driver, *args, **kwargs):
        driver.env = BrokenEnvironment()
        raise primary

    monkeypatch.setattr(LearnedProductionDriver, "_load", load)
    with pytest.raises(ValueError, match="invalid model weights") as caught:
        LearnedProductionDriver("unused", None)
    assert caught.value is primary
    assert any("environment close failed" in note for note in primary.__notes__)
