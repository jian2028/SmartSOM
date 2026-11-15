"""Actual official learner updates over factory decisions and learned waits."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
import yaml
from ray.rllib.algorithms.algorithm import Algorithm

from smartsom.config.base_diagnostic import BaseAlgorithm
from smartsom.learning.rllib_config import build_algorithm_config
from smartsom.learning.rllib_env import FactoryMultiAgentEnv
from tests.integration.test_rllib_dqn_checkpoint import assert_state_equal
from tests.unit.test_base_reservations import sim_with


@pytest.mark.parametrize("method", ["ppo", "dqn"])
@pytest.mark.parametrize("remote", [False, True])
def test_static_resident_requests_match_pinned_official_requests(method, remote):
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.algorithms.ppo import PPOConfig

    from smartsom.config.rllib_resources import RLlibResources

    profile = RLlibResources.model_validate(
        {
            "resources": {"num_cpus_for_main_process": 2, "memory_gib": 3},
            "learners": {
                "num_learners": int(remote),
                "num_cpus_per_learner": 3,
                **({"memory_gib": 4} if remote else {}),
            },
            "env_runners": {"memory_gib": 1},
            "evaluation_config": {"num_cpus_per_env_runner": 0.5},
        }
    )
    config = PPOConfig() if method == "ppo" else DQNConfig()
    args = profile.arguments()
    config.resources(**args["resources"]).learners(**args["learners"])
    config.env_runners(**args["env_runners"])
    config.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=4,
        evaluation_config=profile.evaluation_arguments(),
    )
    actual = config.algo_class.default_resource_request(config).bundles
    declared = profile.declared_bundles(algorithm=method, evaluation_workers=4)

    def total(bundles):
        return {
            key: sum(bundle.get(key, 0) for bundle in bundles)
            for key in ("CPU", "GPU", "smartsom_memory_gib")
        }

    assert total(actual) == total(declared)


def test_initialization_timestamp_survives_training_metric_reduction():
    from types import SimpleNamespace

    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    from smartsom.learning.rllib_metrics import FactoryProgress

    metrics = MetricsLogger(root=True)
    algorithm = SimpleNamespace(metrics=metrics)
    callback = FactoryProgress()
    callback.on_algorithm_init(algorithm=algorithm)
    expected = metrics.peek("smartsom_algorithm_initialized_unix")
    for _ in range(2):
        metrics.reduce()
        result = {}
        callback.on_train_result(
            algorithm=algorithm, metrics_logger=metrics, result=result
        )
        assert result["algorithm_initialized_unix"] == expected


def test_training_worlds_follow_official_reset_seed_and_preserve_fixed_evaluation():
    from smartsom.config.base_diagnostic import generate_scenario, load_inputs
    from smartsom.learning.rllib_worlds import TrainingWorlds

    root = Path(__file__).resolve().parents[2]
    ex, factory, workload, _, _ = load_inputs(
        root / "experiments/examples/small/default/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 2, "initial_orders": 1})
    ex = ex.model_copy(
        update={
            "training_conditions": ("normal", "low"),
            "condition_sampling": "balanced_cycle",
        }
    )
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    )
    scenario = generate_scenario(factory, workload, 101, "normal", 12)
    config = {
        "scenario": scenario,
        "options": options,
        "algorithm": "dqn",
        "training_worlds": TrainingWorlds(ex, factory, workload, 101, 12),
    }
    streams = []
    for _ in range(2):
        env = FactoryMultiAgentEnv(config)
        try:
            stream = []
            for i in range(3):
                env.reset(seed=721 if i == 0 else None)
                stream.append((dict(env.world_identity), env.episode.sim.scenario))
            env.set_evaluation_case(
                {
                    "case_id": "fixed",
                    "scenario": scenario,
                    "action_seed": 3,
                    "mode": "argmax",
                }
            )
            env.reset()
            assert env.episode.sim.scenario == scenario
            assert env.world_identity == {"case_id": "fixed"}
            streams.append(stream)
        finally:
            env.close()
    assert streams[0] == streams[1]
    assert [identity["condition"] for identity, _ in streams[0]] == [
        "normal",
        "low",
        "normal",
    ]
    assert len({identity["world_seed"] for identity, _ in streams[0]}) == 3


def test_builder_preserves_official_replay_defaults_and_explicit_overrides():
    from ray.rllib.algorithms.dqn import DQNConfig

    from smartsom.learning.rllib_dqn import ReplayCheckpointDQN

    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    )
    defaults = DQNConfig().replay_buffer_config
    defaults["type"] = (
        "smartsom.learning.rllib_dqn.RestorableMultiAgentPrioritizedEpisodeReplayBuffer"
    )
    for overrides in ({}, {"replay_buffer_config": {"capacity": 123}}):
        original = deepcopy(overrides)
        config = build_algorithm_config(
            scenario=sim_with(n=2).scenario,
            options=options,
            algorithm="dqn",
            seed=101,
            training=overrides,
        )
        for key, value in defaults.items():
            assert config.replay_buffer_config[key] == overrides.get(
                "replay_buffer_config", {}
            ).get(key, value)
        assert config.algo_class is ReplayCheckpointDQN
        assert config.store_buffer_in_checkpoints
        assert overrides == original


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize(
    "fusion,ownership",
    [("base", "shared"), ("attention", "shared"), ("concat", "independent")],
)
def test_official_factory_sampling_and_learning(
    algorithm, fusion, ownership, monkeypatch, tmp_path
):
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    social = (
        {"kind": "none"} if fusion == "base" else {"kind": "implicit", "fusion": fusion}
    )
    if fusion == "concat":
        social["slots"] = {
            "stations": 8,
            "vehicles": 8,
            "input_jobs": 2,
            "post_jobs": 2,
        }
    policy = doc["production"]["policies"]["dqn"]
    policy["observation"]["params"] = {"social": social}
    policy["network"]["params"] = {"social": social}
    policy["parameter_sharing"] = {"default": ownership}
    options = BaseAlgorithm.model_validate(doc)
    env_config = {
        "scenario": replace(sim_with(n=2).scenario, tick_limit=12),
        "options": options,
        "algorithm": algorithm,
        "action_seed": 101,
    }
    probe = FactoryMultiAgentEnv(env_config)
    specs = set(probe.agent_modules.values())
    observations, _ = probe.reset()
    for _ in range(500):
        assert all(
            probe.observation_spaces[a].contains(obs) for a, obs in observations.items()
        )
        actions = {
            agent: int(np.flatnonzero(obs["action_mask"])[0])
            for agent, obs in observations.items()
        }
        observations, _, terminated, truncated, _ = probe.step(actions)
        if terminated["__all__"] or truncated["__all__"]:
            assert observations
            assert all(obs["action_mask"][0] for obs in observations.values())
            break
    else:
        pytest.fail("factory did not reach its fixed horizon")
    probe.close()
    training = (
        {"train_batch_size_per_learner": 64, "minibatch_size": 32, "num_epochs": 1}
        if algorithm == "ppo"
        else {
            "train_batch_size_per_learner": 32,
            "num_steps_sampled_before_learning_starts": 32,
            "replay_buffer_config": {"capacity": 1024, "alpha": 0.6, "beta": 0.4},
        }
    )
    config = build_algorithm_config(
        scenario=env_config["scenario"],
        options=options,
        algorithm=algorithm,
        seed=101,
        training=training,
        env_runners={"num_env_runners": 0, "rollout_fragment_length": 32},
        learners={"num_learners": 0, "num_gpus_per_learner": 0},
        reporting={
            "min_sample_timesteps_per_iteration": 64,
            "min_time_s_per_iteration": 0,
        },
    )
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
    )
    learner = None
    try:
        learner = config.build_algo()
        before = {
            name: {
                key: value.copy()
                for key, value in learner.get_module(name).get_state().items()
            }
            for name in specs
        }
        for _ in range(2):
            progress = learner.train()
        assert progress["physical_ticks"] > 0
        assert progress["episodes"] > 0
        saved_progress = {
            name: progress[name] for name in ("physical_ticks", "episodes")
        }
        assert any(
            not np.array_equal(value, learner.get_module(name).get_state()[key])
            for name, weights in before.items()
            for key, value in weights.items()
        )
        assert all(
            torch.isfinite(parameter).all()
            for name in specs
            for parameter in learner.get_module(name).parameters()
        )
        saved = deepcopy(learner.get_state())
        checkpoint = learner.save_to_path(str(tmp_path / "official-checkpoint"))
        learner.stop()
        learner = None
        learner = Algorithm.from_checkpoint(checkpoint)
        restored = learner.get_state()
        for name, value in saved_progress.items():
            assert learner.metrics.peek("smartsom_" + name + "_lifetime") == value
        assert_state_equal(saved["learner_group"], restored["learner_group"])
        if algorithm == "dqn":
            assert_state_equal(saved["replay_buffer"], restored["replay_buffer"])
        restored_online = {
            name: {
                key: value.copy()
                for key, value in learner.get_module(name).get_state().items()
            }
            for name in specs
        }
        continued = learner.train()
        assert continued["physical_ticks"] > saved_progress["physical_ticks"]
        assert continued["episodes"] >= saved_progress["episodes"]
        assert learner.iteration == saved["training_iteration"] + 1
        assert any(
            not np.array_equal(value, learner.get_module(name).get_state()[key])
            for name, weights in restored_online.items()
            for key, value in weights.items()
        )
        assert all(
            torch.isfinite(p).all()
            for name in specs
            for p in learner.get_module(name).parameters()
        )
    finally:
        if learner is not None:
            learner.stop()
        ray.shutdown()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_fixed_resident_evaluation_panel(algorithm, monkeypatch):
    from smartsom.learning.rllib_evaluation import configure_panel

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    )
    scenario = replace(sim_with(n=2).scenario, tick_limit=12)
    cases = [
        {
            "case_id": str(i),
            "scenario": scenario,
            "action_seed": 17 + i,
            "mode": "sample" if algorithm == "ppo" else "argmax",
        }
        for i in range(4)
    ]
    config = build_algorithm_config(
        scenario=scenario,
        options=options,
        algorithm=algorithm,
        seed=41,
        env_runners={"num_env_runners": 0},
        learners={"num_learners": 0, "num_gpus_per_learner": 0},
    )
    training_epsilon = deepcopy(getattr(config, "epsilon", None))
    config = configure_panel(config, cases, num_env_runners=2)
    assert getattr(config, "epsilon", None) == training_epsilon
    assert "epsilon" not in config.evaluation_config
    assert config.get_evaluation_config_object().explore is (algorithm == "ppo")
    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
    )
    learner = None
    try:
        learner = config.build_algo()
        first = learner.evaluate()["env_runners"]
        second = learner.evaluate()["env_runners"]
        # Published scientific rows must be identical on repeated fixed panels,
        # including stochastic action selection with the frozen action seed.
        assert first["cases"] == second["cases"]
        assert [row["case_id"] for row in first["cases"]] == ["0", "1", "2", "3"]
        assert all(row["truncated"] for row in first["cases"])
        assert first["joint_cost"] is None
        assert not first["complete_panel"]
        assert first["makespan"] is None
    finally:
        if learner is not None:
            learner.stop()
        ray.shutdown()


def test_tune_uses_official_heterogeneous_requests_and_checkpoints(
    monkeypatch, tmp_path
):
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    from smartsom.experiments.rllib_trainable import RLlibTrainable

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    )
    configs = []
    for algorithm, cpus in (("ppo", 1), ("dqn", 2)):
        config = build_algorithm_config(
            scenario=replace(sim_with(n=2).scenario, tick_limit=12),
            options=options,
            algorithm=algorithm,
            seed=101,
            training={
                "train_batch_size_per_learner": 64,
                "minibatch_size": 32,
                "num_epochs": 1,
            }
            if algorithm == "ppo"
            else {
                "train_batch_size_per_learner": 32,
                "num_steps_sampled_before_learning_starts": 32,
                "replay_buffer_config": {"capacity": 1024},
            },
            env_runners={"num_env_runners": 0, "rollout_fragment_length": 32},
            learners={"num_learners": 0, "num_gpus_per_learner": 0},
            resources={"num_cpus_for_main_process": cpus},
            reporting={
                "min_sample_timesteps_per_iteration": 64,
                "min_time_s_per_iteration": 0,
            },
        )
        configs.append(config)
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    try:
        results = tune.Tuner(
            RLlibTrainable,
            param_space={"algorithm_config": tune.grid_search(configs)},
            tune_config=tune.TuneConfig(scheduler=FIFOScheduler(), reuse_actors=False),
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="official-algorithms",
                verbose=0,
                stop={"training_iteration": 2},
                failure_config=tune.FailureConfig(max_failures=0),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1,
                    checkpoint_at_end=True,
                    num_to_keep=2,
                ),
            ),
        ).fit()
        assert len(results) == 2
        for result in results:
            assert result.error is None
            assert result.metrics["training_iteration"] == 2
            assert len(result.best_checkpoints) == 2
            with result.checkpoint.as_directory() as checkpoint:
                restored = Algorithm.from_checkpoint(checkpoint)
                try:
                    assert restored.iteration == 2
                    restored.train()
                    assert restored.iteration == 3
                finally:
                    restored.stop()

        class Interrupted(RLlibTrainable):
            def step(self):
                marker = Path(self.config["interruption_marker"])
                if self.iteration == 1 and not marker.exists():
                    marker.write_text("failed after first official checkpoint")
                    raise RuntimeError("injected interruption")
                return super().step()

            def load_checkpoint(self, checkpoint_dir):
                super().load_checkpoint(checkpoint_dir)
                assert self.algorithm.iteration == 1
                Path(self.config["restore_marker"]).write_text("restored iteration 1")

        resume_config = {
            "algorithm_config": configs[1],
            "interruption_marker": str(tmp_path / "interruption"),
            "restore_marker": str(tmp_path / "restored"),
        }
        failed = tune.Tuner(
            Interrupted,
            param_space=resume_config,
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="official-resume",
                verbose=0,
                stop={"training_iteration": 3},
                failure_config=tune.FailureConfig(max_failures=0),
                checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=1),
            ),
        ).fit()
        assert failed[0].error is not None
        assert not (tmp_path / "restored").exists()
        resumed = tune.Tuner.restore(
            str(tmp_path / "official-resume"),
            trainable=Interrupted,
            resume_errored=True,
            param_space=resume_config,
        ).fit()
        assert resumed[0].error is None
        assert resumed[0].metrics["training_iteration"] == 3
        assert (tmp_path / "restored").read_text() == "restored iteration 1"
    finally:
        ray.shutdown()


def test_official_tune_retains_best_and_latest_without_model_copies(
    monkeypatch, tmp_path
):
    from ray import tune

    from smartsom.experiments.rllib_trainable import (
        checkpoint_config,
        checkpoint_references,
        score_checkpoint,
    )

    class PanelSequence(tune.Trainable):
        """Retention fixture only; real RLlib state recovery is tested above."""

        def step(self):
            value = self.config["costs"][self.iteration]
            return score_checkpoint(
                {
                    "evaluation": {
                        "env_runners": {
                            "complete_panel": value is not None,
                            "panel_split": "periodic_evaluation",
                            "joint_cost": value,
                        }
                    }
                }
            )

        def save_checkpoint(self, checkpoint_dir):
            (Path(checkpoint_dir) / "fixture.json").write_text(
                json.dumps(
                    {
                        "iteration": self.iteration,
                    }
                )
            )
            return checkpoint_dir

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
    )
    try:
        results = tune.Tuner(
            PanelSequence,
            param_space={"costs": tune.grid_search([[4.0, 2.0, 5.0], [None] * 3])},
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="best-last",
                verbose=0,
                stop={"training_iteration": 3},
                checkpoint_config=checkpoint_config(),
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        ).fit()
        for result in results:
            assert result.error is None
            refs = checkpoint_references(result)
            with refs["last"].as_directory() as folder:
                assert (
                    json.loads((Path(folder) / "fixture.json").read_text())["iteration"]
                    == 3
                )
            if result.config["costs"][0] is None:
                assert refs["best"] is None
            else:
                with refs["best"].as_directory() as folder:
                    assert (
                        json.loads((Path(folder) / "fixture.json").read_text())[
                            "iteration"
                        ]
                        == 2
                    )
                assert refs["best"].path != refs["last"].path
    finally:
        ray.shutdown()


def test_real_rllib_skipped_and_truncated_panels_do_not_reuse_best_score(
    monkeypatch, tmp_path
):
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig

    from smartsom.experiments.rllib_trainable import (
        CHECKPOINT_SCORE,
        RLlibTrainable,
        checkpoint_config,
        checkpoint_references,
    )

    def panel_fixture(algorithm, group):
        # Only panel metrics are controlled. RLlib training, metric compilation,
        # iterations and checkpoint save/restore all execute for real.
        value = 2.0 if algorithm.iteration == 1 else None
        metrics = {
            "complete_panel": value is not None,
            "joint_cost": value,
            "panel_split": "periodic_evaluation",
        }
        for name, item in metrics.items():
            algorithm.metrics.log_value(
                ("evaluation", "env_runners", name), item, reduce="item"
            )
        return metrics, 0, 0

    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=0, rollout_fragment_length=32)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(train_batch_size_per_learner=64, minibatch_size=32, num_epochs=1)
        .evaluation(evaluation_interval=2, custom_evaluation_function=panel_fixture)
        .debugging(seed=101)
    )
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    try:
        result = tune.Tuner(
            RLlibTrainable,
            param_space={"algorithm_config": config},
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="stale-panel",
                verbose=0,
                stop={"training_iteration": 4},
                checkpoint_config=checkpoint_config(),
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        ).fit()[0]
        assert result.error is None
        assert result.metrics_dataframe[CHECKPOINT_SCORE].tolist() == [
            float("inf"),
            2.0,
            float("inf"),
            float("inf"),
        ]
        refs = checkpoint_references(result)
        for name, expected in (("best", 2), ("last", 4)):
            with refs[name].as_directory() as path:
                restored = Algorithm.from_checkpoint(path)
                try:
                    assert restored.iteration == expected
                finally:
                    restored.stop()
    finally:
        ray.shutdown()


@pytest.mark.parametrize("checkpoint_selection", [False, True])
def test_scientific_milestones_and_final_panel_finish_before_trial(
    monkeypatch, tmp_path, checkpoint_selection
):
    from ray import tune

    from smartsom.experiments.rllib_trainable import (
        CheckpointReferences,
        RLlibTrainable,
        checkpoint_config,
    )
    from smartsom.learning.rllib_evaluation import configure_panel

    root = Path(__file__).resolve().parents[2]
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((root / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    )
    scenario = replace(sim_with(n=2).scenario, tick_limit=12)
    config = build_algorithm_config(
        scenario=scenario,
        options=options,
        algorithm="ppo",
        seed=101,
        training={
            "train_batch_size_per_learner": 64,
            "minibatch_size": 32,
            "num_epochs": 1,
        },
        env_runners={"num_env_runners": 0, "rollout_fragment_length": 32},
        learners={"num_learners": 0, "num_gpus_per_learner": 0},
    )
    config = configure_panel(
        config,
        [{"case_id": "periodic", "scenario": scenario, "action_seed": 3}],
        num_env_runners=1,
        interval=None,
        test_cases=[{"case_id": "test", "scenario": scenario, "action_seed": 7}],
    )
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    try:
        result = tune.Tuner(
            RLlibTrainable,
            param_space={
                "algorithm_config": config,
                "execution": {
                    "physical_ticks": 1,
                    "final_evaluation": True,
                    "final_checkpoint_selection": ["best", "last"]
                    if checkpoint_selection
                    else [],
                    "periodic_evaluation": {
                        "unit": "physical_ticks",
                        "milestones": [0, 1, 2],
                    },
                },
                "checkpoint_reference_directory": str(tmp_path / "references"),
            },
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="scientific-boundaries",
                verbose=0,
                stop={"training_iteration": 5},
                checkpoint_config=checkpoint_config(),
                callbacks=[CheckpointReferences()],
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        ).fit()[0]
        assert result.error is None
        assert result.metrics["training_iteration"] == (
            3 if checkpoint_selection else 2
        )  # initial panel, then one official iteration
        assert result.metrics["physical_ticks"] >= 2
        final = result.metrics["final_evaluation"]
        if checkpoint_selection:
            assert final["best"] == {"status": "no_eligible_checkpoint"}
            final = final["last"]
            assert final["checkpoint"]["algorithm_iteration"] == 1
            committed = json.loads(Path(final["checkpoint"]["references"]).read_text())
            assert Path(committed["last"]["path"]).is_dir()
            assert committed["last"]["path"] == result.checkpoint.path
        assert final["panel_split"] == "test"
        assert final["cases"][0]["case_id"] == "test"
        reports = [
            json.loads(line)
            for line in (Path(result.path) / "result.json").read_text().splitlines()
        ]
        periodic = reports[1]["evaluation"]["env_runners"]
        assert periodic["scheduled_milestones"] == [1, 2]
        assert (
            periodic["actual_progress"]["physical_ticks"]
            == result.metrics["physical_ticks"]
        )
        with result.checkpoint.as_directory() as folder:
            state = json.loads((Path(folder) / "smartsom-runtime.json").read_text())
            assert state["phase"] == "complete"
            assert state["evaluated_milestones"] == [0, 1, 2]
            restored = Algorithm.from_checkpoint(folder)
            try:
                assert restored.iteration == 1
                assert (
                    restored.metrics.peek("smartsom_physical_ticks_lifetime")
                    == result.metrics["physical_ticks"]
                )
            finally:
                restored.stop()
    finally:
        ray.shutdown()


@pytest.mark.parametrize("search_mode", ["fixed", "censored", "not_evaluated"])
def test_public_cli_runs_official_rllib_trial(tmp_path, search_mode):
    import subprocess
    import sys

    from tests.unit.test_algorithm_components import author

    root = Path(__file__).resolve().parents[2]
    document = yaml.safe_load(
        (root / "configs/algorithms/v2-base-ddqn.yaml").read_text()
    )
    document["production"]["policies"]["dqn"]["learner"] = {
        "target": "smartsom.learning.rllib_config.ppo_config",
        "params": {
            "lr": 0.0003,
            "train_batch_size_per_learner": 64,
            "minibatch_size": 32,
            "num_epochs": 1,
        },
    }
    path = author(tmp_path, document)
    ex = yaml.safe_load(path.read_text())
    scenario = yaml.safe_load(Path(ex["scenario"]).read_text())
    scenario["jobs"]["orders"].update(total=2, initial=1)
    scenario["jobs"]["arrivals"]["mean_interval_ticks"] = 1.0
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    ex["scenario"] = "scenario.yaml"
    ex.update(
        drain_frozen=True,
        drain_ticks=12,
        train_ticks=1,
        train_seeds=[101],
        final_evaluation=False,
        periodic_evaluation={
            "milestones": [0]
            if search_mode == "censored"
            else [1]
            if search_mode == "not_evaluated"
            else []
        },
        periodic_evaluation_cases=4,
        test_cases=4,
        run={
            "task": "train",
            "seconds": 0.01 if search_mode == "not_evaluated" else None,
            "seed": 101,
            "output": str(tmp_path / "run"),
        },
    )
    path.write_text(yaml.safe_dump(ex))
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "resources": {"cpus": 1 if search_mode == "fixed" else 2},
                "output_root": str(tmp_path / "outputs"),
                "rllib": {"env_runners": {"num_env_runners": 0}},
                "tracking": {"mode": "disabled"},
            }
        )
    )
    search_path = tmp_path / "search.yaml"
    search_path.write_text(
        yaml.safe_dump(
            {
                "algorithm": "optuna",
                "num_samples": 1,
                "seed": 7,
                "train_seed": 101,
                "objectives": [
                    {
                        "panel": "sample",
                        "metric": "configured.all." + metric,
                        "mode": direction,
                    }
                    for metric, direction in (
                        ("makespan", "min"),
                        ("mean_tardiness", "min"),
                        ("final_passing_rate", "max"),
                    )
                ],
                "parameters": {
                    "algorithm.production.policies.dqn.learner.params.lr": {
                        "distribution": "choice",
                        "values": [0.0003],
                    }
                },
            }
        )
    )
    command = [sys.executable, "-m", "smartsom.experiments.cli"]
    for action in ("check", "run"):
        completed = subprocess.run(
            command
            + [action, str(path), "--profile", str(profile), "--progress", "off"]
            + (
                ["--search", str(search_path)]
                if action == "run" and search_mode != "fixed"
                else []
            ),
            capture_output=True,
            text=True,
            timeout=120,
        )
        (tmp_path / (action + ".log")).write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = [
        (path.parent, json.loads(path.read_text()))
        for path in tmp_path.rglob("batch.json")
    ]
    if search_mode == "fixed":
        ((run_root, report),) = [
            (p, r) for p, r in reports if "algorithm_iteration" in r
        ]
    else:
        ((template, search_report),) = [
            (p, r) for p, r in reports if "pareto_candidates" in r
        ]
        assert search_report["status"] == "completed"
        assert search_report["pareto_candidates"] == []
        report = search_report["trials"][0]["metrics"]
        assert report["search_evaluation"]["status"] == search_mode
        (run_root,) = [
            p.parent for p in (template / "trials").glob("*/suggestion.json")
        ]
    assert report["status"] == "completed"
    assert report["algorithm_iteration"] == (0 if search_mode == "not_evaluated" else 1)
    assert report["physical_ticks"] >= (0 if search_mode == "not_evaluated" else 1)
    refs = list(run_root.glob("checkpoint-refs-*.json"))
    assert len(refs) == 1
    last = json.loads(refs[0].read_text())["last"]
    assert (Path(last["path"]) / "smartsom-runtime.json").is_file()


def test_final_best_and_last_load_distinct_official_weights(monkeypatch, tmp_path):
    from hashlib import sha256

    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig

    from smartsom.experiments.rllib_trainable import (
        CheckpointReferences,
        RLlibTrainable,
        checkpoint_config,
    )
    from smartsom.learning.rllib_evaluation import FixedPanels

    class WeightPanel(FixedPanels):
        def __call__(self, algorithm, group):
            digest = sha256()
            for key, value in sorted(algorithm.get_module().get_state().items()):
                digest.update(key.encode())
                digest.update(np.asarray(value).tobytes())
            metrics = {
                "complete_panel": True,
                "joint_cost": float(algorithm.iteration + 1),
                "panel_split": "test" if self.final else "periodic_evaluation",
                "weights_sha256": digest.hexdigest(),
            }
            for name, value in metrics.items():
                algorithm.metrics.log_value(
                    ("evaluation", "env_runners", name), value, reduce="item"
                )
            return metrics, 0, 0

    class TwoIterations(RLlibTrainable):
        def _budget_done(self, result):
            return self.algorithm.iteration >= 2

    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=0, rollout_fragment_length=32)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(train_batch_size_per_learner=64, minibatch_size=32, num_epochs=1)
        .evaluation(
            evaluation_interval=1,
            custom_evaluation_function=WeightPanel([], "periodic_evaluation", [{}]),
        )
        .debugging(seed=101)
    )
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=100 * 1024 * 1024,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    try:
        result = tune.Tuner(
            TwoIterations,
            param_space={
                "algorithm_config": config,
                "checkpoint_reference_directory": str(tmp_path / "references"),
                "execution": {
                    "final_evaluation": True,
                    "final_checkpoint_selection": ["best", "last"],
                },
            },
            run_config=tune.RunConfig(
                storage_path=str(tmp_path),
                name="weight-selection",
                verbose=0,
                checkpoint_config=checkpoint_config(),
                callbacks=[CheckpointReferences()],
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        ).fit()[0]
        assert result.error is None
        reports = [
            json.loads(line)
            for line in (Path(result.path) / "result.json").read_text().splitlines()
        ]
        expected = [
            r["evaluation"]["env_runners"]["weights_sha256"] for r in reports[:2]
        ]
        assert expected[0] != expected[1]
        final = result.metrics["final_evaluation"]
        assert final["best"]["weights_sha256"] == expected[0]
        assert final["last"]["weights_sha256"] == expected[1]
        references = json.loads(
            Path(final["last"]["checkpoint"]["references"]).read_text()
        )
        for selection, iteration in (("best", 1), ("last", 2)):
            restored = Algorithm.from_checkpoint(references[selection]["path"])
            try:
                assert restored.iteration == iteration
                digest = sha256()
                for key, value in sorted(restored.get_module().get_state().items()):
                    digest.update(key.encode())
                    digest.update(np.asarray(value).tobytes())
                assert digest.hexdigest() == expected[iteration - 1]
            finally:
                restored.stop()
    finally:
        ray.shutdown()


@pytest.mark.parametrize("task", ["calibrate", "evaluate"])
def test_calibration_tune_checkpoint_restores_receipts_without_repeating_cases(
    tmp_path, monkeypatch, task
):
    """Scientific calibration shares Tune recovery, without a model/optimizer."""
    import shutil

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import control, reservation_runtime
    from smartsom.experiments.reservation import allocate
    from smartsom.experiments.rllib_execution import trial_config
    from smartsom.experiments.rllib_trainable import RLlibTrainable

    repository = Path(__file__).resolve().parents[2]
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-experiment/v1",
                "scenario": str(repository / "configs/scenarios/small.yaml"),
                "algorithm": str(repository / "configs/algorithms/rllib-base-ppo.yaml"),
                "run": {
                    "task": task,
                    "methods": ["rule", "random"],
                    "calibration_cases": 1,
                    "seconds": 120,
                },
                "drain_frozen": True,
                "test_cases": 4,
                "periodic_evaluation_cases": 4,
                "tracking": {"mode": "disabled"},
            }
        )
    )
    plan = compile_reservation(
        path,
        sets=[
            "scenario.jobs.orders.total=2",
            "scenario.jobs.orders.initial=1",
            "scenario.jobs.arrivals.mean_interval_ticks=4",
            "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        ],
    )
    root = allocate(plan, root=tmp_path / "run", announce=False)
    config = trial_config(root)
    if task == "evaluate":
        # Exercise local failure injection at receipt boundaries. The public
        # comparator test separately runs the real Ray placement-group workers.
        from smartsom.experiments import parallel_evaluation

        case_results = parallel_evaluation.case_results

        def local_cases(settings, tasks, prepared, **kwargs):
            return case_results(
                {**settings, "worker_resources": None},
                tasks,
                prepared,
                workers=1,
                on_progress=kwargs["on_progress"],
            )

        monkeypatch.setattr(parallel_evaluation, "case_results", local_cases)
    original_commit = reservation_runtime.commit_case

    def interrupt_after_commit(*args):
        original_commit(*args)
        raise control.StopRequested()

    monkeypatch.setattr(reservation_runtime, "commit_case", interrupt_after_commit)
    worker = RLlibTrainable(config=config)
    partial = tmp_path / "partial"
    partial.mkdir()
    try:
        assert worker.train()["control_boundary"] is True
        assert worker.algorithm is None
        worker.save_checkpoint(str(partial))
    finally:
        worker.stop()
    receipts = list((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 1
    original_receipt = receipts[0].read_bytes()
    shutil.rmtree(root / "case-commits")
    monkeypatch.setattr(reservation_runtime, "commit_case", original_commit)
    episode = reservation_runtime.episode
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[4], args[5]))
        return episode(*args, **kwargs)

    monkeypatch.setattr(reservation_runtime, "episode", counted)
    resumed = RLlibTrainable(config=config)
    complete = tmp_path / "complete"
    complete.mkdir()
    try:
        resumed.load_checkpoint(str(partial))
        assert receipts[0].read_bytes() == original_receipt
        assert resumed.train()["done"] is True
        assert len(calls) == 7
        report_name = "calibration.json" if task == "calibrate" else "summary.json"
        report = json.loads((root / report_name).read_text())
        if task == "calibrate":
            assert report["status"] == "frozen"
            assert len(report["results"]) == 8
            assert {r["method"] for r in report["results"]} == {"rule", "random"}
            assert report["calibration_identity"]
        else:
            assert set(report) == {"rule", "random"}
            assert report["rule"]["all"]["cases"] == 4
            assert len((root / "cases.jsonl").read_text().splitlines()) == 8
        resumed.save_checkpoint(str(complete))
    finally:
        resumed.stop()
    (root / report_name).unlink()
    if task == "evaluate":
        (root / "cases.jsonl").unlink()
        (root / "research-summary.json").unlink()
    shutil.rmtree(root / "case-commits")
    restored = RLlibTrainable(config=config)
    try:
        restored.load_checkpoint(str(complete))
        assert restored.train()["done"] is True
        assert len(calls) == 7  # Completed calibration does not run again.
        assert json.loads((root / report_name).read_text()) == report
        if task == "evaluate":
            assert len((root / "cases.jsonl").read_text().splitlines()) == 8
            assert (root / "research-summary.json").is_file()
        assert len(list((root / "case-commits").glob("case-*.json"))) == 8
        assert not (root / "last.pt").exists()
        assert restored.algorithm is None
        payload_path = complete / "scientific-task.json"
        payload = json.loads(payload_path.read_text())
        if task == "evaluate":
            incomplete = json.loads(payload_path.read_text())
            incomplete["files"].pop("cases.jsonl")
            incomplete["file_sha256"].pop("cases.jsonl")
            payload_path.write_text(json.dumps(incomplete))
            with pytest.raises(ValueError, match="missing completed results"):
                restored.load_checkpoint(str(complete))
            assert json.loads((root / report_name).read_text()) == report
        payload["files"][report_name] = "{}"
        payload_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="checksum"):
            restored.load_checkpoint(str(complete))
        assert json.loads((root / report_name).read_text()) == report
    finally:
        restored.stop()
