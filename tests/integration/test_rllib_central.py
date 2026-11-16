"""PR42 semantics through official single-agent PPO and compact candidates."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    generate_scenario,
    load_inputs,
)
from smartsom.controllers.e1a.network import Network, pack
from smartsom.controllers.e1a.reward import create_current_reward
from smartsom.learning.rllib_central_env import CentralFactoryEnv, CentralVectorEnv
from smartsom.learning.rllib_central_models import CentralCandidateHead
from smartsom.learning.rllib_config import build_algorithm_config

ROOT = Path(__file__).resolve().parents[2]


def central_inputs(size="small", *, orders=None, drain=12):
    ex, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/current/{size}/morris-ppo/experiment.yaml"
    )
    if orders is not None:
        workload = workload.model_copy(update={"orders": orders, "initial_orders": 1})
    return generate_scenario(factory, workload, 101, "normal", drain), options


def test_scaled_pr42_reward_is_not_normalized_j():
    reward = create_current_reward(orders=10, quality_penalty=1000, scale=0.001)
    transition = SimpleNamespace(outcomes={"overdue_time": 30, "failed": 2})
    assert reward(transition)["team"] == pytest.approx(-0.203)
    scenario, options = central_inputs(orders=2)
    env = CentralFactoryEnv({"scenario": scenario, "options": options})
    try:
        assert not env.episode.normalized_reward
    finally:
        env.close()


def test_dense_candidates_preserve_original_scores_and_gradients():
    scenario, options = central_inputs("medium")
    env = CentralFactoryEnv({"scenario": scenario, "options": options})
    try:
        obs, _ = env.reset(seed=101)
        request = env.episode.frame.request
        assert len(request.candidates) > 10000
        assert env.observation_space.contains(obs)
        head = CentralCandidateHead(
            env.observation_space,
            {"advance_only_when_idle": True, "owner_count": len(env.owners)},
        )
        original = Network(False, advance_only_when_idle=True)
        original.actor(pack([request], critics=None, options=options))
        original.load_state_dict(head.model.state_dict())
        batch = {
            key: torch.from_numpy(value).unsqueeze(0) for key, value in obs.items()
        }
        outputs = []
        for model, logits in (
            (original, original.actor(pack([request], critics=None, options=options))),
            (head.model, head(batch)),
        ):
            assert torch.isneginf(logits[0, -1])  # ADVANCE beside work.
            assert torch.isfinite(logits[0, :-1]).all()
            distribution = torch.distributions.Categorical(logits=logits)
            (
                distribution.log_prob(torch.tensor([0])) + distribution.entropy()
            ).sum().backward()
            outputs.append((logits, [p.grad for p in model.parameters()]))
        torch.testing.assert_close(outputs[0][0], outputs[1][0])
        for left, right in zip(outputs[0][1], outputs[1][1], strict=True):
            torch.testing.assert_close(left, right)
    finally:
        env.close()


def test_advance_is_one_decision_and_truncation_has_real_global_state():
    scenario, options = central_inputs(orders=2, drain=3)
    env = CentralFactoryEnv({"scenario": scenario, "options": options})
    try:
        obs, info = env.reset(seed=101)
        while True:
            request = env.episode.frame.request
            tick = info["physical_tick"]
            # First work action; ADVANCE is last and legal only when alone.
            action = 0
            advance = request.candidates[action].kind == "advance"
            if len(request.candidates) > 1:
                with pytest.raises(ValueError, match="ADVANCE is masked"):
                    env.step(len(request.candidates) - 1)
            waits = env.episode.wait_steps
            obs, reward, done, truncated, info = env.step(action)
            assert env.episode.wait_steps - waits == int(advance)
            assert (info["physical_tick"] > tick) == advance
            assert env.observation_space.contains(obs)
            if done or truncated:
                assert truncated and not done
                assert info["factory_metrics"]["joint_cost"] is None
                assert np.isfinite(obs["critic"]).all()
                assert len(obs["valid"]) > 0
                break
    finally:
        env.close()


def test_vector_adapter_preserves_ragged_candidate_rows():
    scenario, options = central_inputs(orders=2)
    env = CentralVectorEnv({"scenario": scenario, "options": options, "num_envs": 1})
    try:
        obs, _ = env.reset(seed=101)
        assert obs["candidates"].shape[0] == 1
        assert obs["associations"].shape[:2] == obs["candidates"].shape[:2]
        assert obs["valid"].shape == obs["candidates"].shape[:2]
    finally:
        env.close()


def test_public_configuration_rejects_mismatched_critic_and_central_dqn():
    scenario, options = central_inputs(orders=2)
    with pytest.raises(ValueError, match="requires official PPO"):
        build_algorithm_config(
            scenario=scenario, options=options, algorithm="dqn", seed=1
        )
    data = yaml.safe_load(
        (ROOT / "configs/algorithms/current/small/morris-ppo.yaml").read_text()
    )
    data["production"]["policies"]["ppo"]["critic_observation"]["params"] = {
        "time_scale": 300
    }
    options = BaseAlgorithm.model_validate(data)
    with pytest.raises(ValueError, match="same public global observation scales"):
        build_algorithm_config(
            scenario=scenario, options=options, algorithm="ppo", seed=1
        )


def test_public_morris_ppo_panels_best_last_and_official_restore(tmp_path, monkeypatch):
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    from tests.helpers.rllib_runtime import author_runtime, command, last_checkpoint
    from tests.integration.test_rllib_dqn_checkpoint import assert_state_equal

    path, profile = author_runtime(tmp_path, evaluation=True)
    profile_data = yaml.safe_load(profile.read_text())
    profile_data["resources"]["cpus"] = 2
    profile.write_text(yaml.safe_dump(profile_data))
    data = yaml.safe_load(
        (ROOT / "configs/algorithms/current/small/morris-ppo.yaml").read_text()
    )
    data["production"]["policies"]["ppo"]["learner"]["params"].update(
        train_batch_size_per_learner=64, minibatch_size=32, num_epochs=1
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(data))
    command("check", path, "--profile", profile)
    command("run", path, "--profile", profile, "--progress", "off", timeout=180)
    root = tmp_path / "run"
    report = json.loads((root / "batch.json").read_text())
    assert report["status"] == "completed"
    refs = json.loads(next(root.glob("checkpoint-refs-*.json")).read_text())
    assert refs["best"] is not None
    for selection in ("best", "last"):
        panel = report["final_evaluation"][selection]
        assert panel["panel_split"] == "test"
        assert panel["selection_mode"] == "sample"
        assert panel["complete_panel"]
        assert len(panel["cases"]) == 4
        assert len({row["case_id"] for row in panel["cases"]}) == 4
        assert Path(refs[selection]["path"]).is_dir()
    checkpoint = last_checkpoint(root)
    assert (
        json.loads((checkpoint / "smartsom-runtime.json").read_text())["phase"]
        == "complete"
    )
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
    )
    restored = None
    try:
        restored = Algorithm.from_checkpoint(checkpoint)
        before = restored.learner_group.get_state()
        iteration = restored.iteration
        assert iteration == refs["last"]["algorithm_iteration"]
        saved = restored.save_to_path(str(tmp_path / "roundtrip"))
        restored.restore_from_path(saved)
        assert_state_equal(before, restored.learner_group.get_state())
        weights = {
            key: value.copy()
            for key, value in restored.get_module().get_state().items()
            if isinstance(value, np.ndarray)
        }
        restored.train()
        assert restored.iteration == iteration + 1
        assert any(
            not np.array_equal(value, restored.get_module().get_state()[key])
            for key, value in weights.items()
        )
        assert all(torch.isfinite(p).all() for p in restored.get_module().parameters())
    finally:
        if restored is not None:
            restored.stop()
        ray.shutdown()
