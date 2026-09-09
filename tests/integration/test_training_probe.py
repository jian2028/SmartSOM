"""Doctor startup uses each actual backend factory without sampling or updating."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from test_training_sampling import assert_state_equal

from smartsom.config import resolve_training_run
from smartsom.config.codec import primitive
from smartsom.experiments.training_controls import TrainingControls
from smartsom.experiments.training_probe import probe_training_backend

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


@pytest.mark.parametrize("name", ["sb3", "rllib", "marl"])
@pytest.mark.parametrize("num_envs,processes", [(1, 0), (2, 2)])
def test_actual_backend_probe_keeps_budget_rng_and_zero_training_counts(
    name, num_envs, processes, tmp_path
):
    packages = ["torch", "gymnasium", "sb3_contrib" if name == "sb3" else "ray"]
    if name == "marl":
        packages.append("pettingzoo")
    missing = [
        package for package in packages if importlib.util.find_spec(package) is None
    ]
    if missing:
        required = (
            "SMARTSOM_REQUIRE_MARL" if name == "marl" else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required learning extras missing: {missing}")
        pytest.skip(f"optional learning extras missing: {missing}")
    import torch

    from smartsom.learning.training_state import rng_state

    resolved = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
    controls = TrainingControls(
        validation=None,
        num_envs=num_envs,
        sampling_processes=processes,
        numerical_threads=1,
    )
    before, threads = rng_state(), torch.get_num_threads()
    report = probe_training_backend(resolved, controls, output_root=tmp_path)
    assert report["status"] == "passed"
    assert report["sampled_steps"] == report["learner_updates"] == 0
    assert report["device"] == "cpu" and report["numerical_threads"] == 1
    assert report["num_envs"] == num_envs and report["sampling_processes"] == processes
    assert report["configured_budget"] == primitive(resolved.run.budget)
    assert report["configured_budget"]["environment_steps"] == (
        1024 if name == "sb3" else 4096
    )
    assert all(count > 0 for count in report["policy_parameters"].values())
    assert json.loads((Path(report["run_dir"]) / "probe.json").read_text()) == report
    assert not (Path(report["run_dir"]) / "unused-checkpoint").exists()
    assert_state_equal(before, rng_state())
    assert torch.get_num_threads() == threads
    if name != "sb3":
        import ray

        assert not ray.is_initialized()
