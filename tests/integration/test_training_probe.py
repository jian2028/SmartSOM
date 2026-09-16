"""Doctor startup uses each actual backend factory without sampling or updating."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from test_training_sampling import assert_state_equal

from smartsom.api import load_config, prepare
from smartsom.config.codec import digest, primitive
from smartsom.config.extensions import ExtensionRef, ExtensionSpec
from smartsom.experiments.training_controls import TrainingControls
from smartsom.experiments.training_probe import probe_training_backend

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def require_backend(name):
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


@pytest.mark.parametrize("name", ["sb3", "rllib", "marl"])
@pytest.mark.parametrize("num_envs,processes", [(1, 0), (2, 2)])
def test_actual_backend_probe_keeps_budget_rng_and_zero_training_counts(
    name, num_envs, processes, tmp_path
):
    require_backend(name)
    import torch

    from smartsom.learning.training_state import rng_state

    config = load_config(ROOT / f"configs/runs/learning_{name}.yaml")
    resolved = prepare(config)
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
    assert report["topology_semantics"] == "configured"
    assert report["algorithm"] == primitive(resolved.resolved.algorithm)
    identity = report["identity"]
    assert identity["runtime"]["num_envs"] == num_envs
    assert identity["runtime"]["sampling_processes"] == processes
    assert len(identity["recipe"]) == 64
    from smartsom.learning.checkpoint import file_hash

    source_files = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in sorted((ROOT / "src").rglob("*.py"))
    }
    source_files.update(
        {name: file_hash(ROOT / name) for name in ("pyproject.toml", "uv.lock")}
    )
    assert identity["source"] == digest(source_files)
    assert report["configured_budget"] == primitive(config.training)
    assert report["configured_budget"]["total_steps"] == (
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


def test_probe_records_bound_extension_identity(tmp_path):
    require_backend("sb3")
    config = load_config(ROOT / "configs/runs/learning_sb3.yaml")
    author_spec = ExtensionSpec(
        observation=ExtensionRef(name="builtin.dict", version="1")
    )
    config.algorithm.extensions = author_spec
    resolved = prepare(config)
    report = probe_training_backend(
        resolved, TrainingControls(validation=None), output_root=tmp_path
    )
    observed = report["algorithm"]["extensions"]["observation"]
    assert observed["name"] == "builtin.dict" and observed["version"] == "1"
    assert len(observed["code_sha256"]) == 64
    assert author_spec.observation.code_sha256 is None
    assert report["sampled_steps"] == report["learner_updates"] == 0
    assert json.loads((Path(report["run_dir"]) / "probe.json").read_text()) == report


@pytest.mark.parametrize(
    "backend_fails,metadata_fails,close_fails",
    [
        (True, True, True),
        (True, False, True),
        (False, False, True),
        (False, True, False),
    ],
)
def test_probe_preserves_primary_failure_and_closes_environment(
    backend_fails, metadata_fails, close_fails, monkeypatch, tmp_path
):
    require_backend("sb3")
    from smartsom.experiments import training_probe
    from smartsom.learning import production
    from smartsom.learning.production_env import ProductionEnv

    backend_error = RuntimeError("original backend failure")
    metadata_error = OSError("metadata write failed")
    cleanup_error = OSError("environment cleanup failed")
    closed = []
    original_close = ProductionEnv.close

    def backend(scenario, algorithm, directory, *, cleanup, **kwargs):
        env = ProductionEnv(scenario, algorithm.max_jobs)
        cleanup.callback(env.close)
        if backend_fails:
            raise backend_error
        return {"sampled_steps": 0, "learner_updates": 0}

    def close(env):
        closed.append(env)
        original_close(env)
        if close_fails:
            raise cleanup_error

    def failing_write(*args, **kwargs):
        raise metadata_error

    monkeypatch.setattr(production, "_train_sb3", backend)
    monkeypatch.setattr(ProductionEnv, "close", close)
    if metadata_fails:
        monkeypatch.setattr(training_probe, "write_json", failing_write)
    resolved = prepare(load_config(ROOT / "configs/runs/learning_sb3.yaml"))
    with pytest.raises((RuntimeError, OSError)) as caught:
        probe_training_backend(
            resolved, TrainingControls(validation=None), output_root=tmp_path
        )
    expected = (
        backend_error
        if backend_fails
        else cleanup_error
        if close_fails
        else metadata_error
    )
    assert caught.value is expected
    assert len(closed) == 1
    notes = "\n".join(getattr(caught.value, "__notes__", []))
    if backend_fails and close_fails:
        assert "backend environment cleanup also failed" in notes
    if metadata_fails:
        assert "probe failure metadata also could not be saved" in notes
    else:
        report = json.loads(next(tmp_path.glob("probe-*/probe.json")).read_text())
        assert report["status"] == "failed" and report["reason"] == str(expected)
