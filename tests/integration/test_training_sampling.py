"""Real ordered sampling, exact quotas, and complete recovery across three PPOs."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_training_run
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls
from smartsom.experiments.training_lifecycle import inspect_resume_checkpoint

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def small_training(name, output):
    resolved = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
    spec = resolved.algorithm.algorithm
    parameters = spec.parameters.model_copy(
        update={
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 2,
            "hidden_sizes": (32, 32),
        }
    )
    return replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={"algorithm": spec.model_copy(update={"parameters": parameters})}
        ),
        run=resolved.run.model_copy(
            update={
                "budget": resolved.run.budget.model_copy(
                    update={"environment_steps": 128}
                ),
                "output_root": str(output),
            }
        ),
    )


def assert_state_equal(left, right):
    import numpy as np
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def sampled(request, tmp_path_factory):
    packages = [
        "torch",
        "gymnasium",
        "sb3_contrib" if request.param == "sb3" else "ray",
    ]
    if request.param == "marl":
        packages.append("pettingzoo")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        required = (
            "SMARTSOM_REQUIRE_MARL"
            if request.param == "marl"
            else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required learning extras missing: {missing}")
        pytest.skip(f"optional learning extras missing: {missing}")
    resolved = small_training(
        request.param, tmp_path_factory.mktemp(f"streams-{request.param}")
    )
    results = []
    for processes in (0, 2):
        controls = TrainingControls(
            checkpoint_every_updates=1,
            validation=None,
            num_envs=2,
            sampling_processes=processes,
        )
        continuous = train_one(resolved, controls=controls)
        partial = train_one(resolved, controls=replace(controls, stop_after_updates=2))
        resumed = train_one(
            resolved, controls=replace(controls, resume_from=partial.last_checkpoint)
        )
        results.append((controls, continuous, partial, resumed))
    return request.param, resolved, results


def test_two_stream_resume_matches_weights_optimizer_rng_and_episode_prefix(sampled):
    from smartsom.learning.training_state import load_state

    name, _, results = sampled
    for _, continuous, partial, resumed in results:
        assert partial.status == "interrupted" and partial.environment_steps == 64
        assert resumed.status == "completed" and resumed.environment_steps == 128
        manifests = [
            json.loads((r.checkpoint_dir / "checkpoint.json").read_text())
            for r in (continuous, resumed)
        ]
        assert (
            manifests[0]["final_weights_sha256"] == manifests[1]["final_weights_sha256"]
        )
        assert (continuous.run_dir / "episodes.jsonl").read_bytes() == (
            resumed.run_dir / "episodes.jsonl"
        ).read_bytes()
        summaries = [
            json.loads((r.last_checkpoint / "state_summary.json").read_text())
            for r in (continuous, resumed)
        ]
        assert summaries[0] == summaries[1]
        assert summaries[0]["environment_steps"] == 128
        assert summaries[0]["learner_updates"] == (8 if name == "sb3" else 4)
        assert_state_equal(
            *(load_state(r.last_checkpoint / "rng.pkl") for r in (continuous, resumed))
        )
        assert inspect_resume_checkpoint(resumed.last_checkpoint)["ppo_updates"] == 4
        report = audit_training(resumed.run_dir)
        assert report["environment_steps"] == 128 and report["num_envs"] == 2
        assert report["ppo_updates"] == 4 and report["budget_completed"]
        partial_report = audit_training(partial.run_dir)
        assert (
            partial_report["environment_steps"] == 64
            and not partial_report["budget_completed"]
        )
        rows = [
            json.loads(line)
            for line in (resumed.run_dir / "episodes.jsonl").read_text().splitlines()
        ]
        for stream in range(2):
            members = [row for row in rows if row["stream_id"] == stream]
            assert [row["local_episode"] for row in members] == list(
                range(len(members))
            )
            assert sum(len(row["steps"]) for row in members) == 64
        if name == "marl":
            assert report["agent_steps"] == 128 * 12


def test_process_completion_order_does_not_change_training(sampled):
    _, _, results = sampled
    left, right = results[0][1], results[1][1]
    assert (left.run_dir / "episodes.jsonl").read_bytes() == (
        right.run_dir / "episodes.jsonl"
    ).read_bytes()
    a, b = [
        json.loads((r.checkpoint_dir / "checkpoint.json").read_text())
        for r in (left, right)
    ]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert (left.last_checkpoint / "state_summary.json").read_bytes() == (
        right.last_checkpoint / "state_summary.json"
    ).read_bytes()


def test_full_resume_rejects_changed_stream_execution_identity(sampled):
    _, resolved, results = sampled
    controls, _, partial, _ = results[0]
    before = set(Path(resolved.run.output_root).iterdir())
    with pytest.raises(ValueError, match="identical source"):
        train_one(
            resolved,
            controls=replace(
                controls, resume_from=partial.last_checkpoint, sampling_processes=2
            ),
        )
    assert set(Path(resolved.run.output_root).iterdir()) == before
