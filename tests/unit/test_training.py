"""Scientific input preparation and checkpoint preflight without learner imports."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from smartsom.config import resolve_training_run
from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.models import (
    AlgorithmFile,
    EpisodeBudget,
    LearningAlgorithm,
    TrainingRunSpec,
)
from smartsom.config.training import episode_root, framework_seed
from smartsom.experiments.cli import main
from smartsom.experiments.evidence import write_json
from smartsom.learning.checkpoint import (
    CheckpointFile,
    CheckpointManifest,
    file_hash,
    resolve_checkpoint_reference,
    structural_identity,
    validate_checkpoint,
)
from smartsom.learning.checkpoint import require_backend as real_require_backend

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_optional_framework_required(monkeypatch):
    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )


def training(provider="sb3"):
    return resolve_training_run(ROOT / f"configs/runs/learning_{provider}.yaml")


def test_micro_is_exact_frozen_prefix_and_no_physical_conversion():
    r = training()
    source = json.loads(
        (ROOT / "data/reference/idetc/converted/S00/workload.json").read_text()
    )
    assert len(r.base.workload.operations) == 9
    jobs = sorted(
        (j for o in source["workload"]["orders"] for j in o["jobs"]),
        key=lambda j: j["job_id"],
    )[:4]
    assert [primitive(j) for o in r.base.workload.orders for j in o.jobs] == jobs
    assert len(r.base.factory.machines) == 8 and len(r.base.factory.transport.agvs) == 4
    assert r.run.budget.environment_steps == 1024
    assert r.base.machine_events is None and r.base.processing_times is None


def test_episode_seed_golden_pairing_and_fixed_base():
    sb, rl = training(), training("rllib")
    assert framework_seed(101, "sb3.maskable_ppo") == 1572694888
    assert framework_seed(101, "rllib.ppo") == 189546015
    expected = (
        "4613d7d5295299e51f2d1951284b07a981664077f0116c473f9703333f98485b",
        "23c4664bc81bf02f5f41c4a95309256417f0acf4ab7d1165eb4bbfe975bb6e6f",
    )
    for i in range(5):
        a, b = sb.episode(i), rl.episode(i)
        assert a == b and a.input.workload is sb.base.workload
        assert a.root_seed == episode_root(101, i)
        if i < 2:
            assert a.input_sha256 == expected[i]
        assert sum(j.release_at == 0 for j in a.input.arrivals.jobs) == 2
        assert all(
            j.reveal_at == j.release_at and 0 <= j.release_at <= 20
            for j in a.input.arrivals.jobs
        )
        assert {s.domain for s in a.seeds if s.consumed} == {"demand", "quality"}
    assert sb.episode(0).input.quality != sb.episode(1).input.quality
    with pytest.raises(FrozenInstanceError):
        sb.framework_seed = 1


def test_validate_training_does_not_simulate_or_create_output(
    tmp_path, monkeypatch, capsys
):
    r = training()
    config = primitive(r.run) | {"output_root": str(tmp_path / "absent")}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "smartsom.engine.Simulator",
        lambda *a, **k: pytest.fail("validate created Simulator"),
    )
    assert main(["validate", str(path)]) == 0
    assert "valid training" in capsys.readouterr().out
    assert not (tmp_path / "absent").exists()
    assert main(["run", str(path)]) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"environment_steps": 0},
        {"environment_steps": True},
        {"max_ticks": 1.5},
        {"max_decisions": False},
        {"solver_time_limit_seconds": 60},
    ],
)
def test_invalid_training_budgets(changes):
    data = primitive(training().run)
    data["budget"].update(changes)
    with pytest.raises(ValueError):
        TrainingRunSpec.model_validate_json(json.dumps(data))


def bundle(tmp_path):
    training_run = training()
    p = tmp_path / "checkpoint"
    p.mkdir()
    (p / "model.zip").write_bytes(b"fixture for manifest validation only")
    s = training_run.algorithm.algorithm
    manifest = CheckpointManifest(
        schema="smartsom.checkpoint/v1",
        provider=s.provider,
        projection=s.projection,
        parameters=s.parameters,
        structure_sha256=structural_identity(training_run.base, s.projection),
        files=(CheckpointFile(path="model.zip", sha256=file_hash(p / "model.zip")),),
        dependencies=(),
        initial_weights_sha256="1" * 64,
        final_weights_sha256="2" * 64,
        environment_steps=1024,
        learner_updates=4,
        framework_seed=training_run.framework_seed,
    )
    write_json(p / "checkpoint.json", manifest)
    algorithm = training_run.algorithm.model_copy(
        update={"algorithm": s.model_copy(update={"checkpoint": str(p)})}
    )
    algorithm = resolve_checkpoint_reference(algorithm, tmp_path / "algorithm.yaml")
    resolved = replace(
        training_run.base,
        algorithm=algorithm,
        run=training_run.base.run.model_copy(update={"budget": EpisodeBudget()}),
    )
    return resolved, p


def test_checkpoint_preflight_accepts_random_changes_but_not_structure_or_projection(
    tmp_path,
):
    r, p = bundle(tmp_path)
    assert validate_checkpoint(r).environment_steps == 1024
    other = training().episode(4).input
    assert validate_checkpoint(
        replace(r, arrivals=other.arrivals, quality=other.quality)
    )
    with pytest.raises(ConfigurationError, match="incompatible"):
        validate_checkpoint(
            replace(
                r,
                algorithm=r.algorithm.model_copy(
                    update={
                        "algorithm": r.algorithm.algorithm.model_copy(
                            update={
                                "projection": replace(
                                    r.algorithm.algorithm.projection, max_jobs=5
                                )
                            }
                        )
                    }
                ),
            )
        )
    from smartsom.domain import Machine

    with pytest.raises(ConfigurationError, match="incompatible"):
        validate_checkpoint(
            replace(
                r,
                factory=replace(
                    r.factory,
                    machines=(*r.factory.machines, Machine("new")),
                    transport=None,
                    holding_buffer=None,
                ),
            )
        )
    (p / "model.zip").write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="digest"):
        validate_checkpoint(r)


def test_training_rejects_checkpoint_and_rollout_overshoot(tmp_path):
    r = training()
    run = primitive(r.run)
    run["budget"]["environment_steps"] = 257
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(run))
    with pytest.raises(ConfigurationError, match="whole PPO rollouts"):
        resolve_training_run(path)
    algorithm = primitive(r.algorithm)
    algorithm["algorithm"]["checkpoint"] = "somewhere"
    a = tmp_path / "a.yaml"
    a.write_text(yaml.safe_dump(algorithm))
    run["algorithm"] = str(a)
    run["budget"]["environment_steps"] = 1024
    path.write_text(yaml.safe_dump(run))
    with pytest.raises(ConfigurationError, match="without a checkpoint"):
        resolve_training_run(path)


def test_learner_parameters_and_projection_are_strict():
    r = primitive(training().algorithm)
    for field, bad in (
        ("n_steps", True),
        ("learning_rate", float("inf")),
        ("batch_size", 7),
    ):
        changed = json.loads(json.dumps(r))
        changed["algorithm"]["parameters"][field] = bad
        with pytest.raises(ValueError):
            AlgorithmFile.model_validate_json(json.dumps(changed))
    assert isinstance(training().algorithm.algorithm, LearningAlgorithm)


def test_episode_materialization_does_not_read_source_files(monkeypatch):
    r = training()
    expected = digest(r.episode(3))
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("episode reread file")
    )
    assert digest(r.episode(3)) == expected


def test_fixed_inputs_do_not_resample_and_global_rng_is_untouched():
    import random

    from smartsom.config import resolve_run
    from smartsom.config.models import FixedQuality

    r = training()
    base = resolve_run(ROOT / "configs/runs/quality_combined.yaml")
    base = replace(
        base,
        scenario=base.scenario.model_copy(
            update={"quality": FixedQuality(kind="fixed", path="unused.json")}
        ),
    )
    r = replace(r, base=base)
    state = random.getstate()
    for i in range(3):
        inputs = r.episode(i).input
        assert inputs.arrivals == base.arrivals
        assert inputs.processing_times == base.processing_times
        assert inputs.machine_events == base.machine_events
        assert inputs.quality == base.quality
        assert not any(s.consumed for s in r.episode(i).seeds)
    assert random.getstate() == state


def test_generated_base_is_frozen_before_episode_materialization(monkeypatch):
    from smartsom.config import resolve_run

    base = resolve_run(ROOT / "configs/runs/generated_fjsp_spt.yaml")
    r = replace(training(), base=base)
    monkeypatch.setattr(
        "smartsom.config.materialization.generate_fjsp",
        lambda *a, **kw: pytest.fail("regenerated base workload"),
    )
    assert r.episode(0).input.workload is base.workload
    assert r.episode(99).input.workload is base.workload


def test_learner_nonfinite_values_fail_instead_of_becoming_json_nan(tmp_path):
    from contextlib import ExitStack

    from smartsom.experiments.training import TrainingEvidence

    with ExitStack() as stack:
        evidence = TrainingEvidence(tmp_path, training(), stack, None)
        for value in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="nonfinite learner"):
                evidence.learner(1, {"loss": value})
        evidence.learner(1, {"loss": 1.0})
    assert "NaN" not in (tmp_path / "learner_metrics.jsonl").read_text()


def test_missing_backend_is_clear_and_does_not_install(monkeypatch):
    import importlib.metadata

    with monkeypatch.context() as m:
        m.setattr(
            importlib.metadata,
            "version",
            lambda name: (_ for _ in ()).throw(
                importlib.metadata.PackageNotFoundError(name)
            ),
        )
        with pytest.raises(ConfigurationError, match="learning-rllib"):
            real_require_backend("rllib.ppo")


def test_training_write_failure_retains_original_cause_and_failed_summary(
    tmp_path, monkeypatch
):
    pytest.importorskip("gymnasium")
    from smartsom.experiments import training as module

    r = training()
    r = replace(r, run=r.run.model_copy(update={"output_root": str(tmp_path)}))
    original = module.write_json
    failure = OSError("injected initial evidence failure")
    calls = 0

    def fail_once(path, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        original(path, value)

    monkeypatch.setattr(module, "write_json", fail_once)
    with pytest.raises(module.TrainingFailedError) as caught:
        module.train_one(r)
    assert caught.value.cause is failure
    directory = caught.value.run_dir
    assert json.loads((directory / "summary.json").read_text())["status"] == "failed"
    assert "makespan" not in json.loads((directory / "summary.json").read_text())
    assert (directory / "failure.json").exists()


def test_backend_exception_preserves_partial_actions_and_trace(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    import sys
    import types

    from smartsom.experiments.training import TrainingFailedError, train_one

    r = training()
    r = replace(r, run=r.run.model_copy(update={"output_root": str(tmp_path)}))
    backend = types.ModuleType("smartsom.learning.sb3")
    failure = RuntimeError("injected backend failure")

    def fail(resolved, env, evidence, checkpoint):
        env.reset()
        env.step(next(i for i, x in enumerate(env.projected.actions) if x is not None))
        raise failure

    backend.train = fail
    monkeypatch.setitem(sys.modules, "smartsom.learning.sb3", backend)
    with pytest.raises(TrainingFailedError) as caught:
        train_one(r)
    assert caught.value.cause is failure
    directory = caught.value.run_dir
    record = json.loads((directory / "episodes.jsonl").read_text())
    assert len(record["steps"]) == 1 and record["end_reason"] == "training_error"
    detail = json.loads((directory / "episode_000000_failure.json").read_text())
    assert detail["trace"] and detail["input"]
    assert record["makespan"] is None
