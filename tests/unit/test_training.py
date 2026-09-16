"""Frozen scientific episodes and grid checkpoint preflight, without learner imports."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from smartsom.config import resolve_training_run
from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.experiment import ExperimentConfig, load_config, prepare
from smartsom.config.models import AlgorithmFile, TrainingRunSpec
from smartsom.config.production import checkpoint_algorithm_document
from smartsom.config.training import episode_root, framework_seed
from smartsom.experiments.cli import main
from smartsom.experiments.evidence import write_json
from smartsom.learning.checkpoint import file_hash
from smartsom.learning.checkpoint import require_backend as real_require_backend
from smartsom.learning.production_contract import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    validate_model_contract,
)
from smartsom.trace.production import seal_checkpoint, state_hash

ROOT = Path(__file__).resolve().parents[2]


def training(provider="sb3"):
    return resolve_training_run(ROOT / f"configs/runs/learning_{provider}.yaml")


def bundle(tmp_path, provider="sb3"):
    """Manifest-only fixture: it cannot train or predict and is never used as a model."""
    prepared = training(provider)
    recipe = prepared.resolved
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.fixture").write_bytes(
        b"preflight fixture, not model weights"
    )
    metadata = {
        "schema": "smartsom.production-checkpoint/v1",
        "provider": recipe.algorithm.provider,
        "algorithm": primitive(recipe.algorithm),
        "scenario": primitive(recipe.scenario),
        "factory_hash": state_hash(recipe.scenario.factory),
        "quality_probability_visibility": recipe.scenario.quality_probability_visibility,
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "environment_steps": 4096 if provider == "marl" else 1024,
        "learner_updates": 16 if provider == "marl" else 4,
    }
    roles = (
        ["agv_policy", "buffer_policy", "machine_policy", "quality_policy"]
        if provider == "marl"
        else ["default_policy"]
        if provider == "rllib"
        else ["policy"]
    )
    if provider != "sb3":
        metadata["modules"] = roles
    metadata["weights"] = dict.fromkeys(roles, "a" * 64)
    for role in roles:
        member = "model.zip" if provider == "sb3" else f"{role}.pt"
        (checkpoint / member).write_bytes(b"preflight fixture, not model weights")
    write_json(checkpoint / "checkpoint.json", metadata)
    seal_checkpoint(checkpoint)
    algorithm_path = tmp_path / "algorithm.json"
    write_json(
        algorithm_path,
        checkpoint_algorithm_document(
            metadata, checkpoint, file_hash(checkpoint / "checkpoint.json")
        ),
    )
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    config.algorithm.source = str(algorithm_path)
    config.output.root = str(tmp_path / "runs")
    return prepare(config, training=False), checkpoint


def test_micro_preserves_raw_jobs_and_capabilities_in_explicit_grid():
    prepared = training()
    recipe = prepared.resolved
    scenario = recipe.scenario
    source = json.loads(
        (ROOT / "data/reference/idetc/converted/S00/workload.json").read_text()
    )
    jobs = sorted(
        (j for o in source["workload"]["orders"] for j in o["jobs"]),
        key=lambda j: j["job_id"],
    )[:4]
    assert sum(len(d.steps) for d in scenario.demands) == 9
    for demand, original in zip(scenario.demands, jobs, strict=True):
        assert demand.demand_id == original["job_id"]
        for step, op in zip(demand.steps, original["operations"], strict=True):
            assert step.operation_id == op["operation_id"]
            assert dict(step.machine_nominal_ticks) == {
                m["machine_id"]: m["nominal_ticks"] for m in op["modes"]
            }
            assert {
                m.machine_id
                for m in scenario.factory.machines
                if step.operation_type in m.operation_types
            } == set(dict(step.machine_nominal_ticks))
    assert len(scenario.factory.machines) == 8 and len(scenario.factory.agvs) == 4
    assert scenario.factory.ports and scenario.factory.buffers
    assert json.loads(prepared.config_json)["training"]["total_steps"] == 1024
    assert not scenario.outages and not scenario.processing_samples


def test_episode_seed_golden_pairing_and_fixed_base():
    sb, rl = training().resolved, training("rllib").resolved
    assert framework_seed(101, "sb3.maskable_ppo") == 1572694888
    assert framework_seed(101, "rllib.ppo") == 189546015
    expected = (
        "b5cfeacc17009114180e040124fd8a4a445a77be71f032dfb70c18c969f4ce22",
        "60d801e57479ea8dc47deca00b7aec1cf698204d747e5857a5f7e30b76fa5469",
    )
    for index in range(5):
        seed = episode_root(101, index)
        a, b = sb.episode(seed), rl.episode(seed)
        assert a == b and a.seed == seed
        assert [(d.demand_id, d.steps) for d in a.demands] == [
            (d.demand_id, d.steps) for d in sb.scenario.demands
        ]
        if index < 2:
            assert digest(a) == expected[index]
        assert sum(d.release_at == 0 for d in a.demands) == 2
        assert all(
            d.reveal_at == d.release_at and 0 <= d.release_at <= 20 for d in a.demands
        )
    assert (
        sb.episode(episode_root(101, 0)).seed != sb.episode(episode_root(101, 1)).seed
    )
    with pytest.raises(FrozenInstanceError):
        sb.algorithm_seed = 1


def test_validate_training_does_not_simulate_or_create_output(
    tmp_path, monkeypatch, capsys
):
    config = load_config(ROOT / "configs/runs/learning_sb3.yaml")
    config.output.root = str(tmp_path / "absent")
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(primitive(config)))
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("validate created simulator"),
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
    data = yaml.safe_load((ROOT / "configs/runs/learning_sb3.yaml").read_text())
    data["budget"].update(changes)
    with pytest.raises(ValueError):
        TrainingRunSpec.model_validate_json(json.dumps(data))


def test_checkpoint_preflight_accepts_random_changes_but_not_structure_or_projection(
    tmp_path,
):
    prepared, checkpoint = bundle(tmp_path)
    recipe = prepared.resolved
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    validate_model_contract(
        metadata, recipe.episode(episode_root(101, 4)), recipe.algorithm
    )
    factory = recipe.scenario.factory
    reordered = replace(
        factory,
        name="工厂新名称",
        machines=tuple(
            replace(m, name=f"机器 {m.machine_id}") for m in factory.machines[::-1]
        ),
        ports=factory.ports[::-1],
        agvs=factory.agvs[::-1],
    )
    validate_model_contract(
        metadata, replace(recipe.scenario, factory=reordered), recipe.algorithm
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_model_contract(
            metadata,
            recipe.scenario,
            recipe.algorithm.model_copy(update={"max_jobs": 5}),
        )
    with pytest.raises(ValueError, match="factory differs"):
        validate_model_contract(
            metadata,
            replace(
                recipe.scenario,
                factory=replace(factory, machines=factory.machines[:-1]),
            ),
        )
    (checkpoint / "weights.fixture").write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="digest"):
        prepare(
            ExperimentConfig.model_validate_json(prepared.config_json), training=False
        )
    assert not (tmp_path / "runs").exists()


def test_checkpoint_unicode_factory_hash_and_frozen_factory_integrity():
    recipe = training().resolved
    case = replace(
        recipe.scenario, factory=replace(recipe.scenario.factory, name="中文工厂")
    )
    metadata = {
        "schema": "smartsom.production-checkpoint/v1",
        "scenario": primitive(case),
        "factory_hash": state_hash(case.factory),
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
    }
    validate_model_contract(metadata, case)
    metadata["scenario"]["factory"]["grid"]["width"] += 1
    with pytest.raises(ValueError, match="hash disagrees"):
        validate_model_contract(metadata, case)


def test_training_rejects_checkpoint_and_rollout_overshoot(tmp_path):
    config = load_config(ROOT / "configs/runs/learning_sb3.yaml")
    config.training.total_steps = 257
    with pytest.raises(ConfigurationError, match="whole updates"):
        prepare(config)
    config.training.total_steps = 1024
    data = yaml.safe_load(Path(config.algorithm.source).read_text())
    data["algorithm"]["checkpoint"] = "somewhere"
    path = tmp_path / "algorithm.yaml"
    path.write_text(yaml.safe_dump(data))
    config.algorithm.source = str(path)
    with pytest.raises(ConfigurationError, match="without a checkpoint"):
        prepare(config)


def test_learner_parameters_and_projection_are_strict():
    original = yaml.safe_load(
        (ROOT / "configs/algorithms/sb3_maskable_ppo.yaml").read_text()
    )
    for field, bad in (
        ("n_steps", True),
        ("learning_rate", float("inf")),
        ("batch_size", 7),
    ):
        changed = json.loads(json.dumps(original))
        changed["algorithm"]["parameters"][field] = bad
        with pytest.raises(ValueError):
            AlgorithmFile.model_validate_json(json.dumps(changed))


def test_episode_materialization_does_not_read_source_files(monkeypatch):
    recipe = training().resolved
    seed = episode_root(101, 3)
    expected = digest(recipe.episode(seed))
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("episode reread file")
    )
    assert digest(recipe.episode(seed)) == expected


def test_fixed_inputs_do_not_resample_and_global_rng_is_untouched():
    import random

    from smartsom.config import resolve_run

    recipe = resolve_run(ROOT / "configs/runs/quality_m0.yaml").resolved
    base = recipe.scenario
    state = random.getstate()
    for index in range(3):
        case = recipe.episode(episode_root(101, index))
        assert case.demands == base.demands
        assert case.outages == base.outages
        assert case.processing_samples == base.processing_samples
        assert case.quality_samples == base.quality_samples and case.quality_samples
    assert random.getstate() == state


def test_generated_base_is_frozen_before_episode_materialization(monkeypatch):
    from smartsom.config import resolve_run

    recipe = resolve_run(ROOT / "configs/runs/generated_fjsp_spt.yaml").resolved
    monkeypatch.setattr(
        "smartsom.config.production.random.Random",
        lambda *a, **k: pytest.fail("regenerated base workload"),
    )
    assert recipe.episode(episode_root(101, 0)).demands == recipe.scenario.demands
    assert recipe.episode(episode_root(101, 99)).demands == recipe.scenario.demands


def test_missing_backend_is_clear_and_does_not_install(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError(name)
        ),
    )
    with pytest.raises(ConfigurationError, match="learning-rllib"):
        real_require_backend("rllib.ppo")


def failing_training(tmp_path, monkeypatch, backend):
    pytest.importorskip("gymnasium")
    from smartsom.experiments.training import TrainingFailedError, train_one

    config = load_config(ROOT / "configs/runs/sb3_production.yaml")
    config.output.root = str(tmp_path / "runs")
    monkeypatch.setattr("smartsom.learning.production.train_sb3", backend)
    with pytest.raises(TrainingFailedError) as caught:
        train_one(prepare(config))
    root = caught.value.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed" and not record["paths"].get("checkpoint")
    return caught.value, root, root / record["paths"]["training"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_learner_nonfinite_values_fail_before_writing_metrics(
    tmp_path, monkeypatch, value
):
    def backend(*args, **kwargs):
        kwargs["evidence"].bind()
        kwargs["on_update"](
            256, 1, {"loss": value}, lambda *args: pytest.fail("nonfinite checkpoint")
        )

    caught, _, attempt = failing_training(tmp_path, monkeypatch, backend)
    assert "nonfinite learner" in str(caught.cause)
    assert (attempt / "learner_metrics.jsonl").read_text() == ""


def test_training_write_failure_retains_original_cause_and_failed_record(
    tmp_path, monkeypatch
):
    from smartsom.experiments import production_training as module
    from smartsom.experiments.training import TrainingFailedError, train_one

    failure = OSError("injected grid recipe write failure")
    original = module.write_json

    def fail_once(path, value):
        if path.name == "grid_recipe.json":
            raise failure
        original(path, value)

    monkeypatch.setattr(module, "write_json", fail_once)
    config = load_config(ROOT / "configs/runs/sb3_production.yaml")
    config.output.root = str(tmp_path / "runs")
    with pytest.raises(TrainingFailedError) as caught:
        train_one(prepare(config))
    assert caught.value.cause is failure
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["failure"]["message"] == str(failure)


def test_backend_exception_preserves_partial_decisions_and_committed_trace(
    tmp_path, monkeypatch
):
    pytest.importorskip("gymnasium")
    from smartsom.learning.sampling import OrderedSamplingPool

    failure = RuntimeError("injected backend failure")

    def backend(*args, **kwargs):
        evidence = kwargs["evidence"]
        evidence.bind()
        with OrderedSamplingPool(kwargs["sampling_spec"], 1, 0, evidence) as pool:
            pool.reset()
            # Real adapter decisions and physical commits, with no fake learner claim.
            for _ in range(3):
                pool.step(
                    [
                        next(
                            i
                            for i, allowed in enumerate(
                                pool.local[0].adapter.action_masks()
                            )
                            if allowed
                        )
                    ]
                )
            raise failure

    caught, _, attempt = failing_training(tmp_path, monkeypatch, backend)
    assert caught.cause is failure
    rows = json.loads((attempt / "active_episodes.json").read_text())
    assert len(rows) == 1 and len(rows[0]["steps"]) == 3
    assert rows[0]["trace"] and rows[0]["input_sha256"]
    assert rows[0]["makespan"] is None
