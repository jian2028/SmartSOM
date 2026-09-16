"""Frozen inputs, scientific random streams and public grid run evidence."""

import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from smartsom.algorithms.production import ScriptedProductionPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.config.models import ProfileFile
from smartsom.config.production import WorkloadFile, materialize
from smartsom.domain import FactorySpec, Machine
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import audit
from smartsom.workloads import IntegerRange, StaticJSPProfile, generate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path):
    for directory in ("configs", "data"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    # Historical fixture bytes remain available as historical evidence only.
    shutil.copytree(ROOT / "tests/fixtures/fixed_trace", tmp_path / "historical")
    return tmp_path


def edit(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def run_path(bundle, name="crossing"):
    return bundle / "configs" / "runs" / f"{name}.yaml"


def json_file(directory, name):
    return json.loads((directory / name).read_text())


def json_lines(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


@pytest.mark.parametrize(
    "name,makespan", [("crossing", 32), ("fjsp_fast", 32), ("fjsp_slow", 33)]
)
def test_config_and_code_paths_match_with_complete_evidence(bundle, name, makespan):
    prepared = resolve_run(run_path(bundle, name))
    recipe = prepared.resolved
    sim = ProductionSimulator(recipe.scenario)
    direct = [sim.step(command) for command in recipe.algorithm.commands]
    actual = run_one(prepared, verbose=False)
    assert sim.status == actual.simulation_result.status == "completed"
    assert sim.tick == actual.simulation_result.makespan == makespan
    assert sim.snapshot() == actual.simulation_result.final_state
    directory = actual.run_dir
    assert {p.name for p in directory.iterdir()} == {"run.json", "trace.jsonl"}
    recorded = json_lines(directory, "trace.jsonl")
    for expected, row in zip(direct, recorded, strict=True):
        assert all(
            primitive(expected[k]) == row[k]
            for k in ("state", "events", "actions", "reward", "rejections")
        )
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "completed" and manifest["result"]["tick"] == makespan
    assert digest(manifest["inputs"]["scenario"]) == digest(recipe.scenario)
    assert (
        manifest["source"]["git"]["commit"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert manifest["source"]["packages"]["pydantic"]
    assert manifest["provider"] == "builtin.scripted"
    assert audit(directory)["status"] == "passed"


def test_seed_and_generator_golden_values(bundle):
    from smartsom.config.production import named_seed

    recipe = resolve_run(run_path(bundle, "generated")).resolved
    assert named_seed(42, "workload") == 2123014631237417021
    assert [
        (d.demand_id, [(s.operation_type, s.nominal_ticks) for s in d.steps])
        for d in recipe.scenario.demands
    ] == [
        ("demand_0001", [("operation_2", 4)]),
        ("demand_0002", [("operation_1", 2), ("operation_1", 5)]),
    ]
    assert recipe.scenario.demands == recipe.episode(1234).demands


def test_generator_routes_bounds_and_rng_are_independent(bundle):
    from smartsom.config.production import ScenarioFile

    prepared = resolve_run(run_path(bundle, "generated"))
    factory = prepared.resolved.scenario.factory
    raw = ScenarioFile.model_validate_json(prepared.resolved.settings_json)
    profile = WorkloadFile.model_validate_json(
        json.dumps(
            {
                "schema": "smartsom.workload/v2",
                "profile": {
                    "jobs": 6,
                    "operation_types": ["operation_1", "operation_2"],
                    "min_operations": 1,
                    "max_operations": 2,
                    "nominal_min": 1,
                    "nominal_max": 7,
                },
            }
        )
    )
    before = random.getstate()
    for seed in range(20):
        world = materialize(factory, profile, raw, seed)
        reordered = replace(factory, machines=tuple(reversed(factory.machines)))
        assert world.demands == materialize(reordered, profile, raw, seed).demands
        assert len(world.demands) == 6
        for demand in world.demands:
            assert 1 <= len(demand.steps) <= 2
            assert len({s.operation_id for s in demand.steps}) == len(demand.steps)
            assert all(1 <= s.nominal_ticks <= 7 for s in demand.steps)
    assert random.getstate() == before
    fixed = profile.model_copy(
        update={
            "profile": profile.profile.model_copy(
                update={"nominal_min": 3, "nominal_max": 3}
            )
        }
    )
    assert {
        s.nominal_ticks
        for d in materialize(factory, fixed, raw, 0).demands
        for s in d.steps
    } == {3}


def test_export_import_keeps_the_world_when_seed_and_algorithm_change(bundle):
    recipe = resolve_run(run_path(bundle, "generated")).resolved
    world = recipe.scenario
    frozen = bundle / "configs/workloads/frozen.yaml"
    frozen.write_text(
        json.dumps(
            {"schema": "smartsom.workload/v2", "demands": primitive(world.demands)}
        )
    )
    edit(
        bundle / "configs/scenarios/generated.yaml",
        lambda d: d.update(workload="../workloads/frozen.yaml"),
    )
    edit(
        run_path(bundle, "generated"),
        lambda d: d.update(seed=99, algorithm="../algorithms/spt.yaml"),
    )
    imported = resolve_run(run_path(bundle, "generated")).resolved
    assert json.loads(recipe.workload_source_json)["profile"]
    assert json.loads(imported.workload_source_json)["profile"] is None
    assert imported.scenario.demands == world.demands
    assert imported.scenario.seed != world.seed
    assert imported.algorithm.provider == "builtin.spt"


def test_resolution_is_immutable_and_execution_does_not_reread_inputs(bundle):
    prepared = resolve_run(run_path(bundle))
    with pytest.raises(FrozenInstanceError):
        prepared.scientific_sha256 = "changed"
    with pytest.raises(FrozenInstanceError):
        prepared.resolved.scenario.demands[0].steps[0].nominal_ticks = 100
    # Exposed algorithm objects are detached from the frozen JSON recipe.
    prepared.resolved.algorithm.learning_rate = 0.001
    assert prepared.resolved.algorithm.learning_rate == 0.0003
    detached = primitive(prepared)
    detached["resolved"]["scenario_json"] = "{}"
    shutil.rmtree(bundle / "configs")
    assert run_one(prepared, verbose=False).simulation_result.makespan == 32


def test_content_digest_ignores_semantic_container_order_and_file_format(bundle):
    first = resolve_run(run_path(bundle))
    path = bundle / "configs/workloads/crossing.yaml"
    before = path.read_bytes()
    edit(path, lambda d: d["demands"].reverse())
    second = resolve_run(run_path(bundle))
    assert path.read_bytes() != before
    # Job order has no scheduling authority: compare committed execution.
    a, b = run_one(first, verbose=False), run_one(second, verbose=False)
    assert a.simulation_result == b.simulation_result
    assert (a.run_dir / "trace.jsonl").read_bytes() == (
        b.run_dir / "trace.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    "relative,change",
    [
        *[
            ("configs/runs/generated.yaml", lambda d, value=v: d.update(seed=value))
            for v in (True, 1.0, "1", -1, 2**64)
        ],
        ("configs/runs/generated.yaml", lambda d: d.update(objective="tardiness")),
        ("configs/runs/generated.yaml", lambda d: d.update(budget=10)),
        ("configs/runs/generated.yaml", lambda d: d.update(scenario="missing.yaml")),
        (
            "configs/factories/factory_hand.yaml",
            lambda d: d["factory"]["machines"][0].update(capacity=2),
        ),
        (
            "configs/factories/factory_hand.yaml",
            lambda d: d["factory"]["machines"].append(d["factory"]["machines"][0]),
        ),
        ("configs/workloads/static_jsp.yaml", lambda d: d.update(seed=42)),
        ("configs/workloads/static_jsp.yaml", lambda d: d["profile"].update(jobs=-1)),
        ("configs/workloads/static_jsp.yaml", lambda d: d["profile"].update(jobs=True)),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(min_operations=3, max_operations=2),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_min=0),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_min=3, nominal_max=1),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_min=1.0),
        ),
        ("configs/workloads/static_jsp.yaml", lambda d: d.update(generator="unknown")),
        (
            "configs/scenarios/generated.yaml",
            lambda d: d.update(workload={"path": "extra.json"}),
        ),
        ("configs/scenarios/generated.yaml", lambda d: d.update(workload=None)),
        (
            "configs/scenarios/generated.yaml",
            lambda d: d.update(modules=["machine_breakdown"]),
        ),
        (
            "configs/scenarios/generated.yaml",
            lambda d: d.update(visibility="full_future"),
        ),
        ("configs/scenarios/generated.yaml", lambda d: d.update(termination="horizon")),
        (
            "configs/algorithms/first_feasible.yaml",
            lambda d: d["algorithm"].update(seed=42),
        ),
        (
            "configs/algorithms/first_feasible.yaml",
            lambda d: d["algorithm"].update(provider="builtin.unsupported"),
        ),
        (
            "configs/algorithms/first_feasible.yaml",
            lambda d: d["algorithm"].update(parameters={"unused": 1}),
        ),
    ],
)
def test_invalid_authoring_fails_before_simulator_or_run_directory(
    bundle, monkeypatch, relative, change
):
    import smartsom.engine.production as engine

    monkeypatch.setattr(
        engine,
        "ProductionSimulator",
        lambda *a: pytest.fail("invalid authoring reached engine"),
    )
    edit(bundle / relative, change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "generated"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1])
def test_instance_duration_is_strict(bundle, value):
    edit(
        bundle / "configs/workloads/crossing.yaml",
        lambda d: d["demands"][0]["steps"][0].update(nominal_ticks=value),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(content_sha256="0" * 64),
        lambda d: d["demands"].append(d["demands"][0]),
        lambda d: d["demands"][0]["steps"][0].update(
            machine_nominal_ticks={"missing": 1}
        ),
        lambda d: d["demands"][0]["steps"][0].update(operation_type="missing"),
    ],
)
def test_instance_cross_references_and_digest_fail_before_execution(bundle, change):
    edit(bundle / "configs/workloads/crossing.yaml", change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "text",
    [
        "schema: smartsom.run/v1\nseed: 1\nseed: 2\n",
        '{"schema":"smartsom.run/v1","seed":1,"seed":2}',
        "schema: smartsom.run/v1\nnested: {x: 1, x: 2}\n",
        "!!python/object/apply:builtins.print [unsupported]\n",
        "schema: smartsom.run/v1\na: &a {x: 1}\nb: {<<: *a}\n",
    ],
)
def test_duplicate_keys_and_non_data_yaml_are_rejected(tmp_path, text):
    path = tmp_path / ("run.json" if text.startswith("{") else "run.yaml")
    path.write_text(text)
    with pytest.raises(ConfigurationError):
        resolve_run(path)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda actions: actions.clear(), "exhausted"),
        (lambda actions: actions.pop(), "exhausted"),
        (lambda actions: actions.append(actions[-1]), "extra commands"),
    ],
)
def test_script_failures_keep_evidence_without_successful_objective(
    bundle, change, message
):
    edit(
        bundle / "configs/algorithms/crossing_script.yaml",
        lambda d: change(d["algorithm"]["parameters"]["commands"]),
    )
    with pytest.raises(RunFailedError, match=message) as caught:
        run_one(resolve_run(run_path(bundle)), verbose=False)
    record = json_file(caught.value.run_dir, "run.json")
    assert record["status"] == "failed" and message in record["failure"]["message"]
    assert caught.value.__cause__ is caught.value.cause
    assert record["last_tick"] == len(json_lines(caught.value.run_dir, "trace.jsonl"))


@pytest.mark.parametrize(
    "resource,command",
    [
        ("agvs", [["missing", "WAIT"]]),
        ("machines", [["missing", {"job_id": None, "quality_mode": "normal"}]]),
    ],
)
def test_unknown_script_reference_fails_during_resolution(bundle, resource, command):
    edit(
        bundle / "configs/algorithms/crossing_script.yaml",
        lambda d: d["algorithm"]["parameters"]["commands"][0].update(
            {resource: command}
        ),
    )
    with pytest.raises(ConfigurationError, match="unknown|Extra inputs"):
        resolve_run(run_path(bundle))


def test_invalid_path_value_is_a_configuration_error(bundle):
    edit(run_path(bundle), lambda d: d.update(scenario="bad\0path.yaml"))
    with pytest.raises(ConfigurationError, match="invalid|embedded null"):
        resolve_run(run_path(bundle))
    assert not (bundle / "runs").exists()


def test_unexpected_provider_failure_preserves_partial_trace(bundle, monkeypatch):
    original = ScriptedProductionPolicy.act

    def act(policy, context):
        if policy.position == 1:
            raise RuntimeError("provider crashed")
        return original(policy, context)

    monkeypatch.setattr(ScriptedProductionPolicy, "act", act)
    with pytest.raises(RunFailedError) as caught:
        run_one(resolve_run(run_path(bundle)), verbose=False)
    record = json_file(caught.value.run_dir, "run.json")
    assert record["status"] == "failed" and record["last_tick"] == 1
    assert [row["tick"] for row in json_lines(caught.value.run_dir, "trace.jsonl")] == [
        1
    ]


def test_initialization_failure_has_stage_appropriate_evidence(bundle, monkeypatch):
    def fail(*args):
        raise RuntimeError("cannot construct provider")

    monkeypatch.setattr(ScriptedProductionPolicy, "__init__", fail)
    with pytest.raises(RunFailedError) as caught:
        run_one(resolve_run(run_path(bundle)), verbose=False)
    record = json_file(caught.value.run_dir, "run.json")
    assert record["status"] == "failed" and record["last_tick"] == 0
    assert record["inputs"]["scenario"]


def test_source_capture_failure_is_retained_after_allocation(bundle, monkeypatch):
    import smartsom.experiments.evidence as evidence

    def fail():
        raise RuntimeError("cannot capture source identity")

    monkeypatch.setattr(evidence, "source_identity", fail)
    with pytest.raises(RunFailedError) as caught:
        run_one(resolve_run(run_path(bundle)), verbose=False)
    record = json_file(caught.value.run_dir, "run.json")
    assert record["status"] == "failed" and record["last_tick"] == 0
    assert record["inputs"]["scenario"]


def test_runner_uses_public_step_and_starts_a_fresh_policy(bundle, monkeypatch):
    original = ProductionSimulator.step
    commands = []

    def step(simulator, command):
        commands.append(command)
        return original(simulator, command)

    monkeypatch.setattr(ProductionSimulator, "step", step)
    prepared = resolve_run(run_path(bundle))
    first, second = run_one(prepared, verbose=False), run_one(prepared, verbose=False)
    assert first.simulation_result == second.simulation_result
    # Full replay independently uses exactly those same public semantic commands.
    expected = [
        command for command in prepared.resolved.algorithm.commands for _ in range(2)
    ] * 2
    assert commands == expected
    assert first.run_dir != second.run_dir


def test_cli_validate_is_read_only_and_run_has_meaningful_exit_codes(
    bundle, monkeypatch, capsys
):
    import smartsom.engine.production as engine

    with monkeypatch.context() as patch:
        patch.setattr(
            engine, "ProductionSimulator", lambda *a: pytest.fail("validate simulated")
        )
        assert main(["validate", str(run_path(bundle))]) == 0
        assert "valid workload_sha256=" in capsys.readouterr().out
        assert not (bundle / "runs").exists()
    assert main(["run", str(run_path(bundle))]) == 0
    assert "completed makespan=32" in capsys.readouterr().out
    edit(
        bundle / "configs/algorithms/crossing_script.yaml",
        lambda d: d["algorithm"]["parameters"].update(commands=[]),
    )
    assert main(["run", str(run_path(bundle))]) == 1
    assert "run failed in" in capsys.readouterr().err
    assert main(["validate", str(bundle / "missing.yaml")]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_cli_subprocess_is_stable_across_hash_seed_and_working_directory(
    bundle, tmp_path
):
    path = run_path(bundle)
    script = """
import sys
from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.experiments import run_one
r = resolve_run(sys.argv[1])
print(canonical_json({'world': r.resolved.scenario, 'result': run_one(r, verbose=False).simulation_result}))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script, str(path)],
            cwd=cwd,
            env=dict(os.environ, PYTHONHASHSEED=seed),
            text=True,
        )
        for seed, cwd in (("1", ROOT), ("17", tmp_path), ("321", tmp_path.parent))
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    executed = subprocess.run(
        [sys.executable, "-m", "smartsom.experiments.cli", "validate", str(path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert executed.returncode == 0 and "valid workload_sha256=" in executed.stdout


def test_profile_dataclasses_and_pydantic_envelope_share_validation():
    with pytest.raises(ValueError):
        IntegerRange(2, 1)
    with pytest.raises(ValueError):
        StaticJSPProfile(True, 1, IntegerRange(1, 1), IntegerRange(1, 1))
    profile = ProfileFile.model_validate_json(
        json.dumps(
            {
                "schema": "smartsom.workload-profile/v1",
                "generator": "static_jsp_v1",
                "profile": {
                    "order_count": 1,
                    "jobs_per_order": 1,
                    "operations_per_job": {"min": 1, "max": 1},
                    "nominal_ticks": {"min": 2, "max": 2},
                },
            }
        )
    )
    with pytest.raises(FrozenInstanceError):
        profile.profile.nominal_ticks.min = 5
    with pytest.raises(ValueError, match="exceeds"):
        generate(
            FactorySpec((Machine("M1"),)),
            replace(profile.profile, operations_per_job=IntegerRange(2, 2)),
            0,
        )
