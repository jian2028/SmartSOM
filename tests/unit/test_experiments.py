import hashlib
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
from pydantic import ValidationError
from test_static_engine import (
    assert_schedule_is_legal,
    competition_case,
    crossing_case,
)

from smartsom.algorithms import FirstFeasiblePolicy, ScriptedPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, normalize_workload, primitive
from smartsom.config.models import ProfileFile
from smartsom.config.seeds import derive_seeds
from smartsom.domain import FactorySpec, Machine
from smartsom.engine import Simulator, replay
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.workloads import IntegerRange, StaticJSPProfile, generate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path):
    for directory in ("configs", "data"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copytree(ROOT / "tests/fixtures/fixed_trace", tmp_path, dirs_exist_ok=True)
    return tmp_path


def edit(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def run_path(bundle, name="run_fixed_trace"):
    return bundle / "configs" / "runs" / f"{name}.yaml"


def json_file(directory, name):
    return json.loads((directory / name).read_text())


def json_lines(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


@pytest.mark.parametrize(
    ("name", "case", "makespan"),
    [
        ("run_fixed_trace", competition_case, 6),
        ("crossing", crossing_case, 5),
    ],
)
def test_config_and_code_paths_match_with_complete_evidence(
    bundle, name, case, makespan
):
    factory, workload, actions = case()
    resolved = resolve_run(run_path(bundle, name))
    assert resolved.factory == factory
    assert resolved.workload == workload
    direct = Simulator(factory, workload).run(ScriptedPolicy(actions))
    actual = run_one(resolved)
    assert actual.simulation_result == direct == replay(factory, workload, actions)
    assert actual.simulation_result.makespan == makespan
    assert_schedule_is_legal(workload, actual.simulation_result)
    directory = actual.run_dir
    assert {p.name for p in directory.iterdir()} == {
        "resolved_run.yaml",
        "realized_instance.json",
        "manifest.json",
        "progress.log",
        "trace.jsonl",
        "metrics.jsonl",
        "summary.json",
    }
    assert json_lines(directory, "trace.jsonl") == primitive(direct.trace)
    expected_metrics = []
    for record in direct.trace:
        if record.kind == "complete":
            expected_metrics.append(
                {
                    "kind": "completion",
                    "simulation_time": record.simulation_time,
                    "completed_operations": len(expected_metrics) + 1,
                }
            )
    summary = json_file(directory, "summary.json")
    assert summary == {
        "schema": "smartsom.summary/v1",
        "status": "completed",
        "end_reason": "completed",
        "simulation_time": makespan,
        "completed_operations": 4,
        "makespan": makespan,
    }
    assert json_lines(directory, "metrics.jsonl") == [
        *expected_metrics,
        {"kind": "terminal", **summary},
    ]
    manifest = json_file(directory, "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["workload_sha256"] == digest(workload)
    assert (
        manifest["source"]["git"]["commit"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    assert manifest["source"]["packages"]["pydantic"]
    assert manifest["provider_implementation"].endswith("ScriptedPolicy")
    assert len(manifest["sources"]) == 5
    assert not any(seed["consumed"] for seed in manifest["seeds"])
    for name, sha256 in manifest["artifacts"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha256
    saved = yaml.safe_load((directory / "resolved_run.yaml").read_text())
    assert saved["workload"] == primitive(workload)
    assert saved["algorithm"]["algorithm"]["provider"] == "builtin.scripted"


def test_seed_and_generator_golden_values(bundle):
    seeds = derive_seeds(42, generated=True)
    assert [(seed.domain, seed.value) for seed in seeds] == [
        ("workload", 6941565647359864201),
        ("demand", 1575026194469689329),
        ("machine_events", 18229639943394249491),
        ("processing_time", 3797569027003476775),
        ("algorithm", 17364847280822799010),
        ("solver", 9725273414194853204),
    ]
    assert [seed.domain for seed in seeds if seed.consumed] == ["workload"]
    resolved = resolve_run(run_path(bundle, "generated"))
    assert (
        resolved.workload_sha256
        == "c039e0307dc2c71cc3b29d6e1ee3ad466dd7055b44422a2fd254d0678d0ee7a5"
    )
    assert [
        (
            op.operation_id,
            op.modes[0].machine_id,
            op.modes[0].nominal_ticks,
            op.predecessor_ids,
        )
        for op in resolved.workload.operations
    ] == [
        ("order_1/job_1/op_1", "M2", 5, ()),
        ("order_1/job_1/op_2", "M1", 3, ("order_1/job_1/op_1",)),
        ("order_1/job_2/op_1", "M1", 1, ()),
    ]
    assert resolved.provenance.effective_seed == seeds[0].value
    assert resolved.provenance.profile_sha256 == digest(resolved.profile)


def test_generator_routes_bounds_and_rng_are_independent():
    factory = FactorySpec(tuple(Machine(f"M{i}") for i in range(1, 5)))
    profile = StaticJSPProfile(2, 3, IntegerRange(2, 4), IntegerRange(1, 7))
    before = random.getstate()
    for seed in range(20):
        workload = generate(factory, profile, seed)
        assert workload == generate(
            FactorySpec(tuple(reversed(factory.machines))), profile, seed
        )
        assert len(workload.orders) == 2
        for order in workload.orders:
            assert len(order.jobs) == 3
            for job in order.jobs:
                assert 2 <= len(job.operations) <= 4
                route = [op.modes[0].machine_id for op in job.operations]
                assert len(set(route)) == len(route)
                for index, op in enumerate(job.operations):
                    assert op.predecessor_ids == (
                        (job.operations[index - 1].operation_id,) if index else ()
                    )
                    assert 1 <= op.modes[0].nominal_ticks <= 7
                    assert op.modes[0].processing_mode_id == "standard"
        assert_schedule_is_legal(
            workload, Simulator(factory, workload).run(FirstFeasiblePolicy())
        )
    assert random.getstate() == before
    fixed = generate(
        factory, StaticJSPProfile(1, 1, IntegerRange(4, 4), IntegerRange(3, 3)), 0
    )
    assert len(fixed.operations) == 4
    assert {op.modes[0].nominal_ticks for op in fixed.operations} == {3}


def test_export_import_keeps_the_world_when_seed_and_algorithm_change(bundle):
    generated = resolve_run(run_path(bundle, "generated"))
    first = run_one(generated)
    scenario_path = bundle / "configs/scenarios/generated.yaml"
    edit(
        scenario_path,
        lambda data: data.update(
            workload={
                "kind": "instance",
                "path": str(first.run_dir / "realized_instance.json"),
            }
        ),
    )
    edit(run_path(bundle, "generated"), lambda data: data.update(seed=99))
    script_path = bundle / "configs/algorithms/first_feasible.yaml"
    script_path.write_text(
        json.dumps(
            {
                "schema": "smartsom.algorithm/v1",
                "algorithm": {
                    "provider": "builtin.scripted",
                    "parameters": {
                        "actions": primitive(first.simulation_result.actions)
                    },
                },
            }
        )
    )
    imported = resolve_run(run_path(bundle, "generated"))
    assert imported.workload == generated.workload
    assert imported.workload_sha256 == generated.workload_sha256
    assert imported.provenance == generated.provenance
    assert all(not seed.consumed for seed in imported.seeds)
    assert imported.seeds[0].value != imported.provenance.effective_seed
    second = run_one(imported)
    assert second.simulation_result == first.simulation_result
    assert second.run_dir != first.run_dir
    assert (second.run_dir / "trace.jsonl").read_bytes() == (
        first.run_dir / "trace.jsonl"
    ).read_bytes()


def test_resolution_is_immutable_and_execution_does_not_reread_inputs(bundle):
    resolved = resolve_run(run_path(bundle))
    with pytest.raises(FrozenInstanceError):
        resolved.workload_sha256 = "changed"
    with pytest.raises(ValidationError):
        resolved.run.seed = 100
    with pytest.raises(FrozenInstanceError):
        resolved.workload.operations[0].modes[0].nominal_ticks = 100
    with pytest.raises(ValidationError):
        resolved.algorithm.algorithm.parameters.actions = ()
    detached = primitive(resolved)
    detached["workload"]["orders"].clear()
    for source in resolved.sources:
        source.path.unlink()
    assert run_one(resolved).simulation_result.makespan == 6


def test_content_digest_ignores_semantic_container_order_and_file_format(bundle):
    first = resolve_run(run_path(bundle))
    instance_path = bundle / "data/instances/workload_fixed_trace.json"
    edit(instance_path, lambda data: data["workload"]["orders"][0]["jobs"].reverse())
    data = json.loads(instance_path.read_text())
    for job in data["workload"]["orders"][0]["jobs"]:
        job["operations"].reverse()
    instance_path.write_text(json.dumps(data, separators=(",", ":")))
    second = resolve_run(run_path(bundle))
    assert first.workload == second.workload
    assert first.workload_sha256 == second.workload_sha256
    assert first.sources[-1].sha256 != second.sources[-1].sha256
    assert normalize_workload(first.workload) == first.workload


@pytest.mark.parametrize(
    ("relative", "change"),
    [
        ("configs/runs/generated.yaml", lambda d: d.update(seed=True)),
        ("configs/runs/generated.yaml", lambda d: d.update(seed=1.0)),
        ("configs/runs/generated.yaml", lambda d: d.update(seed="1")),
        ("configs/runs/generated.yaml", lambda d: d.update(seed=-1)),
        ("configs/runs/generated.yaml", lambda d: d.update(seed=2**64)),
        ("configs/runs/generated.yaml", lambda d: d.update(objective="tardiness")),
        ("configs/runs/generated.yaml", lambda d: d.update(budget=10)),
        ("configs/runs/generated.yaml", lambda d: d.update(scenario="missing.yaml")),
        (
            "configs/factories/factory_test.yaml",
            lambda d: d["factory"]["machines"][0].update(capacity=2),
        ),
        (
            "configs/factories/factory_test.yaml",
            lambda d: d["factory"]["machines"].append(d["factory"]["machines"][0]),
        ),
        ("configs/workloads/static_jsp.yaml", lambda d: d.update(seed=42)),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(order_count=0),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(jobs_per_order=True),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(operations_per_job={"min": 1, "max": 3}),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_ticks={"min": 0, "max": 3}),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_ticks={"min": 3, "max": 1}),
        ),
        (
            "configs/workloads/static_jsp.yaml",
            lambda d: d["profile"].update(nominal_ticks={"min": 1.0, "max": 3}),
        ),
        ("configs/workloads/static_jsp.yaml", lambda d: d.update(generator="unknown")),
        (
            "configs/scenarios/generated.yaml",
            lambda d: d["workload"].update(instance="extra.json"),
        ),
        (
            "configs/scenarios/generated.yaml",
            lambda d: d["workload"].update(kind="instance"),
        ),
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
    import smartsom.experiments.runner as runner

    def forbidden(*args):
        pytest.fail("invalid authoring reached the engine")

    monkeypatch.setattr(runner, "Simulator", forbidden)
    edit(bundle / relative, change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "generated"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1])
def test_instance_duration_is_strict(bundle, value):
    path = bundle / "data/instances/workload_fixed_trace.json"
    edit(
        path,
        lambda d: d["workload"]["orders"][0]["jobs"][0]["operations"][0]["modes"][
            0
        ].update(nominal_ticks=value),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(content_sha256="0" * 64),
        lambda d: d["workload"].update(orders=[]),
        lambda d: d["workload"]["orders"][0]["jobs"][0]["operations"][0]["modes"][
            0
        ].update(machine_id="missing"),
        lambda d: d["workload"]["orders"][0]["jobs"][0]["operations"][1].update(
            predecessor_ids=["B1"]
        ),
    ],
)
def test_instance_cross_references_and_digest_fail_before_execution(bundle, change):
    edit(bundle / "data/instances/workload_fixed_trace.json", change)
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
    ("change", "message"),
    [
        (lambda actions: actions.clear(), "exhausted"),
        (lambda actions: actions.pop(), "exhausted"),
        (lambda actions: actions.append(actions[-1]), "extra actions"),
        (lambda actions: actions.insert(1, actions[0]), "already started"),
        (lambda actions: actions.insert(0, actions[-1]), "predecessor"),
    ],
)
def test_script_failures_keep_evidence_without_successful_objective(
    bundle, change, message
):
    edit(
        bundle / "configs/algorithms/algorithm_fixed_trace.yaml",
        lambda d: change(d["algorithm"]["parameters"]["actions"]),
    )
    resolved = resolve_run(run_path(bundle))
    with pytest.raises(RunFailedError, match=message) as error:
        run_one(resolved)
    directory = error.value.run_dir
    assert error.value.__cause__ is error.value.cause
    failure = json_file(directory, "failure.json")
    assert failure["stage"] == "simulation"
    assert message in failure["message"]
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_lines(directory, "trace.jsonl")[0]["kind"] == "decision"
    assert not any(
        row["kind"] == "terminal" for row in json_lines(directory, "metrics.jsonl")
    )
    assert "failed" in (directory / "progress.log").read_text()


def test_unknown_script_reference_fails_during_resolution(bundle):
    edit(
        bundle / "configs/algorithms/algorithm_fixed_trace.yaml",
        lambda d: d["algorithm"]["parameters"]["actions"][0].update(
            processing_mode_id="missing"
        ),
    )
    with pytest.raises(ConfigurationError, match="script references unknown"):
        resolve_run(run_path(bundle))


def test_invalid_path_value_is_a_configuration_error(bundle):
    edit(run_path(bundle), lambda d: d.update(scenario="bad\0path.yaml"))
    with pytest.raises(ConfigurationError, match="invalid reference"):
        resolve_run(run_path(bundle))
    assert not (bundle / "runs").exists()


def test_unexpected_provider_failure_preserves_partial_trace(bundle, monkeypatch):
    original = ScriptedPolicy.select_action
    calls = 0

    def fail_second(policy, context):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("provider crashed")
        return original(policy, context)

    monkeypatch.setattr(ScriptedPolicy, "select_action", fail_second)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle)))
    trace = json_lines(error.value.run_dir, "trace.jsonl")
    assert [record["kind"] for record in trace] == [
        "decision",
        "dispatch",
        "complete",
        "decision",
    ]
    summary = json_file(error.value.run_dir, "summary.json")
    assert summary["completed_operations"] == 1
    assert summary["simulation_time"] == 1


def test_initialization_failure_has_stage_appropriate_evidence(bundle, monkeypatch):
    import smartsom.experiments.runner as runner

    def fail(*args):
        raise RuntimeError("cannot construct provider")

    monkeypatch.setattr(runner, "build_provider", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle)))
    directory = error.value.run_dir
    assert json_file(directory, "failure.json")["stage"] == "initialization"
    assert (directory / "realized_instance.json").exists()
    assert (directory / "resolved_run.yaml").exists()
    assert not (directory / "trace.jsonl").exists()


def test_source_capture_failure_is_retained_after_allocation(bundle, monkeypatch):
    import smartsom.experiments.evidence as evidence

    def fail():
        raise RuntimeError("cannot capture source identity")

    monkeypatch.setattr(evidence, "source_identity", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle)))
    directory = error.value.run_dir
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert json_file(directory, "summary.json")["makespan"] is None
    assert (directory / "resolved_run.yaml").exists()
    assert (directory / "realized_instance.json").exists()


def test_runner_uses_public_step_and_starts_a_fresh_policy(bundle, monkeypatch):
    original = Simulator.step
    actions = []

    def step(simulator, action):
        actions.append(action)
        return original(simulator, action)

    monkeypatch.setattr(Simulator, "step", step)
    resolved = resolve_run(run_path(bundle))
    first, second = run_one(resolved), run_one(resolved)
    assert first.simulation_result == second.simulation_result
    assert actions == list(first.simulation_result.actions) * 2
    assert first.run_dir != second.run_dir


def test_cli_validate_is_read_only_and_run_has_meaningful_exit_codes(
    bundle, monkeypatch, capsys
):
    import smartsom.experiments.runner as runner

    with monkeypatch.context() as patch:
        patch.setattr(
            runner, "Simulator", lambda *args: pytest.fail("validate simulated")
        )
        assert main(["validate", str(run_path(bundle))]) == 0
        assert "valid workload_sha256=" in capsys.readouterr().out
        assert not (bundle / "runs").exists()
    assert main(["run", str(run_path(bundle))]) == 0
    assert "completed makespan=6" in capsys.readouterr().out
    edit(
        bundle / "configs/algorithms/algorithm_fixed_trace.yaml",
        lambda d: d["algorithm"]["parameters"].update(actions=[]),
    )
    assert main(["run", str(run_path(bundle))]) == 1
    assert "run failed in" in capsys.readouterr().err
    assert main(["validate", str(bundle / "missing.yaml")]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_cli_subprocess_is_stable_across_hash_seed_and_working_directory(
    bundle, tmp_path
):
    path = run_path(bundle, "generated")
    script = """
import sys
from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.experiments import run_one
r = resolve_run(sys.argv[1])
print(canonical_json({'workload': r.workload, 'digest': r.workload_sha256, 'seeds': r.seeds, 'result': run_one(r).simulation_result}))
"""
    outputs = []
    for seed, cwd in (("1", ROOT), ("17", tmp_path), ("321", tmp_path.parent)):
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script, str(path)],
                cwd=cwd,
                env=dict(os.environ, PYTHONHASHSEED=seed),
                text=True,
            )
        )
    assert outputs[0] == outputs[1] == outputs[2]
    executed = subprocess.run(
        [sys.executable, "-m", "smartsom.experiments.cli", "validate", str(path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert executed.returncode == 0
    assert "valid workload_sha256=" in executed.stdout


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
