"""Strict quality authoring, materialization, pairing and durable run evidence."""

import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml
from test_quality import hand, modes

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import primitive
from smartsom.config.models import QualityFile
from smartsom.config.seeds import derive_seeds
from smartsom.domain.quality import (
    QualityMode,
    QualitySpeedSpec,
    quality_mode_id,
)
from smartsom.engine import Simulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.modules.quality import prepare_quality
from smartsom.workloads.quality import generate_quality, operation_draw

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def files(tmp_path):
    sources = {
        "factory.yaml": "configs/factories/quality.yaml",
        "workload.json": "data/reference/quality/workload.json",
        "quality.json": "data/reference/quality/draws.json",
        "algorithm.yaml": "configs/algorithms/spt_quality_m0.yaml",
    }
    for dest, source in sources.items():
        (tmp_path / dest).write_bytes((ROOT / source).read_bytes())
    write(
        tmp_path / "scenario.yaml",
        {
            "schema": "smartsom.scenario/v1",
            "factory": "factory.yaml",
            "workload": {"kind": "instance", "path": "workload.json"},
            "quality": {"kind": "fixed", "path": "quality.json"},
        },
    )
    write(
        tmp_path / "run.yaml",
        {
            "schema": "smartsom.run/v1",
            "scenario": "scenario.yaml",
            "algorithm": "algorithm.yaml",
            "seed": 42,
            "output_root": "runs",
        },
    )
    return tmp_path


def write(path, data):
    path.write_text(json.dumps(data))


def edit(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    write(path, data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_scale", True),
        ("time_scale", 0),
        ("time_scale", -1),
        ("time_scale", "nan"),
        ("time_scale", "inf"),
        ("error_rate", False),
        ("error_rate", -0.1),
        ("error_rate", 1.1),
        ("error_rate", "nan"),
        ("error_rate", "-Infinity"),
        ("time_scale", "oops"),
        ("quality_mode_id", ""),
    ],
)
def test_invalid_mode_values_fail_before_simulator_or_directory(
    files, monkeypatch, field, value
):
    edit(
        files / "factory.yaml",
        lambda d: d["factory"]["quality_speed"]["default_modes"][0].update(
            {field: value}
        ),
    )
    monkeypatch.setattr(
        "smartsom.experiments.runner.Simulator",
        lambda *a, **k: pytest.fail("constructed Simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(files / "run.yaml")
    assert main(["validate", str(files / "run.yaml")]) == 2
    assert not (files / "runs").exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing_table",
        "unknown_machine",
        "duplicate_machine",
        "duplicate_mode",
        "unknown_field",
        "unknown_probability_visibility",
        "unknown_kind",
        "unknown_quality_operation",
        "missing_draw",
        "duplicate_draw",
        "bool_draw",
        "float_draw",
        "negative_draw",
        "overflow_draw",
        "bad_hash",
        "bad_provenance",
        "seed_in_scenario",
        "factory_off",
    ],
)
def test_invalid_structure_and_references(files, case):
    f = yaml.safe_load((files / "factory.yaml").read_text())
    s = yaml.safe_load((files / "scenario.yaml").read_text())
    q = yaml.safe_load((files / "quality.json").read_text())
    speed = f["factory"]["quality_speed"]
    if case == "missing_table":
        speed.update(
            default_modes=[],
            machine_modes=[{"machine_id": "M1", "modes": modes_primitive()}],
        )
    if case == "unknown_machine":
        speed["machine_modes"] = [{"machine_id": "unknown", "modes": modes_primitive()}]
    if case == "duplicate_machine":
        speed["machine_modes"] = [{"machine_id": "M1", "modes": modes_primitive()}] * 2
    if case == "duplicate_mode":
        speed["default_modes"].append(speed["default_modes"][0])
    if case == "unknown_field":
        speed["default_modes"][0]["typo"] = 1
    if case == "unknown_probability_visibility":
        s["quality"]["probability_visibility"] = "sometimes"
    if case == "unknown_kind":
        s["quality"]["kind"] = "random"
    if case == "unknown_quality_operation":
        q["draws"]["operations"][0]["operation_id"] = "unknown"
    if case == "missing_draw":
        q["draws"]["operations"].pop()
    if case == "duplicate_draw":
        q["draws"]["operations"].append(q["draws"]["operations"][0])
    if case in ("bool_draw", "float_draw", "negative_draw", "overflow_draw"):
        q["draws"]["operations"][0]["draw"] = {
            "bool_draw": True,
            "float_draw": 1.0,
            "negative_draw": -1,
            "overflow_draw": 2**53,
        }[case]
    if case == "bad_hash":
        q["content_sha256"] = "0" * 64
    if case == "bad_provenance":
        q["provenance"] = {"effective_seed": 42}
    if case == "seed_in_scenario":
        s["quality"]["seed"] = 1
    if case == "factory_off":
        f["factory"].pop("quality_speed")
    write(files / "factory.yaml", f)
    write(files / "scenario.yaml", s)
    write(files / "quality.json", q)
    with pytest.raises(ConfigurationError):
        resolve_run(files / "run.yaml")
    assert not (files / "runs").exists()


def modes_primitive():
    return primitive(modes())


@pytest.mark.parametrize(
    "file,contents",
    [
        (
            "quality.json",
            '{"schema":"smartsom.quality-draws/v1","draws":{},"draws":{}}',
        ),
        ("factory.yaml", "schema: smartsom.factory/v1\nfactory: {}\nfactory: {}\n"),
    ],
)
def test_duplicate_keys(files, file, contents):
    (files / file).write_text(contents)
    with pytest.raises(ConfigurationError, match="duplicate key"):
        resolve_run(files / "run.yaml")


def test_fixed_mode_requires_every_candidate_and_enabled_quality(files):
    edit(
        files / "factory.yaml",
        lambda d: d["factory"]["quality_speed"].update(
            machine_modes=[
                {
                    "machine_id": "M2",
                    "modes": [primitive(QualityMode("custom", "1", ".01"))],
                }
            ]
        ),
    )
    with pytest.raises(ConfigurationError, match="all candidate"):
        resolve_run(files / "run.yaml")
    edit(files / "scenario.yaml", lambda d: d.update(quality=None))
    with pytest.raises(ConfigurationError, match="requires enabled quality"):
        resolve_run(files / "run.yaml")


def test_cp_rejects_quality_even_zero_probability(files):
    edit(
        files / "factory.yaml",
        lambda d: [
            row.update(error_rate="0")
            for row in d["factory"]["quality_speed"]["default_modes"]
        ],
    )
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "pyjobshop.cp_sat",
                "interface_kind": "offline_solver",
                "required_information": "full_static",
            },
        },
    )
    with pytest.raises(ConfigurationError, match="does not support quality"):
        resolve_run(files / "run.yaml")


def test_configured_hand_and_reimport_do_not_scale_twice(files):
    r = resolve_run(files / "run.yaml")
    assert main(["validate", str(files / "run.yaml")]) == 0
    assert not (files / "runs").exists()
    result = run_one(r)
    folder = result.run_dir
    f, w, d = hand()
    direct = Simulator(f, w, quality=prepare_quality(f, w, d)).run(SPTPolicy("M0"))
    assert result.simulation_result == direct
    summary = json.loads((folder / "summary.json").read_text())
    assert (summary["makespan"], summary["passed_jobs"], summary["passing_rate"]) == (
        24,
        1,
        1,
    )
    assert json.loads((folder / "realized_instance.json").read_text())[
        "workload"
    ] == primitive(w)
    assert (
        json.loads((folder / "effective_modes.json").read_text())["content_sha256"]
        == r.quality_modes_sha256
    )
    assert "inspected_jobs=1" in (folder / "progress.log").read_text()
    observation = json.loads(
        (folder / "observations.jsonl").read_text().splitlines()[1]
    )
    assert observation["job_quality"][0]["passed"] is None
    edit(
        files / "scenario.yaml",
        lambda s: s.update(
            workload={
                "kind": "instance",
                "path": str(folder / "realized_instance.json"),
            },
            quality={"kind": "fixed", "path": str(folder / "realized_quality.json")},
        ),
    )
    edit(files / "run.yaml", lambda d: d.update(seed=987))
    same = resolve_run(files / "run.yaml")
    assert same.workload_sha256 == r.workload_sha256
    assert same.quality_modes_sha256 == r.quality_modes_sha256
    assert same.quality_draws_sha256 == r.quality_draws_sha256
    assert next(s for s in same.seeds if s.domain == "quality").consumed is False
    assert run_one(same).simulation_result == direct
    edit(
        files / "algorithm.yaml",
        lambda d: d["algorithm"].update(parameters={"quality_mode": "M2"}),
    )
    fast = resolve_run(files / "run.yaml")
    assert fast.quality == r.quality
    assert run_one(fast).simulation_result.makespan == 16
    with pytest.raises(FrozenInstanceError):
        r.quality.modes = ()
    (files / "factory.yaml").write_text("broken")
    assert (
        run_one(r).simulation_result == direct
    )  # Resolved input never rereads sources.


def test_generated_golden_and_seed_isolation(files):
    edit(
        files / "scenario.yaml",
        lambda d: d.update(quality={"kind": "independent_operation_v1"}),
    )
    before = random.getstate()
    r = resolve_run(files / "run.yaml")
    assert random.getstate() == before
    quality_seed = next(x.value for x in r.seeds if x.domain == "quality")
    assert quality_seed == 1400014650531944465
    assert [x.draw for x in r.quality.draws.operations] == [
        2092992623612522,
        7285602162975817,
    ]
    assert (
        r.quality_draws_sha256
        == "3c16f4f62d3df9e029dcf15a616816fa35a9700eaa8158dc007025b2b513e205"
    )
    assert r.seeds[:-1] == derive_seeds(42, generated=False)
    assert operation_draw(quality_seed, "A1") == r.quality.draws.operations[0].draw
    result = run_one(r)
    fixed = QualityFile.model_validate_json(
        (result.run_dir / "realized_quality.json").read_text()
    )
    assert fixed.draws == r.quality.draws
    edit(
        files / "scenario.yaml",
        lambda d: d.update(
            quality={
                "kind": "fixed",
                "path": str(result.run_dir / "realized_quality.json"),
            }
        ),
    )
    edit(files / "run.yaml", lambda d: d.update(seed=99))
    assert resolve_run(files / "run.yaml").quality == r.quality


def test_permutation_and_unrelated_operation_do_not_change_draws():
    f, w, d = hand()
    f2 = replace(
        f,
        machines=tuple(reversed(f.machines)),
        quality_speed=QualitySpeedSpec(tuple(reversed(modes()))),
    )
    job = w.orders[0].jobs[0]
    w2 = replace(
        w,
        orders=(
            replace(
                w.orders[0],
                jobs=(replace(job, operations=tuple(reversed(job.operations))),),
            ),
        ),
    )
    a = prepare_quality(f, w, generate_quality(w, 10))
    b = prepare_quality(f2, w2, generate_quality(w2, 10))
    assert a == b
    assert Simulator(f, w, quality=a).run(SPTPolicy()) == Simulator(
        f2, w2, quality=b
    ).run(SPTPolicy())
    extra = replace(
        job.operations[0], operation_id="unrelated", predecessor_ids=("A2",)
    )
    w3 = replace(
        w,
        orders=(
            replace(
                w.orders[0], jobs=(replace(job, operations=(*job.operations, extra)),)
            ),
        ),
    )
    draws = {x.operation_id: x.draw for x in generate_quality(w3, 10).operations}
    assert all(draws[x.operation_id] == x.draw for x in a.draws.operations)
    assert quality_mode_id("a/b", "c") != quality_mode_id("a", "b/c")


def test_hash_seed_and_working_directory(files):
    edit(
        files / "scenario.yaml",
        lambda d: d.update(quality={"kind": "independent_operation_v1"}),
    )
    program = """from smartsom.config import resolve_run
from smartsom.config.codec import digest
from smartsom.engine import Simulator
from smartsom.algorithms import SPTPolicy
import sys
r=resolve_run(sys.argv[1]);s=Simulator(r.factory,r.workload,quality=r.quality).run(SPTPolicy("M0"))
print(digest((r.quality,s)))"""
    values = []
    for cwd, seed in [(ROOT, "1"), (files, "82")]:
        values.append(
            subprocess.check_output(
                [sys.executable, "-c", program, str(files / "run.yaml")],
                cwd=cwd,
                env={
                    **os.environ,
                    "PYTHONHASHSEED": seed,
                    "PYTHONPATH": str(ROOT / "src"),
                },
                text=True,
            )
        )
    assert values[0] == values[1]


@pytest.mark.parametrize("failure", ["script", "provider", "writer"])
def test_failure_retains_quality_inputs_and_partial_evidence(
    files, monkeypatch, failure
):
    if failure == "script":
        write(
            files / "algorithm.yaml",
            {
                "schema": "smartsom.algorithm/v1",
                "algorithm": {
                    "provider": "builtin.scripted",
                    "parameters": {
                        "actions": [
                            {
                                "operation_id": "A1",
                                "processing_mode_id": quality_mode_id("base", "M1"),
                            }
                        ]
                    },
                },
            },
        )
    if failure == "provider":
        monkeypatch.setattr(
            "smartsom.algorithms.spt.SPTPolicy.select_action",
            lambda *a: (_ for _ in ()).throw(ValueError("provider exploded")),
        )
    if failure == "writer":
        import smartsom.experiments.evidence as evidence

        original = evidence.append_json

        def broken(stream, row):
            if getattr(row, "kind", None) == "quality_check":
                raise OSError("quality write failed")
            return original(stream, row)

        monkeypatch.setattr(evidence, "append_json", broken)
    with pytest.raises(RunFailedError) as exc:
        run_one(resolve_run(files / "run.yaml"))
    folder = exc.value.run_dir
    assert (folder / "realized_quality.json").exists() and (
        folder / "effective_modes.json"
    ).exists()
    assert (folder / "failure.json").exists()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["makespan"] is None and "passing_rate" not in summary
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    if failure != "provider":
        assert (folder / "trace.jsonl").read_text()


def test_cli_success_and_execution_failure(files):
    assert main(["run", str(files / "run.yaml")]) == 0
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "builtin.scripted",
                "parameters": {"actions": []},
            },
        },
    )
    assert main(["run", str(files / "run.yaml")]) == 1


def test_exhausted_run_with_extra_script_action_is_not_quality_success(files):
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "builtin.scripted",
                "parameters": {
                    "actions": [
                        {
                            "operation_id": op,
                            "processing_mode_id": quality_mode_id("base", "M0"),
                        }
                        for op in ("A1", "A2", "A1")
                    ]
                },
            },
        },
    )
    with pytest.raises(RunFailedError) as exc:
        run_one(resolve_run(files / "run.yaml"))
    folder = exc.value.run_dir
    records = [json.loads(x) for x in (folder / "trace.jsonl").read_text().splitlines()]
    assert sum(x["kind"] == "inspection" for x in records) == 1
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["makespan"] is None
    assert "passing_rate" not in summary
