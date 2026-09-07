import hashlib
import json
import os
import subprocess
import sys

import pytest
from test_experiments import ROOT, edit, json_file, json_lines, run_path
from test_experiments import bundle as bundle
from test_processing_times import GOLDEN_DRAWS, REFERENCE, processing_case

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import ProcessingTimeFile
from smartsom.domain import ScheduledOperation
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main


def test_fixed_config_observations_and_external_reference(bundle):
    resolved = resolve_run(run_path(bundle, "processing_fixed"))
    factory, workload, plan = processing_case()
    assert (resolved.factory, resolved.workload, resolved.processing_times) == (
        factory,
        workload,
        plan,
    )
    assert not (bundle / "runs").exists()
    views = []

    class Observe:
        def select_action(self, context):
            views.append(primitive(context))
            return SPTPolicy().select_action(context)

    direct = Simulator(factory, workload, processing_times=plan).run(Observe())
    result = run_one(resolved)
    assert (
        result.simulation_result
        == direct
        == replay(factory, workload, direct.actions, processing_times=plan)
    )
    assert direct.schedule == REFERENCE and direct.makespan == 20
    assert json_lines(result.run_dir, "observations.jsonl") == views
    assert all("actual_ticks" not in json.dumps(view) for view in views)
    assert (
        read_model(
            result.run_dir / "realized_processing_times.json", ProcessingTimeFile
        )[0].processing_times
        == plan
    )
    manifest = json_file(result.run_dir, "manifest.json")
    assert manifest["processing_times_sha256"] == digest(plan)
    assert (
        manifest["sources"][-1]["sha256"]
        == hashlib.sha256(
            (bundle / "data/processing_times/hand.json").read_bytes()
        ).hexdigest()
    )
    assert not any(seed.consumed for seed in resolved.seeds)
    source_dir = ROOT / "data/reference/processing_times"
    sources = json_file(source_dir, "sources.json")
    external = json_file(source_dir, "direct_solver_result.json")
    assert (
        sources["result_sha256"]
        == hashlib.sha256(
            (source_dir / "direct_solver_result.json").read_bytes()
        ).hexdigest()
    )
    translated = tuple(
        sorted(
            (
                ScheduledOperation(
                    f"{'AB'[row['job']]}{row['position'] + 1}",
                    "standard",
                    f"M{row['machine'] + 1}",
                    row["start"],
                    row["end"],
                )
                for row in external["schedule"]
            ),
            key=lambda row: (row.start_time, row.operation_id),
        )
    )
    fixed = tuple(
        ScheduledOperation(**row) for row in json_file(source_dir, "schedule.json")
    )
    assert translated == fixed == direct.schedule
    assert external["metadata"]["status"] == "optimal"
    assert (
        external["makespan"]
        == replay_schedule(factory, workload, fixed, processing_times=plan).makespan
        == 20
    )


def test_generated_provenance_golden_and_export_reimport(bundle):
    path = run_path(bundle, "processing_generated")
    resolved = resolve_run(path)
    provenance = resolved.processing_provenance
    assert provenance.effective_seed == 3797569027003476775
    assert [row.draw for row in provenance.draws] == GOLDEN_DRAWS
    assert [row.actual_ticks for row in resolved.processing_times.modes] == [
        11,
        6,
        4,
        12,
    ]
    assert [seed.domain for seed in resolved.seeds if seed.consumed] == [
        "processing_time"
    ]
    result = run_one(resolved)
    assert result.simulation_result.makespan == 23
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda data: data.update(
            processing_time={
                "kind": "fixed",
                "path": str(result.run_dir / "realized_processing_times.json"),
            },
            workload={
                "kind": "instance",
                "path": str(result.run_dir / "realized_instance.json"),
            },
        ),
    )
    edit(
        path,
        lambda data: data.update(
            seed=99, algorithm="../algorithms/first_feasible.yaml"
        ),
    )
    imported = resolve_run(path)
    assert imported.workload_sha256 == resolved.workload_sha256
    assert imported.processing_times_sha256 == resolved.processing_times_sha256
    assert imported.processing_provenance == resolved.processing_provenance
    assert not any(seed.consumed for seed in imported.seeds)
    for source in imported.sources:
        source.path.unlink()
    assert (
        run_one(imported).simulation_result.schedule
        == result.simulation_result.schedule
    )


@pytest.mark.parametrize(
    "name", ["generated", "generated_fjsp_spt", "generated_arrivals_event"]
)
def test_seed_and_workload_independence_across_existing_generators(bundle, name):
    path = run_path(bundle, name)
    old = resolve_run(path)
    scenario = next(source.path for source in old.sources if source.role == "scenario")
    edit(
        scenario,
        lambda data: data.update(
            visibility="decision_context",
            processing_time={"kind": "uniform_multiplier"},
        ),
    )
    new = resolve_run(path)
    assert new.workload == old.workload and new.workload_sha256 == old.workload_sha256
    assert (
        new.arrivals == old.arrivals
        and new.arrival_provenance == old.arrival_provenance
    )
    assert [
        (seed.domain, seed.value, seed.consumed)
        for seed in old.seeds
        if seed.domain != "processing_time"
    ] == [
        (seed.domain, seed.value, seed.consumed)
        for seed in new.seeds
        if seed.domain != "processing_time"
    ]
    assert run_one(new).simulation_result.makespan > 0


@pytest.mark.parametrize("trigger", ["dispatch", "event"])
def test_combination_evidence_and_unit_equivalence(bundle, trigger):
    path = run_path(bundle, "processing_arrivals_" + trigger)
    resolved = resolve_run(path)
    result = run_one(resolved)
    assert (result.run_dir / "realized_events.jsonl").exists()
    assert (result.run_dir / "realized_processing_times.json").exists()
    assert result.simulation_result == replay(
        resolved.factory,
        resolved.workload,
        result.simulation_result.actions,
        arrivals=resolved.arrivals,
        decision_trigger=resolved.scenario.decision_trigger,
        processing_times=resolved.processing_times,
    )
    scenario = bundle / f"configs/scenarios/processing_arrivals_{trigger}.yaml"
    edit(
        scenario,
        lambda data: data.update(
            processing_time={
                "kind": "uniform_multiplier",
                "profile": {"low": 1, "high": 1},
            }
        ),
    )
    unit = resolve_run(path)
    baseline = Simulator(
        unit.factory,
        unit.workload,
        arrivals=unit.arrivals,
        decision_trigger=unit.scenario.decision_trigger,
    ).run(SPTPolicy())
    assert run_one(unit).simulation_result == baseline
    assert not any(seed.consumed for seed in unit.seeds)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["processing_times"]["modes"].pop(),
        lambda d: d["processing_times"]["modes"].append(
            d["processing_times"]["modes"][0]
        ),
        lambda d: d["processing_times"]["modes"][0].update(operation_id="missing"),
        lambda d: d["processing_times"]["modes"][0].update(
            processing_mode_id="missing"
        ),
        lambda d: d["processing_times"]["modes"][0].update(nominal_ticks=99),
        lambda d: d["processing_times"]["modes"][0].update(actual_ticks=True),
        lambda d: d["processing_times"]["modes"][0].update(actual_ticks=1.0),
        lambda d: d["processing_times"]["modes"][0].update(actual_ticks=0),
        lambda d: d.update(seed=1),
        lambda d: d.update(schema="unsupported/v1"),
    ],
)
def test_invalid_fixed_inputs_fail_before_directory(bundle, mutation, monkeypatch):
    import smartsom.experiments.runner as runner

    monkeypatch.setattr(
        runner, "Simulator", lambda *a, **kw: pytest.fail("simulator created")
    )
    table = bundle / "data/processing_times/hand.json"

    def change(data):
        data.pop("content_sha256")
        mutation(data)

    edit(table, change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "processing_fixed"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "processing",
    [
        {"kind": "uniform_multiplier", "profile": {"low": False, "high": 1}},
        {"kind": "uniform_multiplier", "profile": {"low": 0, "high": 1}},
        {"kind": "uniform_multiplier", "profile": {"low": 2, "high": 1}},
        {"kind": "uniform_multiplier", "profile": {"low": "NaN", "high": 1}},
        {"kind": "uniform_multiplier", "profile": {"low": 1, "high": "Infinity"}},
        {"kind": "uniform_multiplier", "profile": {"seed": 42}},
        {"kind": "uniform_multiplier", "sample_timing": "dispatch"},
        {"kind": "fixed", "path": "absent"},
        {"kind": "fixed", "path": "x", "profile": {}},
    ],
)
def test_invalid_profile_and_references(bundle, processing):
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(processing_time=processing),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "processing_generated"))
    assert not (bundle / "runs").exists()


def test_static_solver_and_full_visibility_rejected_even_with_unit_multiplier(bundle):
    path = run_path(bundle, "processing_generated")
    scenario = bundle / "configs/scenarios/processing_generated.yaml"
    edit(
        scenario,
        lambda d: d.update(
            processing_time={
                "kind": "uniform_multiplier",
                "profile": {"low": 1, "high": 1},
            }
        ),
    )
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(
        ConfigurationError, match="does not support processing uncertainty"
    ):
        resolve_run(path)
    edit(scenario, lambda d: d.update(visibility="full_static"))
    with pytest.raises(ConfigurationError, match="requires decision_context"):
        resolve_run(path)
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("tamper", ["content", "draw", "profile", "coverage", "actual"])
def test_imported_provenance_is_verified(bundle, tamper):
    result = run_one(resolve_run(run_path(bundle, "processing_generated")))
    table = result.run_dir / "realized_processing_times.json"

    def change(data):
        if tamper == "content":
            data["content_sha256"] = "0" * 64
        if tamper == "draw":
            data["provenance"]["draws"][0]["draw"] = -1
        if tamper == "profile":
            data["provenance"]["profile"]["high"] = "1.5"
        if tamper == "coverage":
            data["provenance"]["draws"].pop()
        if tamper == "actual":
            data.pop("content_sha256")
            data["processing_times"]["modes"][0]["actual_ticks"] += 1

    edit(table, change)
    with pytest.raises(ConfigurationError):
        read_model(table, ProcessingTimeFile)


def test_duplicate_keys_and_canonical_order(bundle):
    table = bundle / "data/processing_times/hand.json"
    before = resolve_run(run_path(bundle, "processing_fixed"))
    edit(table, lambda d: d["processing_times"]["modes"].reverse())
    after = resolve_run(run_path(bundle, "processing_fixed"))
    assert before.processing_times == after.processing_times
    assert before.processing_times_sha256 == after.processing_times_sha256
    assert before.sources[-1].sha256 != after.sources[-1].sha256
    table.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ConfigurationError, match="duplicate key"):
        resolve_run(run_path(bundle, "processing_fixed"))


def test_failed_policy_preserves_actual_inputs_and_only_delivered_views(
    bundle, monkeypatch
):
    import smartsom.experiments.runner as runner

    class Broken:
        def __init__(self):
            self.count = 0

        def select_action(self, context):
            self.count += 1
            if self.count == 3:
                raise RuntimeError("intentional policy failure")
            return SPTPolicy().select_action(context)

    monkeypatch.setattr(runner, "_policy", lambda resolved: Broken())
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle, "processing_fixed")))
    directory = error.value.run_dir
    assert (directory / "realized_processing_times.json").exists()
    assert len(json_lines(directory, "observations.jsonl")) == 3
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert all(
        "actual_ticks" not in json.dumps(row)
        for row in json_lines(directory, "observations.jsonl")
    )
    assert any(
        row["kind"] == "complete" for row in json_lines(directory, "trace.jsonl")
    )


def test_cli_hash_seed_cwd_and_decimal_equivalence(bundle):
    path = run_path(bundle, "processing_generated")
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import canonical_json; import sys; r=resolve_run(sys.argv[1]); print(canonical_json([r.processing_times,r.processing_provenance]))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for cwd, seed in [(ROOT, "1"), (bundle, "99")]
    ]
    assert outputs[0] == outputs[1]
    before = resolve_run(path)
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d["processing_time"].update(
            profile={"low": "0.800", "high": "1.200"}
        ),
    )
    after = resolve_run(path)
    assert before.processing_provenance == after.processing_provenance
    assert main(["run", str(path)]) == 0
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(processing_time={"kind": "unknown"}),
    )
    assert main(["validate", str(path)]) != 0


def test_fixed_import_and_run_do_not_draw_randomness(bundle, monkeypatch):
    import smartsom.workloads.processing_times as generation

    resolved = resolve_run(run_path(bundle, "processing_generated"))
    run = run_one(resolved)
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(
            processing_time={
                "kind": "fixed",
                "path": str(run.run_dir / "realized_processing_times.json"),
            }
        ),
    )
    monkeypatch.setattr(
        generation.random,
        "Random",
        lambda *a: pytest.fail("fixed import/run must not draw"),
    )
    imported = resolve_run(run_path(bundle, "processing_generated"))
    assert run_one(imported).simulation_result == run.simulation_result
