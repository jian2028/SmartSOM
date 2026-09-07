import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
from test_arrivals import REFERENCE, arrival_case
from test_experiments import ROOT, edit, json_file, json_lines, run_path
from test_experiments import bundle as bundle

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.arrivals import arrival_rows, read_arrivals
from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.models import AlgorithmFile
from smartsom.config.seeds import derive_seeds
from smartsom.dispatch import WaitNextEvent
from smartsom.domain import JobArrival, ScheduledOperation
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.workloads.arrivals import UniformReleaseProfile, generate_arrivals
from smartsom.workloads.static_jsp import IntegerRange


@pytest.mark.parametrize(
    "suffix,trigger", [("dispatch", "dispatch_available"), ("event", "arrival_event")]
)
def test_config_matches_code_observations_and_frozen_reference(bundle, suffix, trigger):
    path = run_path(bundle, "online_arrivals_" + suffix)
    resolved = resolve_run(path)
    factory, workload, plan = arrival_case()
    assert (resolved.factory, resolved.workload, resolved.arrivals) == (
        factory,
        workload,
        plan,
    )
    assert not (bundle / "runs").exists()
    actual = run_one(resolved)
    views = []

    class ObservedSPT:
        def select_action(self, context):
            views.append(primitive(context))
            return SPTPolicy().select_action(context)

    expected = Simulator(
        factory, workload, arrivals=plan, decision_trigger=trigger
    ).run(ObservedSPT())
    assert actual.simulation_result == expected
    assert expected == replay(
        factory, workload, expected.actions, arrivals=plan, decision_trigger=trigger
    )
    assert expected.schedule == REFERENCE
    assert json_lines(actual.run_dir, "observations.jsonl") == views
    assert [job["job_id"] for job in views[0]["jobs"]] == ["A"]
    assert any(not view["candidates"] for view in views) == (suffix == "event")
    assert read_arrivals(actual.run_dir / "realized_events.jsonl")[0] == plan
    manifest = json_file(actual.run_dir, "manifest.json")
    assert manifest["arrivals_sha256"] == digest(plan)
    assert manifest["decision_trigger"] == trigger
    assert (
        manifest["sources"][-1]["sha256"]
        == hashlib.sha256(
            (bundle / "data/event_sets/online_arrivals.jsonl").read_bytes()
        ).hexdigest()
    )
    assert not any(seed.consumed for seed in resolved.seeds)
    for artifact, sha in manifest["artifacts"].items():
        assert (
            hashlib.sha256((actual.run_dir / artifact).read_bytes()).hexdigest() == sha
        )
    reference = ROOT / "data/reference/online_arrivals"
    source = json_file(reference, "sources.json")
    raw = json_file(reference, "direct_solver_result.json")
    assert (
        hashlib.sha256(
            (reference / "direct_solver_result.json").read_bytes()
        ).hexdigest()
        == source["result_sha256"]
    )
    external = tuple(
        sorted(
            (
                ScheduledOperation(
                    f"{'AB'[row['job']]}{row['position'] + 1}",
                    "standard",
                    f"M{row['machine'] + 1}",
                    row["start"],
                    row["end"],
                )
                for row in raw["schedule"]
            ),
            key=lambda entry: (entry.start_time, entry.operation_id),
        )
    )
    fixed = tuple(
        ScheduledOperation(**row) for row in json_file(reference, "schedule.json")
    )
    assert external == fixed == REFERENCE
    assert (
        replay_schedule(
            factory, workload, fixed, arrivals=plan, decision_trigger=trigger
        ).makespan
        == raw["makespan"]
        == 6
    )


def test_generated_profile_golden_and_rng_isolation(bundle):
    _, workload, _ = arrival_case()
    seed = next(
        seed.value
        for seed in derive_seeds(42, generated=False)
        if seed.domain == "demand"
    )
    assert seed == 1575026194469689329
    profile = UniformReleaseProfile(0, IntegerRange(2, 5), 1)
    before = random.getstate()
    plan = generate_arrivals(workload, profile, seed)
    assert random.getstate() == before
    # Fixed v1 draw order: sorted A then B, one randint per noninitial job.
    assert plan.jobs == (JobArrival("A", 2, 1), JobArrival("B", 3, 2))
    for seed in range(15):
        for initial in range(3):
            for notice in (0, 1, 100):
                profile = UniformReleaseProfile(initial, IntegerRange(1, 4), notice)
                result = generate_arrivals(workload, profile, seed)
                assert sum(row.release_at == 0 for row in result.jobs) == initial
                assert all(1 <= row.release_at <= 4 for row in result.jobs[initial:])
                assert all(
                    row.reveal_at == max(0, row.release_at - notice)
                    for row in result.jobs
                )
                shuffled = replace(
                    workload,
                    orders=(
                        replace(
                            workload.orders[0],
                            jobs=tuple(reversed(workload.orders[0].jobs)),
                        ),
                    ),
                )
                assert result == generate_arrivals(shuffled, profile, seed)
    resolved = resolve_run(run_path(bundle, "generated_arrivals_event"))
    assert resolved.arrivals == plan
    assert [seed.domain for seed in resolved.seeds if seed.consumed] == ["demand"]
    with pytest.raises(FrozenInstanceError):
        resolved.arrivals.jobs = ()
    with pytest.raises(FrozenInstanceError):
        resolved.scenario.arrivals.profile.notice_ticks = 2


@pytest.mark.parametrize(
    "initial,window,notice",
    [
        (-1, (1, 2), 0),
        (True, (1, 2), 0),
        (1.0, (1, 2), 0),
        (0, (0, 2), 0),
        (0, (3, 2), 0),
        (0, (1, 2), -1),
        (0, (1, 2), False),
    ],
)
def test_invalid_generator_ranges(initial, window, notice):
    with pytest.raises(ValueError):
        UniformReleaseProfile(initial, IntegerRange(*window), notice)


def test_all_initial_and_import_do_not_consume_demand_and_export_is_frozen(bundle):
    path = run_path(bundle, "generated_arrivals_event")
    generated = resolve_run(path)
    result = run_one(generated)
    scenario = bundle / "configs/scenarios/generated_arrivals_event.yaml"
    edit(
        scenario,
        lambda data: data.update(
            arrivals={
                "kind": "fixed",
                "path": str(result.run_dir / "realized_events.jsonl"),
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
    assert (
        imported.workload_sha256,
        imported.arrivals_sha256,
        imported.arrival_provenance,
    ) == (
        generated.workload_sha256,
        generated.arrivals_sha256,
        generated.arrival_provenance,
    )
    assert imported.arrivals == generated.arrivals
    assert not any(seed.consumed for seed in imported.seeds)
    # The resolved snapshot no longer depends on authoring files.
    for source in imported.sources:
        source.path.unlink()
    assert run_one(imported).simulation_result.makespan > 0


def test_all_initial_and_fixed_window_consumption(bundle):
    scenario = bundle / "configs/scenarios/generated_arrivals_dispatch.yaml"
    path = run_path(bundle, "generated_arrivals_dispatch")
    edit(scenario, lambda data: data["arrivals"]["profile"].update(initial_job_count=2))
    resolved = resolve_run(path)
    assert not any(seed.consumed for seed in resolved.seeds)
    assert all(row.release_at == row.reveal_at == 0 for row in resolved.arrivals.jobs)
    edit(
        scenario,
        lambda data: data["arrivals"]["profile"].update(
            initial_job_count=0, release_window={"min": 3, "max": 3}
        ),
    )
    resolved = resolve_run(path)
    assert all(row.release_at == 3 for row in resolved.arrivals.jobs)
    assert next(seed.consumed for seed in resolved.seeds if seed.domain == "demand")
    edit(scenario, lambda data: data["arrivals"]["profile"].update(initial_job_count=3))
    with pytest.raises(ConfigurationError, match="exceeds"):
        resolve_run(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0]),
        lambda rows: rows[0].update(job_id="unknown"),
        lambda rows: rows[0].update(release_at=True),
        lambda rows: rows[0].update(reveal_at=0.0),
        lambda rows: rows[0].update(release_at=-1),
        lambda rows: rows[0].update(reveal_at=10),
        lambda rows: rows[0].update(schema="unknown/v1"),
        lambda rows: rows[0].update(seed=42),
    ],
)
def test_invalid_fixed_table_fails_before_simulator_or_directory(
    bundle, change, monkeypatch
):
    import smartsom.experiments.runner as runner

    monkeypatch.setattr(
        runner, "Simulator", lambda *args, **kwargs: pytest.fail("Simulator created")
    )
    table = bundle / "data/event_sets/online_arrivals.jsonl"
    rows = [json.loads(line) for line in table.read_text().splitlines()]
    change(rows)
    table.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "online_arrivals_event"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "contents", [b"", b"\xff", b'{"job_id":"A","job_id":"B"}', b"[]", b"null"]
)
def test_jsonl_decoding_is_strict(tmp_path, contents):
    table = tmp_path / "input"
    table.write_bytes(contents)
    with pytest.raises(ConfigurationError):
        read_arrivals(table)


@pytest.mark.parametrize(
    "patch",
    [
        {"visibility": "full_static"},
        {"arrivals": None},
        {"arrivals": {"kind": "fixed", "path": "absent"}},
        {
            "arrivals": {
                "kind": "uniform_release_v1",
                "profile": {
                    "initial_job_count": 0,
                    "release_window": {"min": 1, "max": 2},
                    "seed": 1,
                },
            }
        },
        {"arrivals": {"kind": "fixed", "path": "x", "profile": {}}},
    ],
)
def test_bad_scenario_and_missing_sources(bundle, patch):
    edit(
        bundle / "configs/scenarios/online_arrivals_event.yaml",
        lambda data: data.update(patch),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "online_arrivals_event"))
    assert not (bundle / "runs").exists()


def test_dynamic_cp_is_rejected_even_for_zero_arrivals(bundle):
    path = run_path(bundle, "online_arrivals_event")
    table = bundle / "data/event_sets/online_arrivals.jsonl"
    table.write_text(
        "\n".join(
            canonical_json(row.model_copy(update={"release_at": 0, "reveal_at": 0}))
            for row in arrival_rows(arrival_case()[2], None)
        )
    )
    edit(path, lambda data: data.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="does not support arrivals"):
        resolve_run(path)
    assert not (bundle / "runs").exists()


def test_explicit_wait_action_and_script_failure_evidence(bundle):
    path = run_path(bundle, "online_arrivals_event")
    algorithm = bundle / "configs/algorithms/spt.yaml"
    data = {
        "schema": "smartsom.algorithm/v1",
        "algorithm": {
            "provider": "builtin.scripted",
            "parameters": {
                "actions": [
                    {"operation_id": "A1", "processing_mode_id": "standard"},
                    {"kind": "wait_next_event"},
                ]
            },
        },
    }
    algorithm.write_text(json.dumps(data))
    resolved = resolve_run(path)
    assert isinstance(
        resolved.algorithm.algorithm.parameters.actions[-1], WaitNextEvent
    )
    with pytest.raises(RunFailedError, match="exhausted") as error:
        run_one(resolved)
    directory = error.value.run_dir
    assert json_file(directory, "summary.json")["makespan"] is None
    assert [
        row["simulation_time"] for row in json_lines(directory, "observations.jsonl")
    ] == [0, 1, 2]
    assert any(row["kind"] == "wait" for row in json_lines(directory, "trace.jsonl"))
    assert (directory / "realized_events.jsonl").exists()
    # Runner cannot silently synthesize a missing wait on the empty tick-1 view.
    data["algorithm"]["parameters"]["actions"][1] = {
        "operation_id": "B1",
        "processing_mode_id": "standard",
    }
    algorithm.write_text(json.dumps(data))
    with pytest.raises(RunFailedError, match="not been released"):
        run_one(resolve_run(path))
    for action in ({}, {"kind": "wait_next_event", "until": 3}, {"kind": "unknown"}):
        data["algorithm"]["parameters"]["actions"] = [action]
        with pytest.raises(ValueError):
            AlgorithmFile.model_validate_json(json.dumps(data))


def test_cli_and_hash_seed_cwd_determinism(bundle, monkeypatch):
    path = run_path(bundle, "generated_arrivals_event")
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    outputs = []
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import canonical_json; from smartsom.algorithms import SPTPolicy; from smartsom.engine import Simulator; import sys; r=resolve_run(sys.argv[1]); print(canonical_json([r.arrivals,Simulator(r.factory,r.workload,arrivals=r.arrivals,decision_trigger=r.scenario.decision_trigger).run(SPTPolicy())]))"
    for seed, cwd in [("1", ROOT), ("99", bundle)]:
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code, str(path)],
                cwd=cwd,
                env={**os.environ, "PYTHONHASHSEED": seed},
                text=True,
            )
        )
    assert outputs[0] == outputs[1]
    monkeypatch.chdir(bundle)
    assert main(["run", str(path)]) == 0
    edit(
        bundle / "configs/scenarios/generated_arrivals_event.yaml",
        lambda data: data.update(seed=1),
    )
    assert main(["validate", str(path)]) != 0


@pytest.mark.parametrize("name", ["generated", "generated_fjsp_spt"])
def test_arrivals_do_not_change_workload_generator_outputs(bundle, name):
    path = run_path(bundle, name)
    original = resolve_run(path)
    scenario = next(
        source.path for source in original.sources if source.role == "scenario"
    )
    edit(
        scenario,
        lambda data: data.update(
            arrivals={
                "kind": "uniform_release_v1",
                "profile": {
                    "initial_job_count": 0,
                    "release_window": {"min": 1, "max": 5},
                },
            },
            visibility="decision_context",
        ),
    )
    dynamic = resolve_run(path)
    assert dynamic.workload == original.workload
    assert dynamic.workload_sha256 == original.workload_sha256
    assert dynamic.provenance == original.provenance
    assert [seed.domain for seed in dynamic.seeds if seed.consumed] == [
        "workload",
        "demand",
    ]
    assert run_one(dynamic).simulation_result.makespan > 0


def test_failure_on_empty_observation_preserves_inputs_and_observed_prefix(
    bundle, monkeypatch
):
    import smartsom.experiments.runner as runner

    class BrokenPolicy:
        def select_action(self, context):
            if not context.candidates:
                raise RuntimeError("failure on empty view")
            return SPTPolicy().select_action(context)

    monkeypatch.setattr(runner, "_policy", lambda resolved: BrokenPolicy())
    resolved = resolve_run(run_path(bundle, "online_arrivals_event"))
    with pytest.raises(RunFailedError, match="failure on empty view") as error:
        run_one(resolved)
    directory = error.value.run_dir
    views = json_lines(directory, "observations.jsonl")
    assert len(views) == 2 and views[-1]["simulation_time"] == 1
    assert not views[-1]["candidates"]
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert [row["kind"] for row in json_lines(directory, "trace.jsonl")] == [
        "decision",
        "dispatch",
        "reveal",
        "decision",
    ]
    assert read_arrivals(directory / "realized_events.jsonl")[0] == resolved.arrivals


def test_fixed_table_provenance_conflicts_and_reordering(bundle):
    resolved = resolve_run(run_path(bundle, "generated_arrivals_event"))
    rows = arrival_rows(resolved.arrivals, resolved.arrival_provenance)
    table = bundle / "timing.jsonl"
    table.write_text("\n".join(canonical_json(row) for row in reversed(rows)))
    plan, provenance, raw_sha = read_arrivals(table)
    assert plan == resolved.arrivals and provenance == resolved.arrival_provenance
    table.write_text("\n".join(canonical_json(row) for row in rows))
    assert read_arrivals(table)[0] == plan
    assert read_arrivals(table)[2] != raw_sha
    rows = (rows[0], rows[1].model_copy(update={"provenance": None}))
    table.write_text("\n".join(canonical_json(row) for row in rows))
    with pytest.raises(ConfigurationError, match="provenance must agree"):
        read_arrivals(table)
