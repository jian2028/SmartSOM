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
from test_arrivals import arrival_case
from test_experiments import ROOT, edit, json_file, json_lines, run_path
from test_experiments import bundle as bundle

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.arrivals import arrival_rows, read_arrivals
from smartsom.config.codec import canonical_json, primitive
from smartsom.config.models import AlgorithmFile
from smartsom.config.production import named_seed
from smartsom.config.seeds import derive_seeds
from smartsom.domain import JobArrival
from smartsom.domain.arrivals import ArrivalPlan
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import audit
from smartsom.workloads.arrivals import UniformReleaseProfile, generate_arrivals
from smartsom.workloads.static_jsp import IntegerRange


@pytest.mark.parametrize("suffix", ["dispatch", "event"])
def test_config_matches_code_observations_and_frozen_reference(bundle, suffix):
    path = run_path(bundle, "online_arrivals_" + suffix)
    prepared = resolve_run(path)
    case = prepared.resolved.scenario
    assert {(d.demand_id, d.release_at, d.reveal_at) for d in case.demands} == {
        ("A", 0, 0),
        ("B", 2, 1),
    }
    assert not (bundle / "runs").exists()
    actual = run_one(prepared, verbose=False)
    sim = ProductionSimulator(case)
    policy = GreedyProductionPolicy(case.factory, rule="spt")
    views, expected = [], []
    while not sim.done:
        views.append(sim.decision())
        command = policy.act(sim.decision(policy.rank(views[-1])))
        expected.append(sim.step(command))
    assert actual.simulation_result.final_state == sim.snapshot()
    rows = json_lines(actual.run_dir, "trace.jsonl")
    assert [{k: row[k] for k in expected[0]} for row in rows] == primitive(expected)
    assert {row["demand"] for row in views[0]["jobs"].values()} == {"A"}
    assert {row["demand"] for row in views[1]["jobs"].values()} == {"A"}
    assert any(row["demand"] == "B" for row in views[2]["jobs"].values())
    assert audit(actual.run_dir)["status"] == "passed"
    manifest = json_file(actual.run_dir, "run.json")
    assert manifest["inputs"]["scenario"] == primitive(case)
    # Historical external results keep their own source hash; they are not a
    # makespan oracle for newly authored grid transport.
    reference = ROOT / "data/reference/online_arrivals"
    source = json_file(reference, "sources.json")
    assert (
        hashlib.sha256(
            (reference / "direct_solver_result.json").read_bytes()
        ).hexdigest()
        == source["result_sha256"]
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
    case = resolved.resolved.scenario
    rng = random.Random(named_seed(42, "arrival"))
    expected = [(name, rng.randint(2, 5)) for name in ("A", "B")]
    assert [(d.demand_id, d.release_at) for d in case.demands] == expected
    with pytest.raises(FrozenInstanceError):
        case.demands = ()
    detached = json.loads(resolved.resolved.settings_json)
    detached["arrivals"]["notice_ticks"] = 500
    assert json.loads(resolved.resolved.settings_json)["arrivals"]["notice_ticks"] == 1


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
    result = run_one(generated, verbose=False)
    frozen = json_file(result.run_dir, "run.json")["inputs"]["scenario"]
    workload = bundle / "frozen-workload.yaml"
    workload.write_text(
        canonical_json({"schema": "smartsom.workload/v2", "demands": frozen["demands"]})
    )
    scenario = bundle / "configs/scenarios/generated_arrivals_event.yaml"
    edit(scenario, lambda data: data.update(arrivals=None, workload=str(workload)))
    edit(
        path,
        lambda data: data.update(
            seed=99, algorithm="../algorithms/first_feasible.yaml"
        ),
    )
    imported = resolve_run(path)
    assert imported.resolved.scenario.demands == generated.resolved.scenario.demands
    shutil.rmtree(bundle / "configs")
    workload.unlink()
    assert run_one(imported, verbose=False).simulation_result.status == "completed"


def test_all_initial_and_fixed_window_consumption(bundle):
    scenario = bundle / "configs/scenarios/generated_arrivals_dispatch.yaml"
    path = run_path(bundle, "generated_arrivals_dispatch")
    edit(scenario, lambda data: data["arrivals"].update(initial_jobs=2))
    case = resolve_run(path).resolved.scenario
    assert all(d.release_at == d.reveal_at == 0 for d in case.demands)
    edit(
        scenario,
        lambda data: data["arrivals"].update(
            initial_jobs=0, release_min=3, release_max=3
        ),
    )
    case = resolve_run(path).resolved.scenario
    assert all(d.release_at == 3 and d.reveal_at == 2 for d in case.demands)
    edit(scenario, lambda data: data["arrivals"].update(initial_jobs=3))
    with pytest.raises(ConfigurationError, match="initial_jobs"):
        resolve_run(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows.append(rows[0]),
        lambda rows: rows[0].update(input_id="unknown"),
        lambda rows: rows[0].update(release_at=True),
        lambda rows: rows[0].update(reveal_at=0.0),
        lambda rows: rows[0].update(release_at=-1),
        lambda rows: rows[0].update(reveal_at=10),
        lambda rows: rows[0].update(schema="unknown"),
        lambda rows: rows[0].update(seed=42),
        lambda rows: rows[0].pop("steps"),
    ],
)
def test_invalid_fixed_table_fails_before_simulator_or_directory(
    bundle, change, monkeypatch
):
    monkeypatch.setattr(
        ProductionSimulator,
        "__init__",
        lambda *a, **k: pytest.fail("Simulator created"),
    )
    table = bundle / "configs/workloads/online_arrivals_event.yaml"
    edit(table, lambda data: change(data["demands"]))
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
        {"mode": "static"},
        {"workload": "absent"},
        {
            "arrivals": {
                "initial_jobs": 0,
                "release_min": 1,
                "release_max": 2,
                "seed": 1,
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
    edit(
        bundle / "configs/workloads/online_arrivals_event.yaml",
        lambda data: [d.update(release_at=0, reveal_at=0) for d in data["demands"]],
    )
    edit(path, lambda data: data.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(
        ConfigurationError, match="CP-SAT has no grid production adapter"
    ):
        resolve_run(path)
    assert not (bundle / "runs").exists()


def test_explicit_wait_action_and_script_failure_evidence(bundle):
    path = run_path(bundle, "online_arrivals_event")
    algorithm = bundle / "configs/algorithms/spt.yaml"
    data = {
        "schema": "smartsom.algorithm/v1",
        "algorithm": {
            "provider": "builtin.scripted",
            "parameters": {"commands": [{}, {}]},
        },
    }
    algorithm.write_text(json.dumps(data))
    prepared = resolve_run(path)
    assert len(prepared.resolved.algorithm.commands) == 2
    with pytest.raises(RunFailedError, match="exhausted") as error:
        run_one(prepared, verbose=False)
    directory = error.value.run_dir
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "failed" and manifest["last_tick"] == 2
    assert [r["tick"] for r in json_lines(directory, "trace.jsonl")] == [1, 2]
    assert manifest["inputs"]["scenario"] == primitive(prepared.resolved.scenario)
    for command in (
        {"unknown": 1},
        {"machines": [["M1", {"job_id": "A"}]]},
        {"agvs": [["agv", "WAIT"], ["agv", "WAIT"]]},
    ):
        data["algorithm"]["parameters"]["commands"] = [command]
        with pytest.raises(ValueError):
            AlgorithmFile.model_validate_json(json.dumps(data))


def test_cli_and_hash_seed_cwd_determinism(bundle, monkeypatch):
    path = run_path(bundle, "generated_arrivals_event")
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    outputs = []
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import canonical_json; from smartsom.algorithms.production import GreedyProductionPolicy; from smartsom.engine.production import ProductionSimulator; import sys; r=resolve_run(sys.argv[1]); c=r.resolved.scenario; s=ProductionSimulator(c); p=GreedyProductionPolicy(c.factory,rule='spt');\nwhile not s.done: s.step(p.act(s.decision(p.rank(s.decision()))))\nprint(canonical_json([c.demands,s.snapshot()]))"
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
    scenario = Path(json.loads(original.config_json)["scenario"])
    edit(
        scenario,
        lambda data: data.update(
            mode="dynamic",
            arrivals={"initial_jobs": 0, "release_min": 1, "release_max": 5},
        ),
    )
    dynamic = resolve_run(path)
    assert [d.steps for d in dynamic.resolved.scenario.demands] == [
        d.steps for d in original.resolved.scenario.demands
    ]
    assert run_one(dynamic, verbose=False).simulation_result.final_state["tick"] > 0


def test_failure_on_empty_observation_preserves_inputs_and_observed_prefix(
    bundle, monkeypatch
):
    from smartsom.experiments import providers

    class BrokenPolicy(GreedyProductionPolicy):
        def act(self, view):
            if view["tick"] == 1:
                raise RuntimeError("failure on empty view")
            return super().act(view)

    monkeypatch.setattr(providers, "GreedyProductionPolicy", BrokenPolicy)
    resolved = resolve_run(run_path(bundle, "online_arrivals_event"))
    with pytest.raises(RuntimeError, match="failure on empty view") as error:
        run_one(resolved, verbose=False)
    directory = error.value.run_dir
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "failed" and manifest["last_tick"] == 1
    assert len(json_lines(directory, "trace.jsonl")) == 1
    assert manifest["inputs"]["scenario"] == primitive(resolved.resolved.scenario)
    assert audit(directory)["ticks"] == 1


def test_fixed_table_provenance_conflicts_and_reordering(bundle):
    resolved = resolve_run(run_path(bundle, "generated_arrivals_event"))
    plan = ArrivalPlan(
        tuple(
            JobArrival(d.demand_id, d.release_at, d.reveal_at)
            for d in resolved.resolved.scenario.demands
        )
    )
    rows = arrival_rows(plan, None)
    table = bundle / "timing.jsonl"
    table.write_text("\n".join(canonical_json(row) for row in reversed(rows)))
    plan, provenance, raw_sha = read_arrivals(table)
    assert plan.jobs == tuple(
        JobArrival(d.demand_id, d.release_at, d.reveal_at)
        for d in resolved.resolved.scenario.demands
    )
    assert provenance is None
    table.write_text("\n".join(canonical_json(row) for row in rows))
    assert read_arrivals(table)[0] == plan
    assert read_arrivals(table)[2] != raw_sha
    rows = [primitive(row) for row in rows]
    rows[1]["provenance"] = {"invalid": True}
    table.write_text("\n".join(canonical_json(row) for row in rows))
    with pytest.raises(ConfigurationError, match="provenance"):
        read_arrivals(table)
