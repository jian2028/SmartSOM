import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main

NAMES = (
    "machine_events_fixed",
    "machine_events_generated",
    "machine_events_arrivals_dispatch",
    "machine_events_arrivals_event",
)


@pytest.mark.parametrize("name", NAMES)
def test_config_code_replay_and_evidence(bundle, name):
    resolved = resolve_run(run_path(bundle, name))
    assert not (bundle / "runs").exists()
    kwargs = dict(
        machine_events=resolved.machine_events,
        arrivals=resolved.arrivals,
        processing_times=resolved.processing_times,
        decision_trigger=resolved.scenario.decision_trigger,
    )
    views = []

    class Observe:
        def select_action(self, context):
            views.append(primitive(context))
            return SPTPolicy().select_action(context)

    direct = Simulator(resolved.factory, resolved.workload, **kwargs).run(Observe())
    actual = run_one(resolved)
    assert (
        actual.simulation_result
        == direct
        == replay(resolved.factory, resolved.workload, direct.actions, **kwargs)
    )
    assert (
        replay_schedule(
            resolved.factory, resolved.workload, direct.schedule, **kwargs
        ).schedule
        == direct.schedule
    )
    assert json_lines(actual.run_dir, "observations.jsonl") == views
    assert json_lines(actual.run_dir, "trace.jsonl") == primitive(direct.trace)
    manifest = json_file(actual.run_dir, "manifest.json")
    assert manifest["machine_events_sha256"] == digest(resolved.machine_events)
    realized = json_file(actual.run_dir, "realized_machine_events.json")
    assert realized["machine_events"] == primitive(resolved.machine_events)
    assert (
        manifest["artifacts"]["realized_machine_events.json"]
        == hashlib.sha256(
            (actual.run_dir / "realized_machine_events.json").read_bytes()
        ).hexdigest()
    )
    if resolved.arrivals:
        assert (actual.run_dir / "realized_events.jsonl").exists()
        assert (actual.run_dir / "realized_processing_times.json").exists()
    with pytest.raises(FrozenInstanceError):
        resolved.machine_events = None


def test_export_reimport_seed_and_algorithm_changes_do_not_regenerate(
    bundle, monkeypatch
):
    name = "machine_events_generated"
    original = resolve_run(run_path(bundle, name))
    run = run_one(original)
    exported = run.run_dir / "realized_machine_events.json"
    edit(
        bundle / f"configs/scenarios/{name}.yaml",
        lambda d: d.update(machine_events={"kind": "fixed", "path": str(exported)}),
    )
    edit(
        run_path(bundle, name),
        lambda d: d.update(seed=999, algorithm="../algorithms/first_feasible.yaml"),
    )
    monkeypatch.setattr(
        "smartsom.config.materialization.generate_machine_events",
        lambda *args: pytest.fail("import resampled"),
    )
    loaded = resolve_run(run_path(bundle, name))
    assert loaded.machine_events == original.machine_events
    assert loaded.machine_event_provenance == original.machine_event_provenance
    assert not next(s.consumed for s in loaded.seeds if s.domain == "machine_events")
    assert original.machine_events_sha256 == loaded.machine_events_sha256
    source = next(s for s in loaded.sources if s.role == "machine_events")
    assert source.sha256 == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert run_one(loaded).simulation_result.makespan > 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=True),
        lambda d: d["machine_events"]["outages"][0].update(start_time=True),
        lambda d: d["machine_events"]["outages"][0].update(end_time=4.0),
        lambda d: d["machine_events"]["outages"][0].update(machine_id="unknown"),
        lambda d: d["machine_events"]["outages"][0].update(start_time=5),
        lambda d: d.update(content_sha256="0" * 64),
        lambda d: d.update(seed=1),
    ],
)
def test_invalid_fixed_input_before_simulator_and_directory(
    bundle, monkeypatch, mutation
):
    edit(bundle / "data/machine_events/hand.json", mutation)
    monkeypatch.setattr(
        "smartsom.experiments.runner.Simulator",
        lambda *a, **k: pytest.fail("constructed simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "machine_events_fixed"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(generation_until_tick=True),
        lambda d: d.update(generation_until_tick=0),
        lambda d: d.update(generation_until_tick=2.0),
        lambda d: d.update(machines=[]),
        lambda d: d["machines"].append(d["machines"][0].copy()),
        lambda d: d["machines"][0].update(mean_uptime_ticks=True),
        lambda d: d["machines"][0].update(mean_uptime_ticks=0),
        lambda d: d["machines"][0].update(mean_uptime_ticks="NaN"),
        lambda d: d["machines"][0].update(repair_ticks={"min": 0, "max": 2}),
        lambda d: d["machines"][0].update(repair_ticks={"min": 3, "max": 2}),
        lambda d: d["machines"][0].update(repair_ticks={"min": 1, "max": True}),
        lambda d: d["machines"][0].update(machine_id="unknown"),
        lambda d: d.update(seed=1),
    ],
)
def test_invalid_profiles(bundle, mutation):
    edit(
        bundle / "configs/scenarios/machine_events_generated.yaml",
        lambda d: mutation(d["machine_events"]["profile"]),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "machine_events_generated"))
    assert not (bundle / "runs").exists()


def test_duplicate_keys_and_cli_validation(bundle):
    config = run_path(bundle, "machine_events_fixed")
    assert main(["validate", str(config)]) == 0 and not (bundle / "runs").exists()
    path = bundle / "data/machine_events/hand.json"
    path.write_text(
        '{"schema":"smartsom.machine-events/v1","machine_events":{"outages":[]},"machine_events":{"outages":[]}}'
    )
    assert main(["validate", str(config)]) != 0
    assert main(["run", str(config)]) != 0 and not (bundle / "runs").exists()


def test_empty_enabled_plan_rejects_cp_and_full_static(bundle):
    (bundle / "data/machine_events/hand.json").write_text(
        '{"schema":"smartsom.machine-events/v1","machine_events":{"outages":[]}}'
    )
    name = "machine_events_fixed"
    edit(
        run_path(bundle, name),
        lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"),
    )
    with pytest.raises(ConfigurationError, match="machine events"):
        resolve_run(run_path(bundle, name))
    edit(
        bundle / f"configs/scenarios/{name}.yaml",
        lambda d: d.update(visibility="full_static"),
    )
    with pytest.raises(ConfigurationError, match="decision_context"):
        resolve_run(run_path(bundle, name))


def test_failure_after_pause_retains_actual_records_without_makespan(
    bundle, monkeypatch
):
    class Fail:
        calls = 0

        def select_action(self, context):
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("policy failed after events")
            return SPTPolicy().select_action(context)

    monkeypatch.setattr("smartsom.experiments.runner.build_provider", lambda _: Fail())
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle, "machine_events_fixed")))
    directory = error.value.run_dir
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert (directory / "realized_machine_events.json").exists()
    assert "pause" in [row["kind"] for row in json_lines(directory, "trace.jsonl")]
    assert json_file(directory, "summary.json")["makespan"] is None
    assert all(
        row.get("makespan") is None for row in json_lines(directory, "metrics.jsonl")
    )


def test_resolved_snapshot_survives_input_deletion_and_write_failure(
    bundle, monkeypatch
):
    resolved = resolve_run(run_path(bundle, "machine_events_fixed"))
    (bundle / "data/machine_events/hand.json").unlink()
    assert run_one(resolved).simulation_result.makespan > 0
    from smartsom.experiments import evidence

    original = evidence.write_json

    def fail(path, value):
        if path.name == "realized_machine_events.json":
            raise OSError("outage write failed")
        original(path, value)

    monkeypatch.setattr(evidence, "write_json", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolved)
    assert json_file(error.value.run_dir, "manifest.json")["status"] == "failed"
    assert "outage write failed" in (error.value.run_dir / "failure.json").read_text()


def test_scripted_provider_with_outages_and_insufficient_actions(bundle):
    path = run_path(bundle, "machine_events_fixed")
    edit(path, lambda d: d.update(algorithm="../algorithms/competition_script.yaml"))
    assert run_one(resolve_run(path)).simulation_result.makespan == 22
    edit(
        bundle / "configs/algorithms/competition_script.yaml",
        lambda d: d["algorithm"]["parameters"]["actions"].pop(),
    )
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(path))
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None
    assert any(
        row["kind"] == "resume"
        for row in json_lines(error.value.run_dir, "trace.jsonl")
    )


def test_seed_independence_and_hashseed_cwd(bundle, tmp_path):
    path = run_path(bundle, "machine_events_generated")
    resolved = resolve_run(path)
    assert next(s.consumed for s in resolved.seeds if s.domain == "machine_events")
    edit(
        bundle / "configs/scenarios/machine_events_generated.yaml",
        lambda d: d.pop("machine_events"),
    )
    off = resolve_run(path)
    assert replace(resolved, machine_events=None).workload == off.workload
    assert [(s.domain, s.value) for s in resolved.seeds] == [
        (s.domain, s.value) for s in off.seeds
    ]
    edit(
        bundle / "configs/scenarios/machine_events_generated.yaml",
        lambda d: d.update(machine_events=primitive(resolved.scenario.machine_events)),
    )
    # Exercise generated input, result and trace across hash seeds and cwd.
    source = path
    code = """from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.engine import Simulator
from smartsom.algorithms import SPTPolicy
import sys
r=resolve_run(sys.argv[1])
print(canonical_json(r.machine_events))
print(canonical_json(Simulator(r.factory,r.workload,arrivals=r.arrivals,processing_times=r.processing_times,machine_events=r.machine_events,decision_trigger=r.scenario.decision_trigger).run(SPTPolicy())))"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(source)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "87")
    ]
    assert outputs[0] == outputs[1]
