import hashlib
from dataclasses import FrozenInstanceError

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import ExecutionScheduleFile
from smartsom.engine import Simulator, replay_schedule
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main


def test_configured_hand_matches_code_and_reusable_timetable(bundle, monkeypatch):
    config = run_path(bundle, "transport_hand")
    resolved = resolve_run(config)
    assert resolved.transport_enabled
    assert not (bundle / "runs").exists()
    assert main(["validate", str(config)]) == 0
    assert not (bundle / "runs").exists()
    views = []

    class Observe:
        def select_action(self, context):
            views.append(primitive(context))
            return SPTPolicy().select_action(context)

    direct = Simulator(resolved.factory, resolved.workload, transport_enabled=True).run(
        Observe()
    )
    result = run_one(resolved)
    assert result.simulation_result == direct
    assert direct.makespan == 15
    assert json_lines(result.run_dir, "observations.jsonl") == views
    assert json_lines(result.run_dir, "trace.jsonl") == primitive(direct.trace)
    saved, _ = read_model(
        result.run_dir / "execution_schedule.json", ExecutionScheduleFile
    )
    assert (
        replay_schedule(
            resolved.factory,
            resolved.workload,
            saved.execution_schedule,
            transport_enabled=True,
        )
        == direct
    )
    manifest = json_file(result.run_dir, "manifest.json")
    assert manifest["transport_sha256"] == digest(resolved.factory.transport)
    assert (
        manifest["artifacts"]["execution_schedule.json"]
        == hashlib.sha256(
            (result.run_dir / "execution_schedule.json").read_bytes()
        ).hexdigest()
    )
    assert "delivered_jobs=1" in (result.run_dir / "progress.log").read_text()
    assert json_file(result.run_dir, "summary.json")["processing_completion_time"] == 14
    with pytest.raises(FrozenInstanceError):
        resolved.transport_enabled = False
    # The shared runner accepts the very same replay policy and verifies its result.
    monkeypatch.setattr(
        "smartsom.experiments.runner.build_provider",
        lambda algorithm: ScheduleReplayPolicy(
            resolved.factory,
            resolved.workload,
            saved.execution_schedule,
            transport_enabled=True,
        ),
    )
    assert run_one(resolved).simulation_result == direct


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["factory"]["transport"]["travel_times"][0].update(ticks=True),
        lambda d: d["factory"]["transport"]["travel_times"][0].update(ticks=1.5),
        lambda d: d["factory"]["transport"]["travel_times"].pop(),
        lambda d: d["factory"]["transport"]["travel_times"].append(
            d["factory"]["transport"]["travel_times"][0]
        ),
        lambda d: d["factory"]["transport"]["agvs"][0].update(initial_node_id="bad"),
        lambda d: d["factory"]["transport"]["agvs"][0].update(capacity=2),
        lambda d: d["factory"]["transport"].update(seed=42),
        lambda d: d["factory"].pop("transport"),
    ],
)
def test_bad_transport_inputs_fail_before_simulator_or_directory(
    bundle, monkeypatch, mutation
):
    edit(bundle / "configs/factories/transport_hand.yaml", mutation)
    monkeypatch.setattr(
        "smartsom.experiments.runner.Simulator",
        lambda *a, **k: pytest.fail("constructed Simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "transport_hand"))
    assert not (bundle / "runs").exists()


def test_toggle_preserves_seed_world_and_cp_rejects_even_zero_matrix(bundle):
    config = run_path(bundle, "transport_hand")
    on = resolve_run(config)
    scenario = bundle / "configs/scenarios/transport_hand.yaml"
    edit(scenario, lambda d: d.update(transport=None))
    off = resolve_run(config)
    assert on.seeds == off.seeds
    assert on.factory == off.factory and on.workload == off.workload
    assert on.workload_sha256 == off.workload_sha256
    assert not off.transport_enabled and off.transport_sha256 is None
    edit(
        scenario,
        lambda d: d.update(
            transport={"kind": "fixed_matrix"}, visibility="full_static"
        ),
    )
    edit(config, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    edit(
        bundle / "configs/factories/transport_hand.yaml",
        lambda d: [
            t.update(ticks=0) for t in d["factory"]["transport"]["travel_times"]
        ],
    )
    with pytest.raises(ConfigurationError, match="transport"):
        resolve_run(config)


def test_provider_failure_and_write_failure_keep_transport_evidence(
    bundle, monkeypatch
):
    from smartsom.experiments import evidence

    config = run_path(bundle, "transport_hand")
    good = resolve_run(config)
    direct = Simulator(good.factory, good.workload, transport_enabled=True).run(
        SPTPolicy()
    )
    algorithm = bundle / "configs/algorithms/spt_transport.yaml"
    edit(
        algorithm,
        lambda d: d.update(
            algorithm={
                "provider": "builtin.scripted",
                "parameters": {"actions": primitive(direct.actions[:-1])},
            }
        ),
    )
    with pytest.raises(RunFailedError) as err:
        run_one(resolve_run(config))
    failed = err.value.run_dir
    assert json_file(failed, "summary.json")["makespan"] is None
    assert any(r["kind"] == "delivery" for r in json_lines(failed, "trace.jsonl"))
    original = evidence.write_json

    def fail_schedule(path, value):
        if path.name == "execution_schedule.json":
            raise OSError("injected schedule write failure")
        return original(path, value)

    monkeypatch.setattr(evidence, "write_json", fail_schedule)
    with pytest.raises(RunFailedError) as err:
        run_one(good)
    assert json_file(err.value.run_dir, "summary.json")["makespan"] is None
    assert json_lines(err.value.run_dir, "trace.jsonl")[-1]["kind"] == "terminate"


def test_script_transport_and_cli_failure_codes(bundle):
    config = run_path(bundle, "transport_hand")
    good = resolve_run(config)
    direct = Simulator(good.factory, good.workload, transport_enabled=True).run(
        SPTPolicy()
    )
    algorithm = bundle / "configs/algorithms/spt_transport.yaml"
    edit(
        algorithm,
        lambda d: d.update(
            algorithm={
                "provider": "builtin.scripted",
                "parameters": {"actions": primitive(direct.actions)},
            }
        ),
    )
    assert run_one(resolve_run(config)).simulation_result == direct
    edit(algorithm, lambda d: d["algorithm"]["parameters"]["actions"].pop())
    assert main(["run", str(config)]) != 0
    edit(
        algorithm,
        lambda d: d["algorithm"]["parameters"]["actions"][0].update(agv_id="unknown"),
    )
    assert main(["validate", str(config)]) != 0


def test_transport_hashseed_cwd_and_input_permutation(bundle, tmp_path):
    import os
    import subprocess
    import sys

    source = run_path(bundle, "transport_combined")
    code = """from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.engine import Simulator
from smartsom.algorithms import SPTPolicy
import sys
r=resolve_run(sys.argv[1])
print(canonical_json((r.factory, r.workload, r.seeds)))
views=[]
class Observe:
 def select_action(self,c):
  views.append(c)
  return SPTPolicy().select_action(c)
result=Simulator(r.factory,r.workload,transport_enabled=True,arrivals=r.arrivals,processing_times=r.processing_times,machine_events=r.machine_events,decision_trigger=r.scenario.decision_trigger).run(Observe())
print(canonical_json((result,views)))
"""
    before = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=bundle,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )

    def reverse_factory(data):
        data["factory"]["machines"].reverse()
        for key in ("nodes", "machine_locations", "agvs", "travel_times"):
            data["factory"]["transport"][key].reverse()

    edit(bundle / "configs/factories/transport_multiple.yaml", reverse_factory)

    def reverse_workload(data):
        for order in data["workload"]["orders"]:
            order["jobs"].reverse()
            for job in order["jobs"]:
                job["operations"].reverse()
                for op in job["operations"]:
                    op["modes"].reverse()

    edit(bundle / "data/reference/transport/combined_workload.json", reverse_workload)
    after = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONHASHSEED": "987"},
    )
    assert before == after


def test_off_retains_all_twelve_pre_item8_golden_cases(bundle):
    import json

    from smartsom.experiments.providers import build_provider

    frozen = json.loads(
        (bundle / "data/reference/transport/disabled_golden.json").read_text()
    )
    for name, expected in frozen["cases"].items():
        # Historical golden IDs stay immutable; only the fixture path moved.
        name = "run_fixed_trace" if name == "competition" else name
        r = resolve_run(run_path(bundle, name))
        result = Simulator(
            r.factory,
            r.workload,
            arrivals=r.arrivals,
            processing_times=r.processing_times,
            machine_events=r.machine_events,
            decision_trigger=r.scenario.decision_trigger,
        ).run(build_provider(r.algorithm))
        assert {
            "factory": r.factory_sha256,
            "workload": r.workload_sha256,
            "seeds": digest(r.seeds),
            "schedule": digest(result.schedule),
            "trace": digest(result.trace),
            "actions": digest(result.actions),
            "makespan": result.makespan,
        } == expected


def test_full_replay_verification_failure_preserves_evidence(bundle, monkeypatch):
    from smartsom.engine import ReplayError

    r = resolve_run(run_path(bundle, "transport_hand"))
    reference, _ = read_model(
        bundle / "data/reference/transport/hand_schedule.json", ExecutionScheduleFile
    )
    policy = ScheduleReplayPolicy(
        r.factory, r.workload, reference.execution_schedule, transport_enabled=True
    )

    def fail(result):
        raise ReplayError("injected transport result mismatch")

    monkeypatch.setattr(policy, "verify_result", fail)
    monkeypatch.setattr(
        "smartsom.experiments.runner.build_provider", lambda algorithm: policy
    )
    with pytest.raises(RunFailedError) as error:
        run_one(r)
    directory = error.value.run_dir
    assert (
        json_file(directory, "failure.json")["message"]
        == "injected transport result mismatch"
    )
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_lines(directory, "trace.jsonl")[-1]["kind"] == "terminate"
    assert not (directory / "execution_schedule.json").exists()
