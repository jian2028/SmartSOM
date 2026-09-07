import hashlib
from dataclasses import FrozenInstanceError

import pytest
from test_buffers import direct_case, post_case, vehicle_case
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import ExecutionScheduleFile
from smartsom.engine import ReplayError, replay, replay_schedule
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main


@pytest.mark.parametrize(
    "name,build,agv,makespan",
    [
        ("buffers_direct_zero", direct_case, False, 6),
        ("buffers_vehicle_zero", vehicle_case, True, 8),
        ("buffers_post_one", post_case, False, 6),
    ],
)
def test_configured_hand_and_exported_v2(
    bundle, name, build, agv, makespan, monkeypatch
):
    path = run_path(bundle, name)
    resolved = resolve_run(path)
    f, w, actions = build()
    assert resolved.factory == f and resolved.workload == w and resolved.buffers_enabled
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    direct = replay(f, w, actions, buffers_enabled=True, transport_enabled=agv)
    result = run_one(resolved)
    assert result.simulation_result == direct and direct.makespan == makespan
    assert json_lines(result.run_dir, "trace.jsonl") == primitive(direct.trace)
    saved, _ = read_model(
        result.run_dir / "execution_schedule.json", ExecutionScheduleFile
    )
    assert saved.schema_id == "smartsom.execution-schedule/v2"
    assert saved.execution_schedule == direct.execution_schedule
    assert (
        replay_schedule(
            f, w, saved.execution_schedule, buffers_enabled=True, transport_enabled=agv
        ).execution_schedule
        == direct.execution_schedule
    )
    assert json_file(result.run_dir, "summary.json")["delivered_jobs"] == len(
        w.orders[0].jobs
    )
    assert "delivered_jobs=" in (result.run_dir / "progress.log").read_text()
    assert json_lines(result.run_dir, "observations.jsonl")[0]["buffers"]
    manifest = json_file(result.run_dir, "manifest.json")
    assert manifest["buffers_sha256"] == digest(f.buffers)
    assert (
        manifest["artifacts"]["execution_schedule.json"]
        == hashlib.sha256(
            (result.run_dir / "execution_schedule.json").read_bytes()
        ).hexdigest()
    )
    monkeypatch.setattr(
        "smartsom.experiments.runner.build_provider",
        lambda _: ScheduleReplayPolicy(
            f, w, saved.execution_schedule, buffers_enabled=True, transport_enabled=agv
        ),
    )
    assert (
        run_one(resolved).simulation_result.execution_schedule
        == direct.execution_schedule
    )
    with pytest.raises(FrozenInstanceError):
        resolved.buffers_enabled = False
    with pytest.raises(FrozenInstanceError):
        resolved.factory.buffers[0].pre_capacity = 10


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["factory"]["buffers"][0].update(pre_capacity=True),
        lambda d: d["factory"]["buffers"][0].update(post_capacity=1.5),
        lambda d: d["factory"]["buffers"][0].update(pre_capacity=-1),
        lambda d: d["factory"]["buffers"][0].update(post_capacity="1"),
        lambda d: d["factory"]["buffers"][0].update(machine_id="unknown"),
        lambda d: d["factory"]["buffers"][0].update(shared_capacity=2),
        lambda d: d["factory"]["buffers"].append(d["factory"]["buffers"][0]),
    ],
)
def test_bad_capacities_fail_before_execution(bundle, mutation, monkeypatch):
    edit(bundle / "configs/factories/buffers_direct_zero.yaml", mutation)
    monkeypatch.setattr(
        "smartsom.experiments.runner.Simulator",
        lambda *a, **k: pytest.fail("constructed simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "buffers_direct_zero"))
    assert not (bundle / "runs").exists()


def test_duplicate_keys_unknown_module_fields_and_cp_rejection(bundle):
    path = run_path(bundle, "buffers_direct_zero")
    factory = bundle / "configs/factories/buffers_direct_zero.yaml"
    old = factory.read_text()
    factory.write_text(
        old.replace("pre_capacity: 0", "pre_capacity: 0\n    pre_capacity: 1", 1)
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        resolve_run(path)
    factory.write_text(old)
    scenario = bundle / "configs/scenarios/buffers_direct_zero.yaml"
    edit(scenario, lambda d: d["buffers"].update(seed=10))
    with pytest.raises(ConfigurationError):
        resolve_run(path)
    edit(
        scenario,
        lambda d: d.update(buffers={"kind": "limited"}, visibility="full_static"),
    )
    edit(factory, lambda d: d["factory"].update(buffers=[]))
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="buffers"):
        resolve_run(path)
    assert not (bundle / "runs").exists()


def test_toggle_seed_identity_and_explicit_transfer_compatibility(bundle):
    path = run_path(bundle, "buffers_direct_zero")
    on = resolve_run(path)
    edit(
        bundle / "configs/scenarios/buffers_direct_zero.yaml",
        lambda d: d.update(buffers=None),
    )
    with pytest.raises(ConfigurationError, match="transfer"):
        resolve_run(path)
    edit(path, lambda d: d.update(algorithm="../algorithms/spt_buffers.yaml"))
    off = resolve_run(path)
    assert (
        on.seeds == off.seeds
        and on.factory == off.factory
        and on.workload_sha256 == off.workload_sha256
    )
    assert not off.buffers_enabled and off.buffers_sha256 is None


@pytest.mark.parametrize("failure", ["script", "provider", "replay", "write"])
def test_failures_preserve_buffer_records_and_never_success_makespan(
    bundle, monkeypatch, failure
):
    from smartsom.algorithms import ScriptedPolicy
    from smartsom.experiments import evidence

    resolved = resolve_run(run_path(bundle, "buffers_vehicle_zero"))
    actions = resolved.algorithm.algorithm.parameters.actions
    if failure == "script":
        monkeypatch.setattr(
            "smartsom.experiments.runner.build_provider",
            lambda _: ScriptedPolicy(actions[:-1]),
        )
    elif failure == "provider":

        class Fail:
            def __init__(self):
                self.n = 0

            def select_action(self, c):
                if self.n == 3:
                    raise RuntimeError("provider after buffer wait")
                self.n += 1
                return actions[self.n - 1]

        monkeypatch.setattr(
            "smartsom.experiments.runner.build_provider", lambda _: Fail()
        )
    elif failure == "replay":
        result = replay(
            resolved.factory,
            resolved.workload,
            actions,
            buffers_enabled=True,
            transport_enabled=True,
        )
        policy = ScheduleReplayPolicy(
            resolved.factory,
            resolved.workload,
            result.execution_schedule,
            buffers_enabled=True,
            transport_enabled=True,
        )

        def fail(_):
            raise ReplayError("buffer schedule mismatch")

        monkeypatch.setattr(policy, "verify_result", fail)
        monkeypatch.setattr(
            "smartsom.experiments.runner.build_provider", lambda _: policy
        )
    else:
        original = evidence.write_json

        def fail(path, value):
            if path.name == "execution_schedule.json":
                raise OSError("buffer schedule writer failed")
            original(path, value)

        monkeypatch.setattr(evidence, "write_json", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolved)
    directory = error.value.run_dir
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert json_file(directory, "failure.json")["message"]
    assert any(
        x["kind"] == "wait_for_unload" for x in json_lines(directory, "trace.jsonl")
    )
    assert json_lines(directory, "observations.jsonl")


def test_deadlock_evidence_and_cli_failure(bundle):
    path = run_path(bundle, "buffers_vehicle_zero")
    edit(
        bundle / "configs/factories/buffers_vehicle_zero.yaml",
        lambda d: (
            d["factory"]["transport"].update(
                agvs=d["factory"]["transport"]["agvs"][:1]
            ),
            d["factory"]["buffers"][0].update(post_capacity=0),
        ),
    )
    edit(
        bundle / "configs/algorithms/buffers_vehicle_zero.yaml",
        lambda d: d["algorithm"]["parameters"].update(
            actions=[
                {
                    "agv_id": "V1",
                    "job_id": "A",
                    "destination": {"kind": "machine", "machine_id": "M1"},
                },
                {"operation_id": "A", "processing_mode_id": "standard"},
                {
                    "agv_id": "V1",
                    "job_id": "B",
                    "destination": {"kind": "machine", "machine_id": "M1"},
                },
            ]
        ),
    )
    assert main(["run", str(path)]) != 0
    directory = next((bundle / "runs").iterdir())
    assert json_file(directory, "failure.json")["exception_type"].endswith(
        "DeadlockError"
    )
    assert "waiting" in json_file(directory, "failure.json")["message"]
    assert json_file(directory, "summary.json")["makespan"] is None


@pytest.mark.parametrize("agv", [False, True])
def test_buffer_hashseed_cwd_and_input_permutation(bundle, tmp_path, agv):
    import os
    import subprocess
    import sys

    source = run_path(bundle, "buffers_combined")
    if not agv:
        edit(
            bundle / "configs/scenarios/buffers_combined.yaml",
            lambda d: d.update(transport=None),
        )
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
result=Simulator(r.factory,r.workload,buffers_enabled=True,transport_enabled=r.transport_enabled,arrivals=r.arrivals,processing_times=r.processing_times,machine_events=r.machine_events,decision_trigger=r.scenario.decision_trigger).run(Observe())
print(canonical_json((result,views)))
"""
    before = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=bundle,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )

    def reverse_factory(data):
        data["factory"]["machines"].reverse()
        data["factory"]["buffers"].reverse()
        for key in ("nodes", "machine_locations", "agvs", "travel_times"):
            data["factory"]["transport"][key].reverse()

    edit(bundle / "configs/factories/buffers_combined.yaml", reverse_factory)

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


def test_pre_item9_committed_main_goldens(bundle):
    import json

    from smartsom.engine import Simulator
    from smartsom.experiments.providers import build_provider

    frozen = json.loads(
        (bundle / "data/reference/buffers/disabled_golden.json").read_text()
    )
    for name, expected in frozen["cases"].items():
        r = resolve_run(run_path(bundle, name))
        kwargs = dict(
            arrivals=r.arrivals,
            processing_times=r.processing_times,
            machine_events=r.machine_events,
            decision_trigger=r.scenario.decision_trigger,
            transport_enabled=r.transport_enabled,
        )
        actual = Simulator(r.factory, r.workload, **kwargs).run(
            build_provider(r.algorithm)
        )
        observed = {
            field: digest(getattr(actual, field))
            for field in (
                "schedule",
                "actions",
                "trace",
                "makespan",
                "execution_schedule",
            )
        }
        observed.update(
            factory=digest(r.factory),
            workload=digest(r.workload),
            seeds=digest(r.seeds),
        )
        assert observed == expected
        if r.transport_enabled:
            assert (
                Simulator(r.factory, r.workload, buffers_enabled=True, **kwargs).run(
                    build_provider(r.algorithm)
                )
                == actual
            )
