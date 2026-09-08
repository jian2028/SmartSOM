import os
import subprocess
import sys

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, run_path
from test_studies import study_file

from smartsom.config import (
    ConfigurationError,
    load_resolved_run,
    resolve_run,
    resolve_study,
)
from smartsom.engine import replay_schedule
from smartsom.experiments import RunFailedError, run_one


def test_configuration_snapshot_and_exact_replay(bundle):
    r = resolve_run(run_path(bundle, "holding_hand"))
    result = run_one(r)
    assert result.simulation_result.makespan == 11
    assert load_resolved_run(result.run_dir / "resolved_run.yaml") == r
    m = json_file(result.run_dir, "manifest.json")
    assert m["holding_buffer"]["capacity"] == 1
    assert m["holding_buffer_sha256"] == r.holding_buffer_sha256
    assert (
        json_file(result.run_dir, "execution_schedule.json")["schema"]
        == "smartsom.execution-schedule/v2"
    )
    assert (
        replay_schedule(
            r.factory,
            r.workload,
            result.simulation_result.execution_schedule,
            transport_enabled=True,
            holding_buffer_enabled=True,
        ).makespan
        == 11
    )
    assert (
        run_one(
            load_resolved_run(result.run_dir / "resolved_run.yaml")
        ).simulation_result
        == result.simulation_result
    )


@pytest.mark.parametrize("capacity", [True, -1, 1.5])
def test_bad_capacity_rejected_before_allocation(bundle, capacity):
    edit(
        bundle / "configs/factories/holding.yaml",
        lambda d: d["factory"]["holding_buffer"].update(capacity=capacity),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "holding_hand"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(transport=None),
        lambda d: d["holding_buffer"].update(unknown=1),
        lambda d: d.update(holding_buffer={"kind": "unknown"}),
    ],
)
def test_invalid_enablement(bundle, change):
    edit(bundle / "configs/scenarios/holding_hand.yaml", change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "holding_hand"))
    assert not (bundle / "runs").exists()


def test_cp_rejected_and_holding_ablation_preserves_inputs(bundle):
    path = run_path(bundle, "holding_hand")
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="holding"):
        resolve_run(path)
    study = study_file(bundle, scenario="holding_hand")
    edit(
        study,
        lambda d: d.update(
            variants=[{"id": "on"}, {"id": "off", "disable": ["holding_buffer"]}]
        ),
    )
    rows = resolve_study(study).entries
    assert len({e.resolved.workload_sha256 for e in rows}) == 1
    assert (
        len(
            {
                tuple(
                    s.value
                    for s in e.resolved.seeds
                    if s.domain not in ("algorithm", "solver")
                )
                for e in rows
            }
        )
        == 1
    )
    assert {e.resolved.holding_buffer_enabled for e in rows} == {False, True}


def test_holding_writer_failure_retains_cause_and_partial_trace(bundle, monkeypatch):
    import smartsom.experiments.evidence as evidence

    original = evidence.append_json

    def broken(stream, value):
        if getattr(value, "kind", None) == "holding_reserve":
            raise OSError("holding trace write failed")
        original(stream, value)

    monkeypatch.setattr(evidence, "append_json", broken)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle, "holding_hand")))
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None
    assert (
        json_file(error.value.run_dir, "failure.json")["message"]
        == "holding trace write failed"
    )
    assert (error.value.run_dir / "trace.jsonl").stat().st_size > 0


def test_reordered_and_other_cwd_hash_seed_inputs_match(bundle):
    path = run_path(bundle, "holding_hand")
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import digest; from smartsom.experiments import run_one; import sys; print(digest(run_one(resolve_run(sys.argv[1])).simulation_result))"
    values = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for cwd, seed in ((bundle, "1"), (bundle.parent, "123"))
    ]
    assert values[0] == values[1]
