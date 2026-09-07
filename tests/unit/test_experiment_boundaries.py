import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.config import resolve_run
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.evidence import artifact_digests
from smartsom.experiments.providers import build_provider


def test_module_switches_and_algorithm_binding_preserve_other_materialized_inputs(
    bundle,
):
    scenario = bundle / "configs/scenarios/generated.yaml"
    edit(
        scenario,
        lambda value: value.update(
            arrivals={
                "kind": "uniform_release_v1",
                "profile": {
                    "initial_job_count": 1,
                    "release_window": {"min": 2, "max": 5},
                },
            },
            processing_time={"kind": "uniform_multiplier"},
        ),
    )
    path = run_path(bundle, "generated")
    both = resolve_run(path)
    edit(scenario, lambda value: value.pop("arrivals"))
    without_arrivals = resolve_run(path)
    assert without_arrivals.workload == both.workload
    assert without_arrivals.processing_times == both.processing_times
    assert without_arrivals.processing_provenance == both.processing_provenance
    assert [s.value for s in without_arrivals.seeds] == [s.value for s in both.seeds]
    edit(path, lambda value: value.update(algorithm="../algorithms/spt.yaml"))
    other_algorithm = resolve_run(path)
    assert other_algorithm.workload == without_arrivals.workload
    assert other_algorithm.processing_times == without_arrivals.processing_times
    assert other_algorithm.seeds == without_arrivals.seeds
    edit(
        scenario,
        lambda value: value.update(
            arrivals={
                "kind": "uniform_release_v1",
                "profile": {
                    "initial_job_count": 1,
                    "release_window": {"min": 2, "max": 5},
                },
            }
        ),
    )
    edit(scenario, lambda value: value.pop("processing_time"))
    without_processing = resolve_run(path)
    assert without_processing.workload == both.workload
    assert without_processing.arrivals == both.arrivals
    assert without_processing.arrival_provenance == both.arrival_provenance
    assert [s.value for s in without_processing.seeds] == [s.value for s in both.seeds]


def test_provider_construction_has_no_unknown_provider_fallback():
    with pytest.raises(ValueError, match="unsupported provider"):
        build_provider(SimpleNamespace(algorithm=SimpleNamespace(provider="unknown")))


@pytest.mark.parametrize(
    "target", ["trace.jsonl", "metrics.jsonl", "observations.jsonl"]
)
def test_writer_failure_retains_original_cause_and_actual_progress(
    bundle, monkeypatch, target
):
    import smartsom.experiments.evidence as evidence
    import smartsom.experiments.runner as runner

    actual = []
    original_step = runner.Simulator.step
    original_append = evidence.append_json
    problem = OSError("simulated evidence write failure")

    def step(simulator, action):
        result = original_step(simulator, action)
        actual[:] = simulator.trace
        return result

    def append(stream, value):
        if Path(stream.name).name == target:
            if target == "observations.jsonl" or actual:
                raise problem
        original_append(stream, value)

    monkeypatch.setattr(runner.Simulator, "step", step)
    monkeypatch.setattr(evidence, "append_json", append)
    resolved = resolve_run(run_path(bundle, "processing_arrivals_dispatch"))
    with pytest.raises(RunFailedError) as error:
        run_one(resolved)
    assert error.value.cause is problem
    directory = error.value.run_dir
    summary = json_file(directory, "summary.json")
    assert summary["status"] == "failed" and summary["makespan"] is None
    assert summary["completed_operations"] == sum(r.kind == "complete" for r in actual)
    assert summary["simulation_time"] == (actual[-1].simulation_time if actual else 0)
    assert json_file(directory, "failure.json")["message"] == str(problem)
    assert json_file(directory, "manifest.json")["status"] == "failed"
    assert (directory / "realized_instance.json").exists()
    assert (directory / "realized_processing_times.json").exists()
    assert (directory / "realized_events.jsonl").exists()
    assert json_lines(
        directory, "trace.jsonl"
    )  # Initial decision survives every failure.


def test_artifact_hashes_stream_large_files_and_preserve_exclusions(
    tmp_path, monkeypatch
):
    payload = b"semantic evidence\n" * 100_000
    (tmp_path / "trace.jsonl").write_bytes(payload)
    (tmp_path / "empty.log").touch()
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "summary.json.tmp").write_text("unfinished")

    def forbid_whole_file_reads(*args):
        raise AssertionError("artifact digest must not load an entire file")

    monkeypatch.setattr(Path, "read_bytes", forbid_whole_file_reads)
    assert artifact_digests(tmp_path) == {
        "empty.log": hashlib.sha256(b"").hexdigest(),
        "trace.jsonl": hashlib.sha256(payload).hexdigest(),
    }
