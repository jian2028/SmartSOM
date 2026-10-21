"""Stage-envelope grouping, persistence, and shared admission consumption."""

import copy
import json

import pytest

from smartsom.experiments import queue_admission, runtime_control
from smartsom.experiments import runtime_resources as model
from smartsom.experiments.evidence import write_json


def prepared():
    return {
        key + "_json": json.dumps(value)
        for key, value in {
            "config": {
                "seed": 1,
                "scenario": "a.yaml",
                "composition": "b.yaml",
                "output": {"name": "one", "root": "runs/a"},
                "training": {
                    "algorithm": "ppo",
                    "parameters": "parameters-a.yaml",
                    "groups": ["machine"],
                },
                "runtime": {
                    "numerical_threads": 1,
                    "sampling_processes": 0,
                    "num_envs": 1,
                },
                "validation": {"enabled": True, "seed": 2, "scenarios": ["v.yaml"]},
            },
            "scenario": {
                "factory": {"machines": [{"machine_id": "m", "capacity": 2}]},
                "demands": [
                    {
                        "release_at": 1,
                        "steps": [{"operation_type": "a"}, {"operation_type": "b"}],
                    }
                ],
            },
            "composition": {"matching": {"name": "nearest"}},
            "parameters": {"batch_size": 16, "network": [32, 32]},
            "policies": {"machine": {"network": {"hidden": [32, 32]}}},
            "training_inputs": {
                "settings": {
                    "factory": "old.yaml",
                    "workload": "jobs.yaml",
                    "tick_limit": 10,
                },
                "workload": {"rate": 2},
            },
            "validation": [
                {
                    "seed": 2,
                    "scenario": {"demands": []},
                    "recipe": {
                        "settings": {"factory": "f.yaml", "workload": "w.yaml"},
                        "workload": {"rate": 3},
                    },
                }
            ],
            "evaluation": [],
        }.items()
    }


def change(data, field, mutate):
    result = copy.deepcopy(data)
    value = json.loads(result[field + "_json"])
    mutate(value)
    result[field + "_json"] = json.dumps(value)
    return result


def test_run_labels_seeds_and_resolved_input_locations_do_not_split_groups():
    original = prepared()
    variant = change(
        original,
        "config",
        lambda c: c.update(
            seed=99,
            scenario="renamed.yaml",
            composition="elsewhere.yaml",
            output={"name": "two", "root": "other"},
        ),
    )
    variant = change(
        variant,
        "config",
        lambda c: c["training"].update(parameters="new-parameters.yaml"),
    )
    variant = change(
        variant,
        "config",
        lambda c: c["validation"].update(seed=999, scenarios=["new-validation.yaml"]),
    )
    variant = change(
        variant,
        "training_inputs",
        lambda c: c["settings"].update(factory="renamed.yaml", workload="new.yaml"),
    )
    variant = change(variant, "validation", lambda c: c[0].update(seed=999))
    variant = change(
        variant,
        "validation",
        lambda c: c[0]["recipe"]["settings"].update(
            factory="other.yaml", workload="elsewhere.yaml"
        ),
    )
    assert model.describe(original) == model.describe(variant)


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("scenario", lambda c: c["factory"]["machines"][0].update(capacity=3)),
        ("scenario", lambda c: c["demands"][0].update(release_at=2)),
        ("scenario", lambda c: c["demands"][0]["steps"].reverse()),
        ("scenario", lambda c: c["factory"]["machines"][0].update(machine_id="other")),
        ("parameters", lambda c: c.update(network=[64, 64])),
        ("policies", lambda c: c["machine"]["network"].update(hidden=[128])),
        ("config", lambda c: c["runtime"].update(sampling_processes=2)),
        ("config", lambda c: c["runtime"].update(num_envs=4)),
        ("training_inputs", lambda c: c["workload"].update(rate=5)),
    ],
)
def test_semantic_workloads_models_and_runtime_layouts_do_not_merge(field, mutate):
    original = prepared()
    assert (
        model.describe(original)["key"]
        != model.describe(change(original, field, mutate))["key"]
    )


def test_phase_envelope_persists_peaks_and_unknown_cpu(tmp_path):
    descriptor = model.describe(prepared())
    model.observe(tmp_path, descriptor, "sampling", memory=300 * 1024**2, cpus=None)
    model.observe(tmp_path, descriptor, "validation", memory=800 * 1024**2, cpus=2)
    prediction = model.estimate(tmp_path, descriptor, cpus=1, memory=model.MEMORY_FLOOR)
    assert prediction["memory"] == 1200 * 1024**2
    assert prediction["cpus"] == pytest.approx(2.2)
    assert "sampling" in prediction["unknown_stages"]
    row = model.snapshot(tmp_path)["groups"][descriptor["key"]]["stages"]["sampling"]
    assert row["cpu_peak"] is None and row["cpu_samples"] == 0
    model.observe(tmp_path, descriptor, "validation", memory=10, cpus=0.1)
    assert (
        model.estimate(tmp_path, descriptor, cpus=1, memory=model.MEMORY_FLOOR)
        == prediction
    )
    fresh = model.estimate(tmp_path / "fresh", descriptor, cpus=1, memory=1)
    assert fresh["memory"] == int(model.MEMORY_FLOOR * model.UNKNOWN_FACTOR)


def test_shared_admission_consumes_future_validation_envelope(tmp_path):
    runtime_control.initialize(tmp_path, 4)
    write_json(tmp_path / "control/owner.json", {"id": "driver"})
    queue_admission.initialize(tmp_path, supported=True)
    runtime_control.submit(tmp_path, "budget", value={"cpus": 4, "memory": 2 * 1024**3})
    runtime_control.apply(tmp_path)
    descriptor = model.describe(prepared())
    model.observe(tmp_path, descriptor, "validation", memory=600 * 1024**2, cpus=2)
    request = {"descriptor": descriptor, "cpus": 1, "memory": model.MEMORY_FLOOR}
    assert queue_admission.reserve(
        tmp_path, "native:first", owner_id="driver", resource_model=request
    )
    assert not queue_admission.reserve(
        tmp_path, "source:second", owner_id="driver", resource_model=request
    )
    slot = queue_admission.snapshot(tmp_path)["slots"]["native:first"]
    assert slot["memory"] == 1125 * 1024**2
    assert slot["cpus"] == pytest.approx(2.2)
    assert "optimizing" in slot["resource_estimate"]["unknown_stages"]


def test_boundary_evidence_ingestion_is_idempotent(tmp_path):
    descriptor = model.describe(prepared())
    worker = tmp_path / "worker"
    (worker / "control").mkdir(parents=True)
    data = {
        "id": "generation",
        "updated_at": 1,
        "stages": {
            "optimizing": {
                "memory_peak": 50,
                "cpu_peak": 0.5,
                "memory_samples": 2,
                "cpu_samples": 1,
            }
        },
    }
    write_json(worker / "control/stage-resources.json", data)
    for _ in range(2):
        model.ingest(tmp_path, descriptor, worker)
    row = model.snapshot(tmp_path)["groups"][descriptor["key"]]["stages"]["optimizing"]
    assert row["memory_samples"] == 2 and row["cpu_samples"] == 1


def test_ingest_preserves_actual_profile_and_monotonic_source_counts(tmp_path):
    original = prepared()
    old = model.describe(original, numerical_threads=1)
    new = model.describe(original, numerical_threads=2)
    assert old["key"] != new["key"]
    worker = tmp_path / "worker"
    (worker / "control").mkdir(parents=True)
    data = {
        "id": "old-profile-generation",
        "resource_descriptor": old,
        "updated_at": 2,
        "stages": {
            "validation": {
                "memory_peak": 1024**3,
                "cpu_peak": 1,
                "memory_samples": 2,
                "cpu_samples": 2,
            }
        },
    }
    write_json(worker / "control/stage-resources.json", data)
    model.ingest(tmp_path, new, worker)
    groups = model.snapshot(tmp_path)["groups"]
    assert old["key"] in groups and new["key"] not in groups
    data["stages"]["validation"].update(memory_samples=1, cpu_samples=1)
    write_json(worker / "control/stage-resources.json", data)
    model.ingest(tmp_path, new, worker)
    data["stages"]["validation"].update(memory_samples=2, cpu_samples=2)
    write_json(worker / "control/stage-resources.json", data)
    model.ingest(tmp_path, new, worker)
    row = model.snapshot(tmp_path)["groups"][old["key"]]["stages"]["validation"]
    assert row["memory_samples"] == row["cpu_samples"] == 2


def test_short_cpu_quantization_window_remains_unknown(monkeypatch):
    from types import SimpleNamespace

    import psutil

    clock = [0.0]
    cpu = [0.0]

    class Process:
        pid = 99

        def __init__(self, pid):
            pass

        def children(self, **kwargs):
            return []

        def create_time(self):
            return 1

        def cpu_times(self):
            return SimpleNamespace(user=cpu[0], system=0)

        def memory_info(self):
            return SimpleNamespace(rss=1000)

    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(model.time, "monotonic", lambda: clock[0])
    sampler = model.TreeSampler()
    assert sampler.sample(99, "optimizing") == (1000, None)
    clock[0], cpu[0] = 0.001, 0.01
    assert sampler.sample(99, "optimizing") == (1000, None)
    clock[0] = 0.101
    assert sampler.sample(99, "optimizing")[1] == pytest.approx(0.01 / 0.101)


def test_tune_request_consumes_stage_model_and_separates_thread_profiles(tmp_path):
    from smartsom.experiments.tuning_broker import AdaptiveBroker

    shape = prepared()
    values = {"CPU": 1, "GPU": 0, "memory": 1024**3}
    profile = {
        **values,
        "concurrency": 2,
        "throughput": 1,
        "update_seconds": 1,
        "restart_seconds": 1,
        "gpu_memory": 0,
    }
    broker = AdaptiveBroker(
        [
            {
                "experiment_id": "one",
                "prepared": shape,
                "run_dir": str(tmp_path / "worker"),
            }
        ],
        {"one": [profile]},
        control_root=tmp_path,
    )
    descriptor = broker._stage_descriptor("one", values)
    model.observe(tmp_path, descriptor, "validation", memory=2 * 1024**3, cpus=2)
    request = broker._request("one", values)
    assert request.memory == 3 * 1024**3
    assert request.cpus == pytest.approx(2.2)
    changed = broker._stage_descriptor("one", {**values, "CPU": 2})
    assert changed["key"] != descriptor["key"]
    assert descriptor == model.describe(shape, task="train-evaluate")


def test_default_host_guard_sums_future_envelopes_without_operator_budget(tmp_path):
    from smartsom.experiments.tuning_resources import ResourceSnapshot

    runtime_control.initialize(tmp_path, 8)
    write_json(tmp_path / "control/owner.json", {"id": "driver"})
    queue_admission.initialize(tmp_path, supported=True)
    descriptor = model.describe(prepared())
    model.observe(tmp_path, descriptor, "validation", memory=2 * 1024**3, cpus=1)
    # Each future envelope is3GiB raw/3.75GiB charged. Two cannot fit the
    # 6GiB usable host, even while current startup RSS is almost zero.
    host = ResourceSnapshot(
        cpus=8, memory_total=8 * 1024**3, memory_available=8 * 1024**3
    )
    request = {"descriptor": descriptor, "cpus": 1, "memory": model.MEMORY_FLOOR}
    assert queue_admission.reserve(
        tmp_path, "native:first", owner_id="driver", resource_model=request, host=host
    )
    assert runtime_control.snapshot(tmp_path)["budget"] == {}
    assert not queue_admission.reserve(
        tmp_path, "source:second", owner_id="driver", resource_model=request, host=host
    )
    # Unverified observations never create capacity from somebody else's RSS.
    queue_admission.update_resources(
        tmp_path,
        "native:first",
        owner_id="driver",
        memory=1,
        observed_memory=3 * 1024**3,
        observed_cpus=1,
    )
    pressure = ResourceSnapshot(
        cpus=8, memory_total=8 * 1024**3, memory_available=4 * 1024**3
    )
    assert not queue_admission.reserve(
        tmp_path, "source:small", owner_id="driver", cpus=1, memory=100, host=pressure
    )


def test_tune_model_highwater_survives_full_stage_coverage_and_pickle(tmp_path):
    import pickle

    from smartsom.experiments.tuning_broker import AdaptiveBroker

    shape = prepared()
    values = {"CPU": 1, "GPU": 0, "memory": 1024**3}
    profile = {
        **values,
        "concurrency": 2,
        "throughput": 1,
        "update_seconds": 1,
        "restart_seconds": 1,
        "gpu_memory": 0,
    }
    broker = AdaptiveBroker(
        [
            {
                "experiment_id": "one",
                "prepared": shape,
                "run_dir": str(tmp_path / "worker"),
            }
        ],
        {"one": [profile]},
        control_root=tmp_path,
    )
    descriptor = broker._stage_descriptor("one", values)
    model.observe(tmp_path, descriptor, "validation", memory=2 * 1024**3, cpus=1)
    assert broker._request("one", values).memory == 3 * 1024**3
    for phase in descriptor["stages"]:
        model.observe(tmp_path, descriptor, phase, memory=model.MEMORY_FLOOR, cpus=0.5)
    estimate = model.estimate(tmp_path, descriptor, cpus=1, memory=1024**3)
    assert not estimate["unknown_stages"]
    assert estimate["memory"] == 2 * 1024**3
    assert broker._request("one", values).memory == 3 * 1024**3
    # Looking ahead to a different profile cannot overwrite the old profile's
    # reservation history, including across serialization of the coordinator.
    broker._stage_request("one", {**values, "CPU": 2})
    restored = pickle.loads(pickle.dumps(broker))
    assert restored._request("one", values).memory == 3 * 1024**3
