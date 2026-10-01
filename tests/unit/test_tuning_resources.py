"""Measured allocation precedence and resource lifecycle admission."""

import subprocess
from types import SimpleNamespace

import pytest

from smartsom.experiments.tuning_resources import (
    GIB,
    ExecutionProfile,
    GPUInfo,
    ProcessUsage,
    ResourceBroker,
    ResourceMonitor,
    ResourceRequest,
    ResourceSnapshot,
    ResourceUnavailable,
)


def snapshot(**updates):
    return ResourceSnapshot(
        **(
            {
                "cpus": 16.0,
                "memory_total": 32 * GIB,
                "memory_available": 24 * GIB,
                "external_cpu_load": 0.0,
            }
            | updates
        )
    )


class Process:
    def __init__(self, pid=10):
        self.pid = pid

    def cpu_affinity(self):
        return list(range(8))

    def create_time(self):
        return self.pid

    def cpu_percent(self, interval=None):
        return 100.0

    def memory_info(self):
        return SimpleNamespace(rss=GIB)

    def name(self):
        return "owned" if self.pid in (10, 11) else "external"

    def children(self, recursive=False):
        return [Process(11)] if self.pid == 10 else []


class Psutil:
    Error = OSError
    Process = Process

    @staticmethod
    def cpu_count():
        return 16

    @staticmethod
    def virtual_memory():
        return SimpleNamespace(total=64 * GIB, available=40 * GIB)

    @staticmethod
    def cpu_percent(interval=None, percpu=False):
        return [50.0] * 8 + [0.0] * 8 if percpu else 25.0

    @staticmethod
    def process_iter():
        return [Process(10), Process(11), Process(99)]


def no_gpu(*args, **kwargs):
    raise FileNotFoundError("nvidia-smi")


def test_nested_allocation_cpu_memory_and_excluded_tree(tmp_path):
    (tmp_path / "cpu.max").write_text("600000 100000")
    (tmp_path / "memory.max").write_text(str(20 * GIB))
    (tmp_path / "memory.current").write_text(str(7 * GIB))
    monitor = ResourceMonitor(
        psutil_module=Psutil,
        environ={"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": str(16 * 1024)},
        cgroup_root=tmp_path,
        run_command=no_gpu,
    )
    first = monitor.snapshot(exclude_pids=(10,))
    assert first.cpus == 4 and first.external_cpu_load is None
    assert "CPU load first sample unavailable" in first.unavailable
    second = monitor.snapshot(exclude_pids=(10,))
    assert second.memory_total == 16 * GIB
    assert second.memory_available == 13 * GIB
    assert second.cpu_load == 100.0 and second.external_cpu_load == 50.0
    assert {p.pid for p in second.processes if p.excluded} == {10, 11}
    assert second.limits == {
        "affinity_cpus": 8,
        "cgroup_cpus": 6.0,
        "slurm_cpus": 4,
        "cgroup_memory": 20 * GIB,
        "slurm_memory": 16 * GIB,
    }
    assert "nvidia GPU metrics unavailable" in second.unavailable


def test_gpu_visible_order_and_bounded_query(tmp_path):
    calls = []

    def query(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            stdout="0, GPU-A, A, 12000, 9000, 10\n2, GPU-C, C, 24000, 20000, 50\n"
        )

    monitor = ResourceMonitor(
        psutil_module=Psutil,
        environ={"CUDA_VISIBLE_DEVICES": "2,GPU-A"},
        cgroup_root=tmp_path,
        run_command=query,
    )
    result = monitor.snapshot()
    assert [(gpu.index, gpu.name) for gpu in result.gpus] == [("0", "C"), ("1", "A")]
    assert result.gpus[0].memory_available == 20000 * 1024**2
    assert calls[0][1]["timeout"] == 2.0 and calls[0][1]["check"]
    assert isinstance(calls[0][0], list)


@pytest.mark.parametrize("visible", ["", "-1"])
def test_disabled_cuda_does_not_query_driver(tmp_path, visible):
    monitor = ResourceMonitor(
        psutil_module=Psutil,
        environ={"CUDA_VISIBLE_DEVICES": visible},
        cgroup_root=tmp_path,
        run_command=lambda *a, **kw: pytest.fail("GPU driver queried"),
    )
    assert monitor.snapshot().gpus == ()


def test_unknown_mig_and_gpu_timeout_are_missing_observations(tmp_path):
    monitor = ResourceMonitor(
        psutil_module=Psutil,
        environ={"CUDA_VISIBLE_DEVICES": "MIG-unknown"},
        cgroup_root=tmp_path,
        run_command=lambda *a, **kw: SimpleNamespace(
            stdout="0, GPU-A, A, 12000, 9000, 10\n"
        ),
    )
    result = monitor.snapshot()
    assert not result.gpus and any("unresolved" in s for s in result.unavailable)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 2)

    monitor.run_command = timeout
    assert "nvidia GPU metrics unavailable" in monitor.snapshot().unavailable


def test_office_and_throughput_reserves_and_owned_rss_count_once():
    observation = snapshot(
        memory_available=10 * GIB,
        processes=(ProcessUsage(1, "ours", 2.0, 4 * GIB, True),),
    )
    office = ResourceBroker(mode="balanced")
    throughput = ResourceBroker(mode="performance")
    assert office.capacity(observation).cpus == 12.8
    assert office.capacity(observation).memory == 14 * GIB - int(32 * GIB * 0.2)
    assert throughput.capacity(observation).cpus == 15.0
    assert throughput.capacity(observation).memory == 14 * GIB - int(32 * GIB * 0.1)


def test_all_lease_lifecycle_states_hold_capacity_until_release():
    broker = ResourceBroker(mode="performance")
    observation = snapshot()
    request = ResourceRequest(7, 2 * GIB)
    broker.reserve("one", request, snapshot=observation)
    broker.reserve("two", request, snapshot=observation, state="staged")
    assert not broker.admit(ResourceRequest(2, GIB), observation).allowed
    for state in ("running", "stopping"):
        broker.transition("one", state, pids=(101,))
        assert not broker.admit(ResourceRequest(2, GIB), observation).allowed
    broker.release("one")
    assert broker.admit(ResourceRequest(2, GIB), observation).allowed
    with pytest.raises(ValueError, match="already exists"):
        broker.reserve("two", request, snapshot=observation)


def test_memory_peak_margin_and_explicit_gpu_budget():
    broker = ResourceBroker(mode="performance")
    observation = snapshot(
        memory_total=16 * GIB,
        memory_available=4 * GIB,
        gpus=(GPUInfo("0", "H20", 16 * GIB, 4 * GIB),),
    )
    assert broker.admit(ResourceRequest(1, GIB, "0", 3 * GIB), observation).allowed
    assert not broker.admit(ResourceRequest(1, GIB, "0", 4 * GIB), observation).allowed
    assert not broker.admit(ResourceRequest(1, GIB, "1", GIB), observation).allowed
    assert not broker.admit(ResourceRequest(1, 2 * GIB), observation).allowed
    with pytest.raises(ResourceUnavailable):
        broker.reserve("too-large", ResourceRequest(1, 2 * GIB), snapshot=observation)
    assert not broker.leases


def test_affinity_load_excludes_busy_processors_outside_the_allocation(tmp_path):
    class OtherAllocationBusy(Psutil):
        @staticmethod
        def cpu_percent(interval=None, percpu=False):
            return [0.0] * 8 + [100.0] * 8 if percpu else 50.0

    monitor = ResourceMonitor(
        psutil_module=OtherAllocationBusy,
        environ={"CUDA_VISIBLE_DEVICES": ""},
        cgroup_root=tmp_path,
    )
    monitor.snapshot()
    result = monitor.snapshot()
    assert result.cpu_load == 0 and result.external_cpu_load == 0
    assert result.cpus == 8


def test_v1_quota_and_memory_limits_are_read_in_bytes(tmp_path):
    (tmp_path / "cpu.cfs_quota_us").write_text("150000")
    (tmp_path / "cpu.cfs_period_us").write_text("100000")
    (tmp_path / "memory.limit_in_bytes").write_text(str(8 * GIB))
    (tmp_path / "memory.usage_in_bytes").write_text(str(3 * GIB))
    result = ResourceMonitor(
        psutil_module=Psutil, environ={"CUDA_VISIBLE_DEVICES": ""}, cgroup_root=tmp_path
    ).snapshot()
    assert result.cpus == 1.5
    assert result.memory_total == 8 * GIB and result.memory_available == 5 * GIB


def test_broker_excludes_lease_processes_when_observing_background_load():
    calls = []

    class Monitor:
        def snapshot(self, exclude_pids=()):
            calls.append(exclude_pids)
            return snapshot()

    broker = ResourceBroker(Monitor(), mode="performance")
    broker.reserve("one", ResourceRequest(1, GIB), snapshot=snapshot(), pids=(200,))
    broker.snapshot(exclude_pids=(100,))
    assert set(calls[-1]) == {100, 200}


@pytest.mark.parametrize(
    "profile", [(0, 1, "cpu"), (1, True, "cpu"), (1, 1, "mps"), (1, 1, "cuda:-1")]
)
def test_invalid_execution_profiles(profile):
    with pytest.raises(ValueError):
        ExecutionProfile(*profile)
