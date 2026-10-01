"""Allocation-aware resource observations and explicit lifecycle reservations.

Loads system monitoring only when requested. Logical reservations are admission
accounting, never a claim of operating-system CPU or memory enforcement.
"""

import csv
import math
import os
import subprocess
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

GIB = 1024**3
LEASE_STATES = frozenset({"pending", "staged", "running", "stopping"})


class ResourceUnavailable(RuntimeError):
    """A required observation could not be obtained; it is not a zero value."""


@dataclass(frozen=True)
class GPUInfo:
    index: str
    name: str
    memory_total: int | None = None
    memory_available: int | None = None
    utilization: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ProcessUsage:
    pid: int
    name: str
    cpu_cores: float | None
    rss: int
    excluded: bool = False


@dataclass(frozen=True)
class ResourceSnapshot:
    cpus: float
    memory_total: int
    memory_available: int
    gpus: tuple[GPUInfo, ...] = ()
    cpu_load: float = 0.0
    external_cpu_load: float | None = None
    latency_ms: float | None = None
    processes: tuple[ProcessUsage, ...] = ()
    limits: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    unavailable: tuple[str, ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("effective CPUs must be positive and finite")
        if not 0 <= self.memory_available <= self.memory_total:
            raise ValueError("memory observation must be bounded by total memory")
        for load in (self.cpu_load, self.external_cpu_load):
            if load is not None and (not math.isfinite(load) or not 0 <= load <= 100):
                raise ValueError("CPU load uses percent of effective allocation")


@dataclass(frozen=True)
class ExecutionProfile:
    threads: int
    concurrency: int
    device: str = "cpu"
    num_envs: int = 1
    sampling_processes: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.threads, self.concurrency)):
            raise ValueError("threads and concurrency must be positive integers")
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(self.sampling_processes) is not int or self.sampling_processes < 0:
            raise ValueError("sampling_processes must be nonnegative")
        if self.sampling_processes and (
            self.num_envs < 2 or self.sampling_processes > self.num_envs
        ):
            raise ValueError(
                "parallel sampling needs at least two environments and no more processes than environments"
            )
        if (
            self.device != "cpu"
            and self.device != "cuda"
            and not self.device.startswith("cuda:")
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<visible index>")
        if self.device.startswith("cuda:") and not self.device[5:].isdecimal():
            raise ValueError("CUDA device requires a nonnegative visible index")


@dataclass(frozen=True)
class ResourceRequest:
    cpus: float
    memory: int
    gpu: str | None = None
    gpu_memory: int = 0

    def __post_init__(self):
        if not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("CPU request must be positive and finite")
        if type(self.memory) is not int or self.memory < 0:
            raise ValueError("memory request must be nonnegative bytes")
        if type(self.gpu_memory) is not int or self.gpu_memory < 0:
            raise ValueError("GPU memory request must be nonnegative bytes")
        if self.gpu is None and self.gpu_memory:
            raise ValueError("GPU memory requires an explicit GPU")


@dataclass(frozen=True)
class ResourceCapacity:
    cpus: float
    memory: int
    gpu_memory: dict[str, int]
    latency_ms: float | None = None


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str | None
    capacity: ResourceCapacity


@dataclass(frozen=True)
class ResourceLease:
    key: str
    request: ResourceRequest
    state: str = "pending"
    pids: tuple[int, ...] = ()


def _number(value):
    try:
        number = int(str(value).split("(", 1)[0])
        return number if number > 0 else None
    except (ValueError, TypeError):
        return None


class ResourceMonitor:
    """Read effective allocation and current load without importing GPU runtimes.

    CPU load is a percentage of the effective allocation, capped at 100. Initial
    psutil CPU samples are explicitly unavailable until a second observation.
    NVIDIA queries are bounded and respect CUDA visibility. Unsupported MIG
    observations remain missing rather than claiming a whole GPU's capacity.
    """

    def __init__(
        self,
        *,
        psutil_module=None,
        environ=None,
        cgroup_root=Path("/sys/fs/cgroup"),
        run_command=subprocess.run,
        latency_probe=None,
        gpu_timeout=2.0,
    ):
        self._psutil = psutil_module
        self.environ = os.environ if environ is None else environ
        self.cgroup_root = Path(cgroup_root)
        self.run_command, self.latency_probe = run_command, latency_probe
        self.gpu_timeout = gpu_timeout
        self._cpu_primed = False
        self._processes = {}

    def _backend(self):
        if self._psutil is None:
            try:
                import psutil
            except ImportError as exc:
                raise ResourceUnavailable(
                    "resource monitoring requires the optional tuning dependencies"
                ) from exc
            self._psutil = psutil
        return self._psutil

    def _cgroup(self):
        root = self.cgroup_root
        # Resolve a nested cgroup-v2 allocation, never only the host root limit.
        if root == Path("/sys/fs/cgroup"):
            try:
                for line in Path("/proc/self/cgroup").read_text().splitlines():
                    if line.startswith("0::"):
                        nested = root / line[3:].lstrip("/")
                        if nested.is_dir():
                            root = nested
            except OSError:
                pass
        return root

    @staticmethod
    def _read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return None

    def _gpus(self, unavailable):
        visible = self.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip() in {"", "-1"}:
            return ()
        try:
            result = self.run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self.gpu_timeout,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            unavailable.append("nvidia GPU metrics unavailable")
            return ()
        records = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        selected = records
        if visible is not None:
            selected = []
            for token in (t.strip() for t in visible.split(",")):
                match = next(
                    (r for r in records if len(r) == 6 and token in {r[0], r[1]}),
                    None,
                )
                if match is None:
                    unavailable.append(f"GPU visibility/metrics unresolved: {token}")
                else:
                    selected.append(match)
        if visible is None:
            count = _number(self.environ.get("SLURM_GPUS_ON_NODE"))
            if count is not None:
                # Without device identifiers do not guess which physical GPU is ours.
                unavailable.append(
                    "Slurm GPU identifiers missing; CUDA visibility needed"
                )
                return ()
        gpus = []
        for index, row in enumerate(selected):
            if len(row) != 6:
                unavailable.append("malformed nvidia-smi observation")
                continue
            try:
                total, free = int(float(row[3]) * 1024**2), int(float(row[4]) * 1024**2)
                utilization = float(row[5])
                if not 0 <= free <= total or not 0 <= utilization <= 100:
                    raise ValueError
                gpus.append(GPUInfo(str(index), row[2], total, free, utilization))
            except ValueError:
                gpus.append(
                    GPUInfo(str(index), row[2], reason="GPU metrics unavailable")
                )
        return tuple(gpus)

    def snapshot(self, exclude_pids=()):
        psutil = self._backend()
        unavailable, limits, origins = [], {}, {}
        host_cpus = psutil.cpu_count() or 1
        cpus = float(host_cpus)
        affinity = None
        origins["cpu"] = "psutil logical CPU count"
        try:
            affinity = psutil.Process().cpu_affinity()
            if affinity:
                limits["affinity_cpus"] = len(affinity)
                cpus = min(cpus, len(affinity))
        except (AttributeError, OSError, psutil.Error):
            unavailable.append("CPU affinity unavailable")
        cg = self._cgroup()
        cgroups = [cg]
        while cgroups[-1] != self.cgroup_root and cgroups[-1].is_relative_to(
            self.cgroup_root
        ):
            cgroups.append(cgroups[-1].parent)
        for cgroup in cgroups:
            raw = self._read(cgroup / "cpu.max")
            parts = raw.split() if raw else []
            if not parts:
                quota_us = self._read(cgroup / "cpu.cfs_quota_us")
                period_us = self._read(cgroup / "cpu.cfs_period_us")
                if quota_us and period_us:
                    parts = [quota_us, period_us]
            if len(parts) == 2 and parts[0] != "max":
                try:
                    quota = int(parts[0]) / int(parts[1])
                    if quota > 0:
                        cpus = min(cpus, quota)
                        limits["cgroup_cpus"] = min(
                            quota, limits.get("cgroup_cpus", quota)
                        )
                except (ValueError, ZeroDivisionError):
                    unavailable.append("cgroup CPU quota unreadable")
        slurm_cpus = _number(self.environ.get("SLURM_CPUS_PER_TASK"))
        if slurm_cpus is not None:
            cpus = min(cpus, slurm_cpus)
            limits["slurm_cpus"] = slurm_cpus
        vm = psutil.virtual_memory()
        total, available = int(vm.total), int(vm.available)
        origins["memory"] = "psutil available memory constrained by allocation"
        for cgroup in cgroups:
            cg_memory = _number(self._read(cgroup / "memory.max")) or _number(
                self._read(cgroup / "memory.limit_in_bytes")
            )
            if cg_memory is not None:
                total = min(total, cg_memory)
                usage = (
                    _number(self._read(cgroup / "memory.current"))
                    or _number(self._read(cgroup / "memory.usage_in_bytes"))
                    or 0
                )
                available = min(available, max(0, cg_memory - usage))
                limits["cgroup_memory"] = min(
                    cg_memory, limits.get("cgroup_memory", cg_memory)
                )
        slurm_memory = _number(self.environ.get("SLURM_MEM_PER_NODE"))
        if slurm_memory is None:
            per_cpu = _number(self.environ.get("SLURM_MEM_PER_CPU"))
            if per_cpu is not None:
                slurm_memory = int(per_cpu * cpus)
        if slurm_memory is not None:
            total = min(total, slurm_memory * 1024**2)
            limits["slurm_memory"] = slurm_memory * 1024**2
        available = min(available, total)
        excluded = set(exclude_pids)
        for pid in tuple(excluded):
            try:
                excluded.update(
                    p.pid for p in psutil.Process(pid).children(recursive=True)
                )
            except (OSError, psutil.Error):
                pass
        processes, owned_cpu = [], 0.0
        current = {}
        for process in psutil.process_iter():
            try:
                pid = process.pid
                previous = self._processes.get(pid)
                # PID reuse cannot reuse a former process's CPU counter.
                same = (
                    previous is not None
                    and previous.create_time() == process.create_time()
                )
                measured = previous if same else process
                cpu = measured.cpu_percent(interval=None) / 100
                cores = cpu if same else None
                rss, name = process.memory_info().rss, process.name()
                current[pid] = measured
                processes.append(ProcessUsage(pid, name, cores, rss, pid in excluded))
                if pid in excluded and cores is not None:
                    owned_cpu += cores
            except (OSError, psutil.Error):
                continue
        self._processes = current
        if affinity:
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            used = (
                sum(per_cpu[index] for index in affinity if index < len(per_cpu)) / 100
            )
            origins["cpu_load"] = "psutil per-CPU load inside affinity mask"
        else:
            used = psutil.cpu_percent(interval=None) * host_cpus / 100
            origins["cpu_load"] = "psutil host CPU load; affinity unavailable"
        load = min(100.0, max(0.0, used / cpus * 100))
        external = min(100.0, max(0.0, (used - owned_cpu) / cpus * 100))
        if not self._cpu_primed:
            unavailable.append("CPU load first sample unavailable")
            load, external = 0.0, None
        self._cpu_primed = True
        try:
            latency = self.latency_probe() if self.latency_probe else None
        except (OSError, ResourceUnavailable):
            latency = None
        if latency is None:
            unavailable.append("interactive latency probe unavailable")
        return ResourceSnapshot(
            cpus,
            total,
            available,
            self._gpus(unavailable),
            load,
            external,
            latency,
            tuple(processes),
            limits,
            origins,
            tuple(unavailable),
        )


class ResourceBroker:
    """All lifecycle reservations count until an explicit verified release.

    Pending, staged and stopping workers consume the same admission budget as
    running workers. Account owned RSS once by restoring it to observed available
    RAM before charging reservations. Memory requests represent measured peak
    bytes; the peak factor is applied here, not at the call site.
    """

    def __init__(self, monitor=None, *, mode="balanced", peak_factor=1.25):
        # Historical frozen Tune plans retain the original spelling on resume.
        mode = {"office": "balanced", "throughput": "performance"}.get(mode, mode)
        if mode not in {"balanced", "performance"}:
            raise ValueError("mode must be balanced or performance")
        if not math.isfinite(peak_factor) or peak_factor < 1:
            raise ValueError("peak_factor must be at least one")
        self.monitor, self.mode, self.peak_factor = (
            monitor or ResourceMonitor(),
            mode,
            peak_factor,
        )
        self.leases = {}
        self._lock = threading.RLock()

    def snapshot(self, exclude_pids=()):
        pids = set(exclude_pids)
        for lease in self.leases.values():
            pids.update(lease.pids)
        return self.monitor.snapshot(exclude_pids=tuple(pids))

    def capacity(self, snapshot):
        cpu_reserve = max(
            1.0, snapshot.cpus * (0.2 if self.mode == "balanced" else 0.05)
        )
        ram_reserve = (
            max(2 * GIB, int(snapshot.memory_total * 0.2))
            if self.mode == "balanced"
            else max(GIB, int(snapshot.memory_total * 0.1))
        )
        external = snapshot.external_cpu_load
        if external is None:
            external = snapshot.cpu_load
        owned_rss = sum(p.rss for p in snapshot.processes if p.excluded)
        gpu_memory = {
            gpu.index: int(gpu.memory_available)
            for gpu in snapshot.gpus
            if gpu.memory_available is not None and gpu.reason is None
        }
        return ResourceCapacity(
            max(0.0, snapshot.cpus * (1 - external / 100) - cpu_reserve),
            max(
                0,
                min(snapshot.memory_total, snapshot.memory_available + owned_rss)
                - ram_reserve,
            ),
            gpu_memory,
            snapshot.latency_ms,
        )

    def admit(self, request, snapshot=None):
        snapshot = snapshot or self.snapshot()
        capacity = self.capacity(snapshot)
        with self._lock:
            cpu_used = sum(lease.request.cpus for lease in self.leases.values())
            memory_used = sum(
                math.ceil(lease.request.memory * self.peak_factor)
                for lease in self.leases.values()
            )
            reason = None
            if cpu_used + request.cpus > capacity.cpus + 1e-9:
                reason = "CPU allocation or external load leaves insufficient capacity"
            elif (
                memory_used + math.ceil(request.memory * self.peak_factor)
                > capacity.memory
            ):
                reason = "available RAM below peak plus reserve budget"
            elif request.gpu is not None:
                available = capacity.gpu_memory.get(request.gpu)
                used = sum(
                    math.ceil(lease.request.gpu_memory * self.peak_factor)
                    for lease in self.leases.values()
                    if lease.request.gpu == request.gpu
                )
                if available is None:
                    reason = "requested GPU capacity unavailable"
                elif (
                    used + math.ceil(request.gpu_memory * self.peak_factor) > available
                ):
                    reason = "available GPU memory below peak budget"
            return Admission(reason is None, reason, capacity)

    def reserve(self, key, request, *, state="pending", snapshot=None, pids=()):
        if state not in LEASE_STATES:
            raise ValueError("invalid resource lease lifecycle state")
        with self._lock:
            if key in self.leases:
                raise ValueError("resource lease already exists")
            decision = self.admit(request, snapshot)
            if not decision.allowed:
                raise ResourceUnavailable(decision.reason)
            lease = ResourceLease(key, request, state, tuple(pids))
            self.leases[key] = lease
            return lease

    def transition(self, key, state, *, pids=None):
        if state not in LEASE_STATES:
            raise ValueError("invalid resource lease lifecycle state")
        with self._lock:
            lease = self.leases[key]
            changed = replace(
                lease, state=state, pids=lease.pids if pids is None else tuple(pids)
            )
            self.leases[key] = changed
            return changed

    def release(self, key):
        with self._lock:
            return self.leases.pop(key)
