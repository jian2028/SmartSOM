"""Real native actors exercise online admission and checkpoint-safe parking.

The real-clock case deliberately paces complete native updates. It qualifies the
production 30-second elapsed-clock gates, not unmodified training throughput.
The controlled case supplies OnlineConcurrency.observe's clock and stable
external CPU input so real-host CPU noise cannot reset its plateau windows.
Actual CPU count, RAM, actor/process ownership and broker clocks remain real.
Raw CPU samples/priming/exclusions and EMA inputs remain diagnostic evidence;
controlled numbers are never hardware throughput evidence. The real-clock case
retains the original resource monitor and pressure guards.
Both cases run in deadline-bounded, privately owned subprocesses.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("torch")
psutil = pytest.importorskip("psutil")
from ray.tune import Callback  # noqa: E402
from test_tuning_ray_native_resize import (  # noqa: E402
    _assert_native_state_equal,
    _state,
)

from smartsom import api  # noqa: E402
from smartsom.config.experiment import prepare  # noqa: E402
from smartsom.experiments.composable import allocate  # noqa: E402
from smartsom.experiments.online_concurrency import OnlineConcurrency  # noqa: E402
from smartsom.experiments.tuning_broker import AdaptiveBroker  # noqa: E402
from smartsom.experiments.tuning_ray import (  # noqa: E402
    SmartSOMTrainable,
    build_tuner,
)
from smartsom.experiments.tuning_session import verify_commit  # noqa: E402

pytestmark = pytest.mark.learning
ROOT = Path(__file__).resolve().parents[2]


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class ObservedOnline(OnlineConcurrency):
    """Record qualification evidence without changing the decision algorithm."""

    def __init__(self, *, controlled):
        super().__init__(2)
        self.controlled = controlled
        self.calls = 0
        self.windows = []
        self.decisions = []

    def observe(self, members, identity, *, ticks, updates, now):
        real_now = now
        self.calls += 1
        if self.controlled:
            # Actual fixed-size committed increments with a deterministic test
            # denominator produce a plateau at two. This is NOT measured speed.
            now = self.calls * 15.0
        baseline = dict(self._baseline or {})
        started, previous = self._started, self._latest_window
        result = super().observe(
            members, identity, ticks=ticks, updates=updates, now=now
        )
        if self._latest_window is not previous:
            deltas = {
                member: {
                    "physical_ticks": self._seen[member][0] - baseline[member][0],
                    "updates": self._seen[member][1] - baseline[member][1],
                }
                for member in members
            }
            self.windows.append(
                {
                    **self._latest_window,
                    "members": list(members),
                    "member_deltas": deltas,
                    "controller_started": started,
                    "controller_finished": now,
                    "observed_real_monotonic": real_now,
                }
            )
        if result:
            self.decisions.append({**result, "qualified_windows": self.windows[-2:]})
        return result


class ControlledCPUInput:
    """Test-only steady external CPU input; retain real host/RAM/ownership facts."""

    def __init__(self, monitor):
        self.monitor = monitor

    def snapshot(self, exclude_pids=()):
        raw = self.monitor.snapshot(exclude_pids=exclude_pids)
        return replace(
            raw,
            external_cpu_load=0.0,
            observations={
                **raw.observations,
                "actual_external_percent": raw.external_cpu_load,
                "actual_sample_cpu_percent": raw.cpu_load,
                "controlled_external_percent": 0.0,
            },
            origins={**raw.origins, "external_cpu_load": "controlled fixture input"},
        )


class ObservedBroker(AdaptiveBroker):
    """Observe the production release gate; never override its answer."""

    def __init__(self, entries, root, *, controlled):
        profiles = {
            row["experiment_id"]: [
                {
                    "CPU": 1,
                    "GPU": 0,
                    "memory": 1,
                    "concurrency": 2,
                    "throughput": 0,
                    "update_seconds": 1,
                    "restart_seconds": 1,
                }
            ]
            for row in entries
        }
        self.controlled = controlled
        super().__init__(
            entries,
            profiles,
            mode="performance",
            global_limit=2,
            uncalibrated_groups=("same-task",),
            online_context={
                "same-task": {"hint": None, "hardware": "fixture", "shape": "fixture"}
            },
            online_root=root,
        )
        self._online["same-task"] = ObservedOnline(controlled=controlled)
        self.parks, self.releases = [], []
        self.max_leases = 0

    def _manager(self):
        manager = super()._manager()
        if self.controlled and not isinstance(manager.monitor, ControlledCPUInput):
            manager.monitor = ControlledCPUInput(manager.monitor)
        return manager

    def defer_resize(self, identity, current_resources, requested_resources):
        assert current_resources == requested_resources
        super().defer_resize(identity, current_resources, requested_resources)
        lease = self._manager().leases[identity]
        assert lease.state == "stopping"
        # A pause request must retain the whole reservation while the actor is
        # still owned. Only refresh's real death observation may release it.
        assert identity in self._retiring
        pressure, capacity, reserved_cpus = self._pressure()
        manager = self._manager()
        reserved_memory = sum(
            math.ceil(item.request.memory * manager.peak_factor)
            for item in manager.leases.values()
        )
        self.parks.append(
            {
                "identity": identity,
                "real_monotonic": time.monotonic(),
                "lease": asdict(lease),
                "actor": self._actors[identity],
                "owner": self._owners[identity],
                "epoch": self._ack[identity]["allocation_epoch"],
                "online_limit": self._online["same-task"].limit,
                # The scheduler acts on the latest already-observed decision.
                # Bind this lease generation to it before Ray begins the pause.
                "online_decision_index": (
                    len(self._online["same-task"].decisions) - 1
                    if self._online["same-task"].decisions
                    else None
                ),
                "resource_pressure": pressure,
                "suspend_reason": self._reasons.get(identity, self._last_reason),
                "cpu_diagnostics": self.admission_diagnostics()["raw_cpu"],
                "pressure_budget": {
                    "reserved_cpus": reserved_cpus,
                    "capacity_cpus": capacity.cpus,
                    "reserved_memory_bytes": reserved_memory,
                    "capacity_memory_bytes": capacity.memory,
                    "external_cpu_load_percent": self._snapshot.external_cpu_load,
                    "sample_cpu_load_percent": self._snapshot.cpu_load,
                    "snapshot_age_seconds": self.clock() - self._last_sample,
                },
                "commit_id": self._ack[identity]["commit_id"],
            }
        )

    def refresh(self, *, force=False):
        before = dict(self._broker.leases) if self._broker else {}
        actors = dict(self._actors)
        children = {key: tuple(value) for key, value in self._children.items()}
        super().refresh(force=force)
        leases = self._manager().leases
        self.max_leases = max(self.max_leases, len(leases))
        assert len(leases) <= 2
        for identity in before.keys() - leases.keys():
            state = self._actor_state(actors[identity]).get("State")
            child_states = [
                self._process_state(*child) for child in children.get(identity, ())
            ]
            assert state == "DEAD" and all(value == "dead" for value in child_states)
            self.releases.append(
                {
                    "identity": identity,
                    "real_monotonic": time.monotonic(),
                    "actor": actors[identity],
                    "actor_state": state,
                    "child_states": child_states,
                    "released_state": before[identity].state,
                }
            )


class PacedNative(SmartSOMTrainable):
    """One return remains one genuine update, including its real checkpoint."""

    def step(self):
        started = time.monotonic()
        result = super().step()
        # Match the already qualified real-clock workload in both cases. The
        # controlled denominator must not turn this into an unpaced CPU-load
        # test. Keeping results flowing also primes live process observations
        # while the second actor starts; no cross-actor startup barrier is used.
        time.sleep(max(0, 2.0 - (time.monotonic() - started)))
        return result

    def load_checkpoint(self, checkpoint_dir):
        marker = verify_commit(checkpoint_dir)
        super().load_checkpoint(checkpoint_dir)
        _assert_native_state_equal(
            self.session.session.state_dict(), _state(checkpoint_dir), exact=True
        )
        proof = {
            "identity": self.config["experiment_id"],
            "pid": os.getpid(),
            "real_monotonic": time.monotonic(),
            "commit_id": marker["commit_id"],
            "updates": marker["updates"],
            "physical_ticks": marker["physical_ticks"],
            "allocation_epoch": self.session.epoch,
            "exact_native_state_equal": True,
            "state_keys": sorted(_state(checkpoint_dir)),
        }
        with (self.session.root / "online-restores.jsonl").open("a") as stream:
            stream.write(json.dumps(proof) + "\n")


class LifecycleObserver(Callback):
    def __init__(self, broker, root, *, controlled):
        self.broker, self.root, self.controlled = broker, root, controlled
        self.results = []
        self.overlap = False
        self.first_overlap = None
        self.stopped_after_evidence = False

    def on_step_begin(self, iteration, trials, **info):
        self.broker.refresh()

    def on_trial_start(self, iteration, trials, trial, **info):
        self.broker.note_actor(trial.config["experiment_id"], trial)

    def on_trial_error(self, iteration, trials, trial, **info):
        self.broker.actor_failed(trial.config["experiment_id"], trial)
        # Abort instead of leaving an unissued preset waiting behind a failed
        # actor. The subprocess wrapper captures the actual worker exception.
        raise AssertionError(f"native online trial failed: {trial.get_error()}")

    def on_trial_result(self, iteration, trials, trial, result, **info):
        identity = trial.config["experiment_id"]
        broker = self.broker
        broker.note_actor(identity, trial)
        marker = verify_commit(result["checkpoint"])
        assert marker["experiment_id"] == identity
        assert marker["commit_id"] == result["commit_id"]
        state = _state(result["checkpoint"])
        broker.observe_formal_update(identity, {**result, **marker})
        if (
            sum(lease.state == "running" for lease in broker._manager().leases.values())
            == 2
        ):
            if not self.overlap:
                self.first_overlap = time.monotonic()
            self.overlap = True
        self.results.append(
            {
                "identity": identity,
                "updates": marker["updates"],
                "physical_ticks": marker["physical_ticks"],
                "commit_id": marker["commit_id"],
                "phase": marker["phase"],
                "allocation_epoch": marker["allocation_epoch"],
                "pid": result["pid"],
                "real_monotonic": time.monotonic(),
                "optimizations": sum(state["optimizations"].values()),
            }
        )
        if (
            not self.controlled
            and identity == "seed-102"
            and marker["updates"] >= 2
            and self.overlap
        ):
            from smartsom.experiments.control import read, write_json

            control_root = self.root / "driver-control"
            owner = read(control_root)
            write_json(
                control_root / "control/stop.json",
                {"id": owner["id"], "requested_at": time.time()},
            )
            self.stopped_after_evidence = True


def _entries(root, controlled):
    entries = []
    parameters = root / "native-ppo-parameters.json"
    _write(
        parameters,
        {
            "schema": "smartsom.algorithm-parameters/v1",
            "parameters": {"batch_size": 2, "n_epochs": 1},
        },
    )
    for seed in (101, 102):
        config = api.load_config(ROOT / "configs/test/runs/train_all_ppo.yaml")
        config.seed = seed
        config.training.total_ticks = 128 if controlled else 400
        config.training.ticks_per_update = 4
        config.training.record_initial = False
        config.training.parameters = str(parameters)
        config.validation.enabled = True
        config.validation.every_updates = 1
        config.validation.replications = 1
        config.evaluation.replications = 1
        config.evaluation.checkpoint = "last"
        config.evaluation.record = False
        config.scenario_overrides["tick_limit"] = 8
        config.runtime.numerical_threads = 1
        config.runtime.sampling_processes = 0
        config.output.root = str(root / "runs")
        config.output.name = f"online-seed-{seed}"
        prepared = prepare(config)
        run_root, record, prepared = allocate(prepared, "training")
        entries.append(
            {
                "experiment_id": f"seed-{seed}",
                "prepared": asdict(prepared),
                "run_dir": str(run_root),
                "record": record,
                "control_spec": {},
                "execution_contract": {"cpu_overhead": 0, "allocation_epoch": 0},
                "calibration_group": "same-task",
                "remaining_updates": config.training.total_ticks
                // config.training.ticks_per_update,
            }
        )
    return entries


def _plateau_lifecycle(report):
    """Correlate one complete plateau cycle, allowing earlier pressure cycles."""
    for decision_index, decision in enumerate(report["decisions"]):
        if not (
            decision["accepted"] is False
            and decision["concurrency"] == 2
            and decision["next_limit"] == 1
        ):
            continue
        decided_at = decision["qualified_windows"][-1]["observed_real_monotonic"]
        for park in sorted(report["parks"], key=lambda row: row["real_monotonic"]):
            if not (
                park["identity"] == "seed-102"
                and park["online_limit"] == 1
                and park["online_decision_index"] == decision_index
                and park["resource_pressure"] is False
                and park["real_monotonic"] >= decided_at
            ):
                continue
            for released in report["releases"]:
                if not (
                    released["identity"] == park["identity"]
                    and released["actor"] == park["actor"]
                    and released["real_monotonic"] >= park["real_monotonic"]
                    and released["released_state"] == "stopping"
                    and released["actor_state"] == "DEAD"
                    and all(state == "dead" for state in released["child_states"])
                ):
                    continue
                for restored in report["restores"]:
                    if (
                        restored["identity"] == park["identity"]
                        and restored["commit_id"] == park["commit_id"]
                        and restored["allocation_epoch"] == park["epoch"]
                        and restored["pid"] != park["owner"][0]
                        and restored["real_monotonic"] >= released["real_monotonic"]
                        and restored["exact_native_state_equal"] is True
                    ):
                        return decision, park, released, restored
    raise AssertionError(
        "no correlated non-pressure plateau park/release/restore cycle"
    )


def _check_evidence(root, controlled, entries, broker, observer):
    controller = broker._online["same-task"]
    report = {
        "timing_regime": "controlled-controller-clock"
        if controlled
        else "paced-real-clock",
        "performance_benchmark": False,
        "resource_sampling": "controlled-external-cpu" if controlled else "real-host",
        "admission_diagnostics": broker.admission_diagnostics(),
        "status": "incomplete",
        "resource_capacity": broker.summary()["resources"],
        "production_window_seconds": controller.window_seconds,
        "cached_hint": controller.hint,
        "scope": {
            "seeds": [101, 102],
            "max_concurrent": 2,
            "learner_threads": 1,
            "sampling_processes": 0,
            "validation_every_updates": 1,
            "minimum_step_wall_seconds": 2.0,
            "sampling_child_lifecycle_covered": False,
        },
        "broker_clock_real": broker.clock is time.monotonic,
        "source": entries[0]["record"]["source"],
        "max_leases": broker.max_leases,
        "two_live_learners_observed": observer.overlap,
        "first_two_live_learners_at": observer.first_overlap,
        "decisions": controller.decisions,
        "commits": observer.results,
        "parks": broker.parks,
        "releases": broker.releases,
        "restores": [],
        "evaluation_summary": [],
        "stopped_after_evidence": observer.stopped_after_evidence,
        "completion_scope": (
            "full native budgets, final evaluation, and park/restore continuity"
            if controlled
            else "scheduler stop after qualified growth and two second-actor commits"
        ),
    }
    for entry in entries:
        restore = Path(entry["run_dir"]) / "online-restores.jsonl"
        if restore.exists():
            report["restores"].extend(
                json.loads(row) for row in restore.read_text().splitlines()
            )
        evaluation = Path(entry["run_dir"]) / "evaluation/tuning-final.json"
        if evaluation.exists():
            rows = json.loads(evaluation.read_text())
            report["evaluation_summary"].append(
                {
                    "identity": entry["experiment_id"],
                    "path": str(evaluation),
                    "case_count": len(rows),
                    "engineering_failures": sum(
                        bool(row["engineering_failure"]) for row in rows
                    ),
                    "seeds": [row["seed"] for row in rows],
                }
            )
    filename = (
        "online-controlled-clock-report.json"
        if controlled
        else "online-real-clock-report.json"
    )
    _write(root / filename, report)
    assert (root / filename).stat().st_size <= 256 * 1024
    assert report["broker_clock_real"] and report["max_leases"] == 2
    assert controller.hint == 1 and controller.window_seconds == 30
    assert observer.overlap
    growth = next(row for row in controller.decisions if row["next_limit"] == 2)
    assert growth["concurrency"] == 1 and growth["accepted"]
    assert growth["window_min_seconds"] == 30
    assert len(growth["qualified_windows"]) == 2
    for window in growth["qualified_windows"]:
        assert window["wall_seconds"] >= 30
        assert all(row["updates"] >= 2 for row in window["member_deltas"].values())
        assert (
            sum(row["physical_ticks"] for row in window["member_deltas"].values())
            == window["physical_ticks"]
        )
    for identity in ("seed-101", "seed-102"):
        commits = [row for row in observer.results if row["identity"] == identity]
        assert len(commits) >= 2 and any(row["optimizations"] > 0 for row in commits)
        # Every native update appears exactly once, including across a restart.
        assert [row["updates"] for row in commits] == list(range(1, len(commits) + 1))
        assert len({row["commit_id"] for row in commits}) == len(commits)
    if not controlled:
        assert observer.stopped_after_evidence
        assert not broker.parks, "real-clock case stops before a plateau comparison"
        report["status"] = "passed"
        _write(root / filename, report)
        return
    assert {row["identity"] for row in report["evaluation_summary"]} == {
        "seed-101",
        "seed-102",
    }
    assert all(
        row["case_count"] == 1 and row["engineering_failures"] == 0
        for row in report["evaluation_summary"]
    )
    rollback, park, released, restored = _plateau_lifecycle(report)
    assert rollback["concurrency"] == 2 and rollback["next_limit"] == 1
    assert (
        observer.first_overlap
        <= rollback["qualified_windows"][-1]["observed_real_monotonic"]
    )
    assert (
        park["real_monotonic"]
        <= released["real_monotonic"]
        <= restored["real_monotonic"]
    )
    assert park["online_limit"] == 1 and not park["resource_pressure"]
    assert released["released_state"] == "stopping"
    assert park["commit_id"] == restored["commit_id"]
    assert park["epoch"] == restored["allocation_epoch"]
    assert park["owner"][0] != restored["pid"]
    assert restored["exact_native_state_equal"]
    assert {
        "learners",
        "policies",
        "collector",
        "random",
        "torch_random",
        "numpy_random",
        "python_random",
    } <= set(restored["state_keys"])
    assert any(
        row["identity"] == "seed-102"
        and row["updates"] > restored["updates"]
        and row["pid"] == restored["pid"]
        and row["real_monotonic"] >= restored["real_monotonic"]
        for row in observer.results
    )
    assert all(
        next(row for row in reversed(observer.results) if row["identity"] == identity)[
            "phase"
        ]
        == "experiment_complete"
        for identity in ("seed-101", "seed-102")
    )

    report["plateau_lifecycle"] = {
        "online_decision_index": park["online_decision_index"],
        "decision_at": rollback["qualified_windows"][-1]["observed_real_monotonic"],
        "park_at": park["real_monotonic"],
        "release_at": released["real_monotonic"],
        "restore_at": restored["real_monotonic"],
        "actor": park["actor"],
        "commit_id": park["commit_id"],
        "allocation_epoch": park["epoch"],
        "parked_pid": park["owner"][0],
        "restored_pid": restored["pid"],
        "restored_updates": restored["updates"],
    }

    report["status"] = "passed"
    _write(root / filename, report)


def _driver(root, controlled):
    """Called only in an isolated child process, never during collection."""
    from smartsom.experiments.control import CURRENT, Scope

    root = Path(root)
    assert not ray.is_initialized()
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    entries = _entries(root, controlled)
    broker = ObservedBroker(entries, root, controlled=controlled)
    observer = LifecycleObserver(broker, root, controlled=controlled)
    scope = Scope()
    token = CURRENT.set(scope)
    # The fixture stops the Tune scheduler at result boundaries. A sibling
    # control root does not interrupt an actor's in-flight native validation.
    control_root = root / "driver-control"
    control_root.mkdir()
    scope.bind(control_root)
    # Short UNIX paths matter on macOS. This child owns this fresh runtime only.
    with tempfile.TemporaryDirectory(prefix="som-online-", dir="/tmp") as ray_root:
        try:
            ray.init(
                address="local",
                num_cpus=2,
                include_dashboard=False,
                log_to_driver=False,
                object_store_memory=80 * 1024**2,
                _temp_dir=ray_root,
                runtime_env={"py_executable": sys.executable},
            )
            grid = build_tuner(
                entries,
                selected={
                    row["experiment_id"]: {"CPU": 1, "GPU": 0, "memory": 1}
                    for row in entries
                },
                broker=broker,
                storage_path=root / "ray-storage",
                name="controlled-clock" if controlled else "real-clock",
                device="cpu",
                allocation_function=broker.allocation,
                callbacks=[observer],
                trainable=PacedNative,
            ).fit()
            assert len(grid) == 2 and not grid.errors
            _check_evidence(root, controlled, entries, broker, observer)
        finally:
            ray.shutdown()
            scope.finish("stopped" if observer.stopped_after_evidence else "completed")
            CURRENT.reset(token)


def _live(identity):
    try:
        process = psutil.Process(identity[0])
        return (
            process.create_time() == identity[1]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _run_owned(tmp_path, *, controlled):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(Path(__file__).parent))
    )
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    owned = set()
    normal_remaining = []
    with (tmp_path / "probe.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from test_tuning_ray_online_lifecycle import _driver; "
                "import sys; _driver(sys.argv[1], sys.argv[2] == 'controlled')",
                str(tmp_path),
                "controlled" if controlled else "real",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        parent = psutil.Process(process.pid)
        owned.add((parent.pid, parent.create_time()))
        deadline = time.monotonic() + (180 if controlled else 240)
        timed_out = False
        try:
            while process.poll() is None:
                try:
                    owned.update(
                        (child.pid, child.create_time())
                        for child in parent.children(recursive=True)
                    )
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.2)
            # A normal child calls ray.shutdown. Check natural cleanup before
            # emergency cleanup, rather than allowing cleanup to hide a leak.
            until = time.monotonic() + 5
            while (
                not timed_out
                and any(_live(item) for item in owned)
                and time.monotonic() < until
            ):
                time.sleep(0.1)
            normal_remaining = [item for item in owned if _live(item)]
        finally:
            survivors = []
            for pid, created in owned:
                try:
                    child = psutil.Process(pid)
                    if (
                        child.create_time() == created
                        and child.status() != psutil.STATUS_ZOMBIE
                    ):
                        child.terminate()
                        survivors.append(child)
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(survivors, timeout=3)
            killed = []
            for child in alive:
                try:
                    if (child.pid, child.create_time()) in owned:
                        child.kill()
                        killed.append(child)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(killed, timeout=3)
            process.wait(timeout=10)
    log = (tmp_path / "probe.log").read_text()
    assert not timed_out, f"owned online driver exceeded deadline\n{log[-12000:]}"
    assert process.returncode == 0, log[-12000:]
    assert not normal_remaining, (
        f"owned processes survived normal shutdown: {normal_remaining}"
    )


def test_real_clock_online_admits_second_native_learner(tmp_path):
    _run_owned(tmp_path, controlled=False)


def test_controlled_clock_online_parks_and_restores_native_learner(tmp_path):
    _run_owned(tmp_path, controlled=True)


def _correlation_fixture():
    """Synthetic report rows test matching only; no actors or timing claims."""
    report = {
        "decisions": [
            {
                "accepted": False,
                "concurrency": 2,
                "next_limit": 1,
                "qualified_windows": [{"observed_real_monotonic": 30}],
            }
        ],
        "parks": [],
        "releases": [],
        "restores": [],
    }
    for index, (at, pressure) in enumerate(((10, True), (20, True), (40, False))):
        identity, actor, commit = "seed-102", f"actor-{index}", f"commit-{index}"
        report["parks"].append(
            {
                "identity": identity,
                "actor": actor,
                "commit_id": commit,
                "online_limit": 2 if pressure else 1,
                "online_decision_index": None if pressure else 0,
                "resource_pressure": pressure,
                "real_monotonic": at,
                "epoch": 1,
                "owner": [index + 10, 1],
            }
        )
        report["releases"].append(
            {
                "identity": identity,
                "actor": actor,
                "real_monotonic": at + 1,
                "released_state": "stopping",
                "actor_state": "DEAD",
                "child_states": [],
            }
        )
        report["restores"].append(
            {
                "identity": identity,
                "commit_id": commit,
                "allocation_epoch": 1,
                "pid": index + 100,
                "real_monotonic": at + 2,
                "exact_native_state_equal": True,
            }
        )
    return report


def test_plateau_evidence_ignores_earlier_pressure_cycles():
    report = _correlation_fixture()
    decision, park, released, restored = _plateau_lifecycle(report)
    assert decision is report["decisions"][0]
    assert park is report["parks"][-1]
    assert released is report["releases"][-1]
    assert restored is report["restores"][-1]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("decisions", "accepted", True),
        ("parks", "resource_pressure", True),
        ("parks", "online_limit", 2),
        ("parks", "online_decision_index", 1),
        ("parks", "real_monotonic", 29),
        ("releases", "actor", "another-actor"),
        ("releases", "real_monotonic", 39),
        ("releases", "actor_state", "ALIVE"),
        ("releases", "child_states", ["live"]),
        ("restores", "commit_id", "another-commit"),
        ("restores", "allocation_epoch", 2),
        ("restores", "pid", 12),
        ("restores", "real_monotonic", 40.5),
        ("restores", "exact_native_state_equal", False),
    ],
)
def test_plateau_evidence_requires_matching_lifecycle(section, key, value):
    report = _correlation_fixture()
    report[section][-1][key] = value
    with pytest.raises(AssertionError, match="no correlated non-pressure plateau"):
        _plateau_lifecycle(report)


@pytest.mark.parametrize("controlled", [False, True])
def test_controlled_cpu_fixture_isolates_pressure_without_changing_host_facts(
    tmp_path, controlled
):
    from types import SimpleNamespace

    from smartsom.experiments.tuning_resources import (
        GIB,
        ProcessUsage,
        ResourceLease,
        ResourceRequest,
        ResourceSnapshot,
    )

    raw = ResourceSnapshot(
        4,
        16 * GIB,
        12 * GIB,
        external_cpu_load=27,
        cpu_load=40,
        processes=(ProcessUsage(101, "actor", 1, GIB, True),),
        observations={"unprimed_process_count": 1, "excluded_cpu_cores": 1},
        unavailable=("CPU load first sample unavailable",),
    )
    exclusions = []

    def snapshot(exclude_pids):
        exclusions.append(exclude_pids)
        return raw

    broker = ObservedBroker(
        [{"experiment_id": "a"}, {"experiment_id": "b"}],
        tmp_path,
        controlled=controlled,
    )
    manager = broker._manager()
    monitor = SimpleNamespace(snapshot=snapshot)
    manager.monitor = monitor
    manager = broker._manager()
    for name in ("a", "b"):
        manager.leases[name] = ResourceLease(name, ResourceRequest(1, 1), "running")
    broker._snapshot = manager.snapshot(exclude_pids=(101,))
    assert exclusions == [(101,)]
    assert broker._pressure()[0] is (not controlled)
    observed = broker._snapshot
    assert observed.cpus == raw.cpus and observed.memory_total == raw.memory_total
    assert (
        observed.memory_available == raw.memory_available
        and observed.processes == raw.processes
    )
    assert observed.unavailable == raw.unavailable
    if controlled:
        assert observed.external_cpu_load == 0
        assert observed.observations["actual_external_percent"] == 27
        assert observed.observations["actual_sample_cpu_percent"] == 40
        assert observed.observations["unprimed_process_count"] == 1
        assert observed.observations["excluded_cpu_cores"] == 1
        assert manager.monitor.monitor is monitor
        # The fixture isolates CPU only: real RAM pressure still requires parking.
        broker._snapshot = replace(observed, memory_available=0, processes=())
        assert broker._pressure()[0] is True
    else:
        assert observed is raw and manager.monitor is monitor


def test_controlled_cpu_input_survives_broker_monitor_reconstruction(tmp_path):
    import pickle

    broker = ObservedBroker([{"experiment_id": "a"}], tmp_path, controlled=True)
    first = broker._manager()
    assert isinstance(first.monitor, ControlledCPUInput)
    assert broker._manager().monitor is first.monitor
    restored = pickle.loads(pickle.dumps(broker))
    assert restored._broker is None
    assert isinstance(restored._manager().monitor, ControlledCPUInput)
