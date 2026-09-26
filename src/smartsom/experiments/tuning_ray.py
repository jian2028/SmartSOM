"""Optional Ray 2.58 execution adapter for already calibrated experiments.

The driver owns Ray initialization, resource observation and durable batch state.
This module owns admission before Tune creates PENDING trials, update-boundary
resource changes, and the Class Trainable/native-session bridge. CPU resources
are logical reservations, not operating-system quotas.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Experiment, Trial
from ray.tune.schedulers import FIFOScheduler, ResourceChangingScheduler
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.search import BasicVariantGenerator, SearchAlgorithm

RAY_CONTRACT_VERSION = "2.58.0"


class TuneBroker(Protocol):
    """Driver-local leases include actors still starting or stopping.

    try_acquire must be idempotent. For an existing reservation undergoing a
    resize, it returns False until the old actor and its children have exited.
    Scheduler callbacks never release leases merely because a trial is PAUSED
    or TERMINATED; the observer confirms process exit first.
    """

    def try_acquire(self, experiment_id: str, resources: dict[str, float]) -> bool: ...

    def defer_resize(
        self,
        experiment_id: str,
        current_resources: dict[str, float],
        requested_resources: dict[str, float],
    ) -> None: ...

    def release(self, experiment_id: str) -> None: ...

    def refresh(self) -> None: ...


def require_ray_contract() -> None:
    """The beta scheduler overrides below are audited for this exact version."""
    if ray.__version__ != RAY_CONTRACT_VERSION:
        raise RuntimeError(
            f"SmartSOM Tune requires Ray {RAY_CONTRACT_VERSION}; "
            f"found {ray.__version__}. Revalidate the scheduler lifecycle first."
        )


def resource_dict(resources: Any) -> dict[str, float]:
    """Validate the single-node, one-GPU-per-trial execution contract."""
    raw = getattr(resources, "required_resources", resources)
    if not isinstance(raw, Mapping):
        raise ValueError("trial resources must be a resource mapping")
    if set(raw) - {"CPU", "GPU", "memory"}:
        raise ValueError("only CPU, GPU and memory trial resources are supported")
    values = {key: float(value) for key, value in raw.items()}
    cpu, gpu = values.get("CPU", 0), values.get("GPU", 0)
    if not math.isfinite(cpu) or cpu < 1 or not cpu.is_integer():
        raise ValueError("CPU reservation must be a positive integer")
    if gpu not in (0, 1):
        raise ValueError("GPU reservation must be zero or one")
    memory = values.get("memory")
    if memory is not None and (not math.isfinite(memory) or memory <= 0):
        raise ValueError("memory reservation must be positive and finite")
    return {"CPU": cpu, "GPU": gpu, **({"memory": memory} if memory else {})}


def _experiment_id(config: Mapping) -> str:
    value = config.get("experiment_id")
    if not isinstance(value, str) or not value:
        raise ValueError("each preset requires a stable experiment_id")
    return value


class PresetSearchAlgorithm(SearchAlgorithm):
    """Issue only frozen presets with an acquired lease, never blocked PENDINGs.

    BasicVariantGenerator supplies Ray's ordinary experiment-to-Trial conversion.
    There is no search over scientific parameters and no Tune restore: native
    continuation is submitted as a preset in a new execution segment.
    """

    def __init__(self, entries: Sequence[dict], selected: Mapping, broker: TuneBroker):
        require_ray_contract()
        self.entries = copy.deepcopy(list(entries))
        identities = [_experiment_id(entry) for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate experiment_id in a Tune segment")
        self.selected = {key: resource_dict(selected[key]) for key in identities}
        self.broker = broker
        self._issued: dict[int, str] = {}
        self._experiment = None
        self._generator = BasicVariantGenerator()
        self._finished = not self.entries
        self._metric = None

    @property
    def total_samples(self):
        return len(self.entries)

    def add_configurations(self, experiments):
        if self._experiment is not None:
            raise ValueError("a preset search algorithm belongs to one segment")
        items = [experiments] if isinstance(experiments, Experiment) else experiments
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("provide exactly one Ray Experiment per device cohort")
        self._experiment = items[0]
        if not isinstance(self._experiment, Experiment):
            raise ValueError("expected a Ray Experiment")
        experiment = copy.copy(self._experiment)
        experiment.spec = copy.deepcopy(self._experiment.spec)
        experiment.spec.update(config={}, num_samples=len(self.entries))
        self._generator.add_configurations(experiment)

    def next_trial(self):
        if self._experiment is None:
            raise RuntimeError("add_configurations must precede trial admission")
        if self._finished:
            return None
        # Scan unissued presets: a large waiting experiment need not block a
        # smaller one for which the observer has a safe measured allocation.
        for index, entry in enumerate(self.entries):
            if index in self._issued:
                continue
            identity = _experiment_id(entry)
            if not self.broker.try_acquire(identity, self.selected[identity]):
                continue
            try:
                # One generator supplies unique IDs for the entire segment.
                # The frozen config is installed before controller.add_trial,
                # placement-group creation, callbacks or actor staging.
                trial = self._generator.next_trial()
                if trial is None:
                    raise RuntimeError("Ray did not create an admitted preset trial")
                trial.config = copy.deepcopy(entry)
                trial.evaluated_params = {"experiment_id": identity}
            except BaseException:
                # No actor has been staged yet, so this reservation can be freed.
                self.broker.release(identity)
                raise
            self._issued[index] = trial.trial_id
            self._finished = len(self._issued) == len(self.entries)
            return trial
        # A resource wait is not exhaustion and has no automatic deadline.
        return None

    def on_trial_complete(self, trial_id, result=None, error=False):
        self._generator.on_trial_complete(trial_id, result=result, error=error)
        # Lease release belongs to the observer, after actual actor exit.

    def save_to_dir(self, dirpath, session_str="state", **kwargs):
        target = Path(dirpath) / f"preset-search-{session_str}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"issued": self._issued, "finished": self._finished}) + "\n"
        )
        temporary.replace(target)

    def restore_from_dir(self, dirpath):
        raise RuntimeError("resume SmartSOM native checkpoints in a new Tune segment")


class _EligibleController:
    def __init__(self, controller, excluded):
        self.controller, self.excluded = controller, excluded

    def get_trials(self):
        return [
            trial
            for trial in self.controller.get_trials()
            if trial.trial_id not in self.excluded
        ]

    def __getattr__(self, name):
        return getattr(self.controller, name)


def _no_allocation(controller, trial, result, scheduler):
    return None


class SafeResourceChangingScheduler(ResourceChangingScheduler):
    """Version-locked RCS with complete resize processing and broker admission.

    Ray 2.58 uses ``changed = changed or set_trial_resources(...)`` in a loop,
    which skips later calls after the first change. This override applies each
    entry. Initial resize occurs in on_trial_add before Tune can stage actors;
    choose_trial_to_run alone cannot gate Tune's eager PENDING actor staging.
    """

    def __init__(
        self, *, selected, broker, allocation_function=None, base_scheduler=None
    ):
        require_ray_contract()
        super().__init__(
            base_scheduler=base_scheduler or FIFOScheduler(),
            resources_allocation_function=allocation_function or _no_allocation,
        )
        self.selected = {key: resource_dict(value) for key, value in selected.items()}
        self.broker = broker

    def _assign(self, trial, resources):
        values = resource_dict(resources)
        before = resource_dict(trial.placement_group_factory)
        if before["GPU"] != values["GPU"]:
            raise ValueError("a trial cannot change its frozen CPU/CUDA device cohort")
        changed = (
            self.set_trial_resources(trial, PlacementGroupFactory([values]))
            if before != values
            else False
        )
        if changed:
            contract = trial.config.setdefault("execution_contract", {})
            contract["allocation_epoch"] = int(contract.get("allocation_epoch", 0)) + 1
        return changed

    def on_trial_add(self, tune_controller, trial, **kwargs):
        # Validate every true initial base before replacing it with calibration.
        super().on_trial_add(tune_controller, trial, **kwargs)
        base = resource_dict(trial.placement_group_factory)
        if base["CPU"] != 1 or "memory" in base:
            raise ValueError(
                "initial cohort base must have CPU=1 and no memory override"
            )
        identity = _experiment_id(trial.config)
        self._assign(trial, self.selected[identity])

    def on_trial_result(self, tune_controller, trial, result):
        decision = self._base_scheduler.on_trial_result(tune_controller, trial, result)
        if decision != TrialScheduler.CONTINUE:
            return decision
        identity = _experiment_id(trial.config)
        current = resource_dict(trial.placement_group_factory)
        # The measured allocation callback does not optimize a learning metric.
        # Call it directly: stock RCS mutates callback.metric/mode, which also
        # rejects otherwise valid bound methods and slot-based callables.
        requested = self._resources_allocation_function(
            tune_controller, trial, result, self
        )
        if requested is not None:
            requested = resource_dict(requested)
            if requested["GPU"] != current["GPU"]:
                raise ValueError("resource changes cannot switch device cohorts")
            if requested != current:
                self.broker.defer_resize(identity, current, requested)
                self._trials_to_reallocate[trial] = requested
                return TrialScheduler.PAUSE
        if getattr(self.broker, "should_pause", lambda identity: False)(identity):
            self.broker.defer_resize(identity, current, current)
            return TrialScheduler.PAUSE
        return TrialScheduler.CONTINUE

    def choose_trial_to_run(self, tune_controller, **kwargs):
        if getattr(tune_controller, "_reuse_actors", False):
            raise ValueError("SmartSOM RCS requires reuse_actors=False")
        waiting = {}
        for trial, requested in self._trials_to_reallocate.items():
            if trial.status == Trial.RUNNING or not self.broker.try_acquire(
                _experiment_id(trial.config), resource_dict(requested)
            ):
                waiting[trial] = requested
                continue
            # Do not short-circuit: every granted resize must be applied.
            self._assign(trial, requested)
        self._trials_to_reallocate = waiting
        excluded = {trial.trial_id for trial in waiting}
        while True:
            trial = self._base_scheduler.choose_trial_to_run(
                _EligibleController(tune_controller, excluded), **kwargs
            )
            if trial is None or trial.status != Trial.PAUSED:
                return trial
            if self.broker.try_acquire(
                _experiment_id(trial.config),
                resource_dict(trial.placement_group_factory),
            ):
                return trial
            excluded.add(trial.trial_id)


def _set_threads(threads: int):
    """Apply both newly loaded and already initialized numerical thread pools."""
    if type(threads) is not int or threads < 1:
        raise ValueError("numerical threads must be a positive integer")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(threads)
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError(
                "Torch inter-op threads were initialized with a value other than 1"
            ) from exc
    limits = threadpool_limits(limits=threads)
    if torch.get_num_threads() != threads or torch.get_num_interop_threads() != 1:
        limits.restore_original_limits()
        raise RuntimeError("Torch did not apply the requested numerical thread limits")
    mismatched = [
        pool.get("prefix", pool.get("internal_api", "unknown"))
        for pool in threadpool_info()
        if pool.get("user_api") in {"blas", "openmp"}
        and pool.get("num_threads") != threads
    ]
    if mismatched:
        limits.restore_original_limits()
        raise RuntimeError(f"numerical thread limits were not applied: {mismatched}")
    return limits


def _session_factory(prepared, root, record, *, allocation_epoch, threads):
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.composable import implementation_identity
    from smartsom.experiments.evidence import source_identity
    from smartsom.experiments.tuning_session import AdaptiveSession

    live, expected = source_identity(), record["source"]
    if implementation_identity() != record["implementation_sha256"] or any(
        live[key] != expected[key] for key in ("python", "platform", "packages")
    ):
        raise ValueError("worker source/dependency identity differs from frozen batch")
    # Ray may execute an exact package closure without .git. Its Python bytes
    # and dependencies still have to match; the driver checks Git before launch.
    if live["git"]["commit"] is not None and live["git"] != expected["git"]:
        raise ValueError("worker checkout identity differs from frozen batch")
    return AdaptiveSession(
        PreparedComposition(**prepared),
        root,
        record,
        allocation_epoch=allocation_epoch,
        threads=threads,
    )


class SmartSOMTrainable(Trainable):
    """One step is one committed native update; Tune iterations are execution-only."""

    def setup(self, config):
        self.session = None
        self._thread_limits = None
        self._model_thread_limits = None
        resources = resource_dict(self.trial_resources)
        contract = copy.deepcopy(config.get("execution_contract", {}))
        overhead = contract.get("cpu_overhead", 0)
        if type(overhead) is not int or overhead < 0:
            raise ValueError("cpu_overhead must be a nonnegative fixed integer")
        threads = int(resources["CPU"]) - overhead
        if threads < 1:
            raise ValueError(
                "CPU allocation must include at least one numerical thread"
            )
        frozen = json.loads(config["prepared"]["config_json"])
        device = frozen.get("runtime", {}).get("device", "cpu")
        if device not in ("cpu", "cuda") or resources["GPU"] != int(device == "cuda"):
            raise ValueError(
                "allocated GPU count differs from the frozen device cohort"
            )
        root = Path(config["run_dir"]).resolve()
        manifest = root / "run.json"
        record = (
            json.loads(manifest.read_text())
            if manifest.is_file()
            else copy.deepcopy(config.get("record"))
        )
        if not isinstance(record, dict):
            raise ValueError(
                "the batch driver must supply an allocated native run record"
            )
        record["tuning"] = {
            "experiment_id": _experiment_id(config),
            "execution_contract": contract,
            "control_spec": copy.deepcopy(config.get("control_spec")),
            "resources": resources,
            "numerical_threads": threads,
            "ray_trial_id": self.trial_id,
            "continuation": config.get("continuation"),
        }
        self._thread_limits = _set_threads(threads)
        try:
            self.session = _session_factory(
                config["prepared"],
                root,
                record,
                allocation_epoch=int(contract.get("allocation_epoch", 0)),
                threads=threads,
            )
            # Model construction may import additional numerical libraries.
            self._model_thread_limits = _set_threads(threads)
            if config.get("continuation"):
                self.load_checkpoint(config["continuation"])
        except BaseException:
            self.cleanup()
            raise

    def step(self):
        result = dict(self.session.step())
        if "done" not in result:
            result["done"] = result.get("status") in {"completed", "early_stopped"}
        result["experiment_id"] = _experiment_id(self.config)
        return result

    def save_checkpoint(self, checkpoint_dir):
        self.session.save_checkpoint(Path(checkpoint_dir))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        # setup has already applied the current allocation. The native loader
        # restores training state only, never historical thread settings.
        self.session.load_checkpoint(Path(checkpoint_dir))

    def cleanup(self):
        try:
            if self.session is not None:
                self.session.close()
                self.session = None
        finally:
            if self._model_thread_limits is not None:
                self._model_thread_limits.restore_original_limits()
                self._model_thread_limits = None
            if self._thread_limits is not None:
                self._thread_limits.restore_original_limits()
                self._thread_limits = None


def build_tuner(
    entries,
    *,
    selected,
    broker,
    storage_path,
    name,
    device,
    allocation_function=None,
    callbacks=(),
    trainable=SmartSOMTrainable,
):
    """Build a fresh segment inside the driver's existing local Ray runtime."""
    require_ray_contract()
    if not ray.is_initialized():
        raise RuntimeError("the batch driver must initialize its Ray runtime first")
    if device not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA device cohorts are supported")
    if not entries:
        raise ValueError("a Tune segment requires at least one experiment")
    selected = {key: resource_dict(value) for key, value in selected.items()}
    gpu = int(device == "cuda")
    if any(value["GPU"] != gpu for value in selected.values()):
        raise ValueError("all allocations in a cohort must have the same GPU count")
    os.environ["TUNE_RESULT_BUFFER_LENGTH"] = "1"
    return tune.Tuner(
        tune.with_resources(trainable, {"CPU": 1, "GPU": gpu}),
        param_space={},
        tune_config=tune.TuneConfig(
            num_samples=len(entries),
            search_alg=PresetSearchAlgorithm(entries, selected, broker),
            scheduler=SafeResourceChangingScheduler(
                selected=selected,
                broker=broker,
                allocation_function=allocation_function,
            ),
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=name,
            storage_path=str(Path(storage_path).resolve()),
            verbose=0,
            callbacks=list(callbacks),
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
        ),
    )
