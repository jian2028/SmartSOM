"""Native update boundaries and portable, verified adaptive continuation.

Only numerical threads and batch concurrency have a versioned identity waiver.
Native strict resume stays unchanged. Wrapped files contain real learner,
optimizer, sampler, RNG and replay state, never weights-only initialization.
"""

import copy
import hashlib
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import canonical_json, digest
from smartsom.experiments.evidence import write_json

CONTRACT = "smartsom.adaptive-continuation/v1"
EXECUTION_FIELDS = frozenset({"numerical_threads", "max_concurrent"})
NATIVE_MEMBERS = ("groups", "controllers", "snapshot.json", "continuation.pkl")


def scientific_identity(prepared):
    """Keep every frozen input; waive two runtime fields and the derived hash."""
    data = asdict(prepared)
    config = json.loads(data.pop("config_json"))
    for key in EXECUTION_FIELDS:
        config.get("runtime", {}).pop(key, None)
    data.pop("scientific_sha256", None)
    return digest({"contract": CONTRACT, "config": config, **data})


def effective_prepared(prepared, threads):
    if type(threads) is not int or threads < 1:
        raise ValueError("adaptive numerical threads must be a positive integer")
    config = json.loads(prepared.config_json)
    config.setdefault("runtime", {})["numerical_threads"] = threads
    return replace(
        prepared,
        config_json=canonical_json(config),
        scientific_sha256=scientific_identity(prepared),
    )


def _source_identity(record):
    if "source" not in record or "implementation_sha256" not in record:
        raise ValueError("adaptive continuation requires original source identity")
    return digest({key: record[key] for key in ("source", "implementation_sha256")})


def _checkpoint_directory(directory):
    directory = Path(directory)
    if (
        not (directory / "commit.json").is_file()
        and (directory / "native/commit.json").is_file()
    ):
        directory = directory / "native"
    if directory.is_symlink():
        raise ValueError("adaptive checkpoint root cannot be a symlink")
    return directory


def file_digests(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint contains a symlink")
        relative = str(path.relative_to(directory))
        if path.is_file() and relative != "commit.json":
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_commit(directory):
    directory = _checkpoint_directory(directory)
    marker = json.loads((directory / "commit.json").read_text())
    actual = file_digests(directory)
    if marker.get("schema") != CONTRACT or marker.get("files") != actual:
        raise ValueError("adaptive checkpoint checksum or contract changed")
    if marker.get("checkpoint_digest") != digest(actual):
        raise ValueError("adaptive checkpoint aggregate digest changed")
    identity = {key: value for key, value in marker.items() if key != "commit_id"}
    if marker.get("commit_id") != digest(identity):
        raise ValueError("adaptive checkpoint commit identity changed")
    return marker


def verify_identity(prepared, record, checkpoint_or_marker):
    """Shared driver/actor gate before learner construction or state unpickling."""
    marker = (
        checkpoint_or_marker
        if isinstance(checkpoint_or_marker, dict)
        else verify_commit(checkpoint_or_marker)
    )
    if marker.get("schema") != CONTRACT:
        raise ValueError("adaptive continuation contract changed")
    if marker.get("commit_id") != digest(
        {key: value for key, value in marker.items() if key != "commit_id"}
    ):
        raise ValueError("adaptive checkpoint commit identity changed")
    if marker.get("scientific_sha256") != scientific_identity(prepared):
        raise ValueError("adaptive checkpoint scientific identity changed")
    if marker.get("experiment_id") != record.get("tuning", {}).get("experiment_id"):
        raise ValueError("adaptive checkpoint experiment identity changed")
    if marker.get("source_identity_sha256") != _source_identity(record):
        raise ValueError("adaptive checkpoint source or dependency identity changed")
    return marker


def _copy(source, target):
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise ValueError("checkpoint dependency cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        file_digests(source)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copyfile(source, target)


def _copy_native(source, target):
    """Copy native payload only, without recursively nesting support trees."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in NATIVE_MEMBERS:
        member = source / name
        if member.exists():
            _copy(member, target / name)
    if not all(
        (target / name).is_file() for name in ("snapshot.json", "continuation.pkl")
    ):
        raise ValueError("native checkpoint lacks snapshot or complete continuation")


def _fsync_directory(directory):
    # Windows has no supported POSIX directory-fsync path here. File contents
    # are still synced; the commit marker reports the namespace durability limit.
    if os.name == "nt":
        return False
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


class AdaptiveSession:
    def __init__(
        self,
        prepared,
        root,
        record,
        *,
        allocation_epoch=0,
        threads=1,
        training_session_factory=None,
        finalizer=None,
    ):
        self.root = Path(root).resolve()
        self.record = copy.deepcopy(record)
        self.original = prepared
        self.identity = scientific_identity(prepared)
        self.source_identity = _source_identity(record)
        self.epoch, self.threads = allocation_epoch, threads
        self.experiment_id = self.record["tuning"]["experiment_id"]
        self.token, self.session = None, None
        self.last_commit, self.final_done = None, False
        self._training_finished, self._last_key = False, None
        self._native_failed = False
        self.finalizer = finalizer
        for name in (
            "config",
            "checkpoints",
            "evidence",
            "evaluation",
            "logs",
            "reports",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        continuation = self.record["tuning"].get("continuation")
        marker = (
            verify_identity(prepared, record, continuation) if continuation else None
        )
        if continuation:
            self._restore_files(_checkpoint_directory(continuation), marker)
        original_file = self.root / "config/original-prepared.json"
        if original_file.exists():
            frozen = replace(prepared, **json.loads(original_file.read_text()))
            if scientific_identity(frozen) != self.identity:
                raise ValueError("original frozen preparation changed")
            self.original = frozen
        else:
            write_json(original_file, asdict(prepared))
        self.prepared = self._portable_prepared(
            effective_prepared(self.original, threads)
        )
        if training_session_factory is None:
            from smartsom.experiments.composable import TrainingSession
            from smartsom.telemetry.runtime import (
                CURRENT,
                DisplayOptions,
                RuntimeDisplay,
            )

            training_session_factory = TrainingSession
            if CURRENT.get() is None:
                display = RuntimeDisplay(
                    DisplayOptions(progress="off", verbose=False),
                    kind="training",
                    quiet=True,
                )
                display.bind(self.root)
                self.token = CURRENT.set(display)
        self.record.update(scientific_sha256=self.identity, status="running")
        try:
            self.session = training_session_factory(
                self.prepared,
                self.root,
                self.record,
                sampling_numerical_threads=(
                    1
                    if self.record.get("tuning", {})
                    .get("execution_contract", {})
                    .get("sampling_child_threads")
                    == 1
                    else self.original.config.runtime.numerical_threads
                ),
            )
            if continuation:
                self.load_checkpoint(continuation, _files_restored=True)
            else:
                controls = self.record["tuning"].get("control_spec") or {}
                if self.session.settings.record_initial or "initial" in controls.get(
                    "names", ()
                ):
                    self.session.save()
            self._runtime("restored" if continuation else "initializing")
        except BaseException:
            self.close()
            raise

    def _portable_prepared(self, prepared):
        declarations = json.loads(prepared.policies_json)
        for group, declaration in declarations.items():
            model = declaration.get("resolved_model")
            if not model:
                continue
            if "/" in group or "\\" in group or group in {".", ".."}:
                raise ValueError("invalid dependency group")
            target = self.root / "dependencies/adaptive" / group
            if not target.exists():
                source = model["source"]
                source = (
                    self.root / source[5:]
                    if source.startswith("$RUN/")
                    else Path(source)
                )
                _copy(source, target)
            model["source"] = str(target)
        return replace(prepared, policies_json=canonical_json(declarations))

    def _restore_files(self, directory, marker):
        support = directory / "support"
        current = self.root / "checkpoints" / f"update-{marker['updates']:06d}"
        _copy_native(directory, current)
        for name in ("checkpoints", "logs", "reports", "evidence", "evaluation"):
            origin = support / name
            if origin.exists():
                for entry in origin.iterdir():
                    _copy(entry, self.root / name / entry.name)
        origin = support / "dependencies"
        if origin.exists():
            for entry in origin.iterdir():
                _copy(entry, self.root / "dependencies/adaptive" / entry.name)
        _copy(
            directory / "original-prepared.json",
            self.root / "config/original-prepared.json",
        )

    def _runtime(self, phase, **values):
        registry = getattr(self, "_owned_children", {})
        created = None
        samplers = self.original.config.runtime.sampling_processes
        complete = samplers == 0
        try:
            import psutil

            process = psutil.Process()
            created = process.create_time()
            observed = set()
            for child in process.children(recursive=True):
                timestamp = child.create_time()
                observed.add(child.pid)
                registry[(child.pid, timestamp)] = {
                    "pid": child.pid,
                    "create_time": timestamp,
                }
            executor = getattr(self.session, "executor", None)
            pool = getattr(executor, "_processes", None)
            if samplers:
                # _max_workers is a ceiling: a sequential submit/result pool
                # can legitimately contain fewer actual workers. Only certify
                # all currently observed descendants at an idle native boundary.
                # Starting another update invalidates the certificate because
                # that submit may lazily create an additional worker.
                complete = self.session is not None and (
                    (phase == "closed" and executor is None)
                    or (
                        phase in {"committed", "restored"}
                        and executor is not None
                        and isinstance(pool, dict)
                        and set(pool) <= observed
                    )
                )
        except ImportError:
            pass
        except (psutil.Error, OSError):
            # macOS process enumeration can raise a native PermissionError,
            # rather than psutil.AccessDenied. Do not mask the native lifecycle
            # error, and never certify ownership after an incomplete scan.
            complete = False
        self._owned_children = registry
        self._children_complete = complete
        write_json(
            self.root / "tuning-runtime.json",
            {
                "schema": "smartsom.tune-worker/v1",
                "experiment_id": self.experiment_id,
                "phase": phase,
                "ray_trial_id": self.record["tuning"].get("ray_trial_id"),
                "pid": os.getpid(),
                "pid_create_time": created,
                "child_processes": list(registry.values()),
                "children_complete": self._children_complete,
                "updated_at": time.time(),
                "threads": self.threads,
                "allocation_epoch": self.epoch,
                "physical_ticks": self.session.ticks if self.session else 0,
                "updates": self.session.updates if self.session else 0,
                **values,
            },
        )

    def _native_checkpoint(self):
        checkpoint = self.root / "checkpoints" / f"update-{self.session.updates:06d}"
        if not (checkpoint / "continuation.pkl").is_file():
            checkpoint = self.session.save()
        return Path(checkpoint)

    def _support(self, staging):
        support = staging / "support"
        support.mkdir()
        selected = {"update-000000", f"update-{self.session.updates:06d}"}
        for pointer_name in ("last", "best"):
            pointer = self.root / "checkpoints" / (pointer_name + ".json")
            if pointer.is_file():
                data = json.loads(pointer.read_text())
                name = data["checkpoint"]
                if not name.startswith("update-") or not name[7:].isdigit():
                    raise ValueError("native checkpoint pointer changed format")
                selected.add(name)
                _copy(pointer, support / "checkpoints" / pointer.name)
        for name in selected:
            origin = self.root / "checkpoints" / name
            if origin.exists() and name != f"update-{self.session.updates:06d}":
                _copy_native(origin, support / "checkpoints" / name)
        for name in ("logs", "reports", "evidence", "evaluation"):
            _copy(self.root / name, support / name)
        dependencies = self.root / "dependencies/adaptive"
        if dependencies.exists():
            _copy(dependencies, support / "dependencies")

    def _commit(self, phase="training"):
        key = (self.session.updates, self.session.ticks, phase, digest(self.record))
        if key == self._last_key and self.last_commit is not None:
            return verify_commit(self.last_commit)
        native = self._native_checkpoint()
        base = self.root / "checkpoints/adaptive"
        base.mkdir(parents=True, exist_ok=True)
        staging = base / (".pending-" + uuid4().hex)
        _copy_native(native, staging)
        _copy(
            self.root / "config/original-prepared.json",
            staging / "original-prepared.json",
        )
        self._support(staging)
        write_json(staging / "record.json", self.record)
        files = file_digests(staging)
        marker = {
            "schema": CONTRACT,
            "experiment_id": self.experiment_id,
            "scientific_sha256": self.identity,
            "source_identity_sha256": self.source_identity,
            "updates": self.session.updates,
            "physical_ticks": self.session.ticks,
            "status": self.record["status"],
            "phase": phase,
            "allocation_epoch": self.epoch,
            "threads": self.threads,
            "files": files,
            "checkpoint_digest": digest(files),
            "directory_sync": "unsupported_windows" if os.name == "nt" else "fsync",
        }
        marker["commit_id"] = digest(marker)
        # Windows _commit/fsync requires write access; r+b never truncates.
        sync_mode = "r+b" if os.name == "nt" else "rb"
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open(sync_mode) as stream:
                    os.fsync(stream.fileno())
        write_json(staging / "commit.json", marker)
        with (staging / "commit.json").open(sync_mode) as stream:
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        final = base / f"update-{self.session.updates:06d}-{marker['commit_id'][:16]}"
        if final.exists():
            verify_commit(final)
            shutil.rmtree(staging)
        else:
            staging.rename(final)
        _fsync_directory(base)
        verify_commit(final)
        write_json(
            self.root / "checkpoints/adaptive-recovery.json",
            {
                "checkpoint": str(final.relative_to(self.root / "checkpoints")),
                "commit_id": marker["commit_id"],
                "physical_ticks": self.session.ticks,
            },
        )
        self.last_commit, self._last_key = final, key
        self._runtime(
            "committed",
            checkpoint=str(final),
            commit_id=marker["commit_id"],
            checkpoint_digest=marker["checkpoint_digest"],
            directory_sync=marker["directory_sync"],
        )
        return marker

    def _metrics(self, marker, stages=None):
        return {
            "done": self.final_done,
            "updates": marker["updates"],
            "physical_ticks": marker["physical_ticks"],
            "status": "completed" if self.final_done else "running",
            "training_status": self.record["status"],
            "phase": marker["phase"],
            "experiment_id": self.experiment_id,
            "commit_id": marker["commit_id"],
            "checkpoint_digest": marker["checkpoint_digest"],
            "checkpoint": str(self.last_commit),
            "run_dir": str(self.root),
            "actual_threads": self.threads,
            "allocation_epoch": self.epoch,
            **(stages or {}),
        }

    def step(self):
        if self._native_failed:
            raise RuntimeError(
                "native update failed; restore the last complete adaptive checkpoint"
            )
        if self.final_done:
            return self._metrics(verify_commit(self.last_commit))
        stages = {}
        if not self._training_finished:
            try:
                self._runtime("training")
                if not self.session.training_done:
                    start = time.monotonic()
                    self.session.step_update()
                    stages["training_update_seconds"] = time.monotonic() - start
                if self.session.training_done:
                    self.session.finish()
                    self._training_finished = True
                self._commit(
                    "training_complete" if self._training_finished else "training"
                )
            except BaseException:
                self._native_failed = True
                raise
        if self._training_finished:
            start = time.monotonic()
            self._finish_evaluation()
            stages["final_evaluation_seconds"] = time.monotonic() - start
            self.record["tuning"]["experiment_status"] = "completed"
            self._commit("experiment_complete")
            self.final_done = True
        return self._metrics(verify_commit(self.last_commit), stages)

    def _finish_evaluation(self):
        if self.finalizer is not None:
            self.finalizer(self)
            return
        from smartsom.experiments.composable import (
            checkpoint_path,
            evaluate_cases,
            evaluation_recipe,
            summarize,
        )
        from smartsom.experiments.composable_study import _control_lock

        self._runtime("evaluation")
        cases = json.loads(self.prepared.evaluation_json)
        output = self.root / "evaluation/tuning-final.json"
        if output.exists():
            rows = json.loads(output.read_text())
        else:
            rows = evaluate_cases(
                evaluation_recipe(
                    self.prepared,
                    checkpoint_path(
                        self.root, self.prepared.config.evaluation.checkpoint
                    ),
                ),
                cases,
                directory=self.root / "evaluation/evidence",
                label="final evaluation",
            )
            write_json(output, rows)
        _validate_cases(rows, cases, "final evaluation")
        write_json(self.root / "evaluation/summary.json", summarize(rows))
        controls = self.record["tuning"].get("control_spec") or {}
        references = {}
        for name in controls.get("names", ()):
            if name not in {"initial", "rule", "random"}:
                raise ValueError("unknown paired control")
            key = self.experiment_id if name == "initial" else controls["pair_key"]
            if not key or any(character in key for character in ("/", "\\")):
                raise ValueError("invalid paired control identity")
            path = Path(controls["directory"]) / (key + "_" + name + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            local = self.root / "evaluation/controls" / (name + ".json")
            if local.is_file():
                rows = json.loads(local.read_text())
                _validate_cases(rows, cases, name + " control")
                references[name] = {
                    "path": str(local.relative_to(self.root)),
                    "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                    "shared_source": str(path),
                }
                continue
            self._runtime("waiting for shared control", control=name)
            with _control_lock(path.with_suffix(".lock")):
                if not path.exists():
                    recipe = (
                        evaluation_recipe(
                            self.prepared, self.root / "checkpoints/update-000000"
                        )
                        if name == "initial"
                        else control_recipe(self.prepared, random=name == "random")
                    )
                    write_json(
                        path, evaluate_cases(recipe, cases, label=name + " control")
                    )
                rows = json.loads(path.read_text())
                _validate_cases(rows, cases, name + " control")
                _copy(path, local)
                references[name] = {
                    "path": str(local.relative_to(self.root)),
                    "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                    "shared_source": str(path),
                }
        self.record["tuning"].update(evaluation_dir="evaluation", controls=references)

    def save_checkpoint(self, directory):
        target = Path(directory) / "native"
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.last_commit is None:
            if self._native_failed:
                raise RuntimeError(
                    "failed native update has no complete adaptive checkpoint"
                )
            self._commit("training")
        marker = verify_commit(self.last_commit)
        if target.exists():
            if verify_commit(target)["commit_id"] != marker["commit_id"]:
                raise ValueError("Ray checkpoint destination contains another commit")
        else:
            shutil.copytree(self.last_commit, target)
        verify_commit(target)
        return str(target)

    def load_checkpoint(self, directory, *, _files_restored=False):
        directory = _checkpoint_directory(directory)
        marker = verify_identity(self.original, self.record, directory)
        if not _files_restored:
            self._restore_files(directory, marker)
        saved_record = json.loads((directory / "record.json").read_text())
        tuning = self.record["tuning"]
        self.record.update(saved_record)
        self.record["tuning"] = {**saved_record["tuning"], **tuning}
        self.record["status"] = saved_record["status"]
        self.session.record = self.record
        with (directory / "continuation.pkl").open("rb") as stream:
            state = pickle.load(stream)
        self.session.restore(state)
        if (
            self.session.ticks != marker["physical_ticks"]
            or self.session.updates != marker["updates"]
        ):
            raise ValueError("adaptive marker and native progress disagree")
        self.last_commit = directory
        self._training_finished = marker["phase"] in {
            "training_complete",
            "experiment_complete",
        }
        self.final_done = marker["phase"] == "experiment_complete"
        self._native_failed = False
        self._last_key = (
            self.session.updates,
            self.session.ticks,
            marker["phase"],
            digest(self.record),
        )
        write_json(self.root / "run.json", self.record)
        self._runtime(
            "restored", checkpoint=str(directory), commit_id=marker["commit_id"]
        )

    def close(self):
        try:
            if self.session is not None:
                try:
                    self.session.close()
                finally:
                    self._runtime(
                        "closed",
                        checkpoint=str(self.last_commit) if self.last_commit else None,
                    )
                    self.session = None
        finally:
            if self.token is not None:
                from smartsom.telemetry.runtime import CURRENT

                try:
                    CURRENT.get().close()
                finally:
                    CURRENT.reset(self.token)
                    self.token = None


def _validate_cases(rows, cases, label):
    actual = [
        (row.get("case_id"), row.get("replication"), row.get("seed")) for row in rows
    ]
    expected = [(case["case"], case["replication"], case["seed"]) for case in cases]
    if actual != expected or any(row.get("engineering_failure", True) for row in rows):
        raise RuntimeError(label + " missing frozen cases or has engineering failures")


def control_recipe(prepared, *, random=False):
    """Create controls with the same physical transport and explicit role rules.

    A matrix study's automatic Mover cannot be reused for a grid experiment.
    Central learner controls use four rule groups against the same world.
    """
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "shortest_path",
    }
    automatic = bool(json.loads(prepared.scenario_json).get("transport_matrix"))
    if automatic:
        names["mover"] = "automatic_travel"
    composition = json.loads(prepared.composition_json)
    declarations = json.loads(prepared.policies_json)
    if composition.get("controller"):
        composition["controller"] = None
        composition["groups"] = {
            role: {"policy": "generated paired control"} for role in names
        }
        composition["bindings"] = {
            role: {"default": role, "overrides": {}} for role in names
        }
        declarations = {role: {"role": role} for role in names}
    for declaration in declarations.values():
        role = declaration["role"]
        declaration.pop("resolved_model", None)
        declaration["implementation"] = {
            "kind": "rule",
            "name": "random"
            if random and not (automatic and role == "mover")
            else names[role],
            "parameters": {},
        }
    return replace(
        prepared,
        policies_json=canonical_json(declarations),
        composition_json=canonical_json(composition),
    )
