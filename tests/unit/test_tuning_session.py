"""Adaptive checkpoints exercise the native continuation protocol without Ray."""

import copy
import json
import pickle
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.config.codec import canonical_json
from smartsom.experiments.evidence import write_json
from smartsom.experiments.tuning_session import (
    AdaptiveSession,
    _validate_cases,
    effective_prepared,
    scientific_identity,
    verify_commit,
    verify_identity,
)


@dataclass(frozen=True)
class Prepared:
    config_json: str
    scenario_json: str = '{"factory":"large","tick_limit":99}'
    composition_json: str = '{"bindings":{"machine":"learner"}}'
    policies_json: str = "{}"
    parameters_json: str = '{"learning_rate":0.001,"hidden_sizes":[16]}'
    validation_json: str = '[{"case":"v","replication":0,"seed":2}]'
    evaluation_json: str = '[{"case":"e","replication":0,"seed":3}]'
    origins_json: str = '{"scenario":"configs/scenarios/base.yaml"}'
    scientific_sha256: str = "strict-native-identity"
    training_inputs_json: str = '{"workload":{"arrivals":[1,2]}}'

    @property
    def config(self):
        raw = json.loads(self.config_json)
        return SimpleNamespace(
            runtime=SimpleNamespace(**raw["runtime"]),
            evaluation=SimpleNamespace(checkpoint="best"),
        )


def prepared(**changes):
    config = {
        "seed": 11,
        "runtime": {
            "numerical_threads": 1,
            "max_concurrent": 2,
            "num_envs": 1,
            "sampling_processes": 0,
            "device": "cpu",
        },
        "training": {"updates": 3, "record_initial": True},
        "logging": {"level": "info"},
        "output": "runs/test",
    }
    return replace(Prepared(canonical_json(config)), **changes)


def record(**tuning):
    return {
        "id": "native-run",
        "status": "running",
        "source": {"commit": "abc", "dirty": False, "dependencies": {"torch": "2"}},
        "implementation_sha256": "source-code-hash",
        "scientific_sha256": "strict-native-identity",
        "tuning": {"experiment_id": "experiment-1", **tuning},
    }


class Native:
    """Fake learner keeps optimizer, replay, RNG, patience and sampler state."""

    def __init__(self, prepared, root, record, *, sampling_numerical_threads=None):
        self.prepared, self.root, self.record = prepared, root, record
        self.threads_before_restore = prepared.config.runtime.numerical_threads
        self.sampling_threads = sampling_numerical_threads
        config = json.loads(prepared.config_json)["training"]
        self.limit = config["updates"]
        self.settings = SimpleNamespace(record_initial=config["record_initial"])
        self.updates = self.ticks = 0
        self.save_calls = self.finish_calls = self.restore_calls = 0
        self.closed = False
        self.rng = random.Random(27)
        self.values = {
            "optimizer": {"moments": [0.0]},
            "replay": [],
            "collector": {"cursor": 0},
            "best_score": None,
            "best_update": 0,
            "no_improvement": 0,
            "policy": [0.0],
        }
        assert all(
            (root / name).is_dir()
            for name in ("config", "checkpoints", "reports", "logs", "evidence")
        )
        for declaration in json.loads(prepared.policies_json).values():
            if declaration.get("resolved_model"):
                assert Path(declaration["resolved_model"]["source"]).exists()

    @property
    def training_done(self):
        return self.updates >= self.limit or self.record["status"] == "early_stopped"

    def state(self):
        return {
            "scientific_sha256": self.prepared.scientific_sha256,
            "updates": self.updates,
            "ticks": self.ticks,
            "rng": self.rng.getstate(),
            "values": copy.deepcopy(self.values),
        }

    def save(self):
        self.save_calls += 1
        path = self.root / "checkpoints" / f"update-{self.updates:06d}"
        (path / "groups/learner").mkdir(parents=True, exist_ok=True)
        (path / "groups/learner/weights.pt").write_bytes(
            str(self.values["policy"]).encode()
        )
        write_json(
            path / "groups/learner/model.json",
            {"identity": self.prepared.scientific_sha256},
        )
        (path / "continuation.pkl").write_bytes(pickle.dumps(self.state()))
        write_json(
            path / "snapshot.json",
            {"updates": self.updates, "physical_ticks": self.ticks},
        )
        pointer = {"checkpoint": path.name}
        write_json(self.root / "checkpoints/last.json", pointer)
        if self.updates <= 1:
            write_json(self.root / "checkpoints/best.json", pointer)
        return path

    def step_update(self):
        assert not self.training_done
        draw = self.rng.random()
        self.updates += 1
        self.ticks += 2
        self.values["policy"][0] += draw
        self.values["optimizer"]["moments"][0] += draw / 2
        self.values["replay"].append(draw)
        self.values["collector"]["cursor"] += 1
        if self.updates == 1:
            self.values.update(best_score=draw, best_update=1)
        else:
            self.values["no_improvement"] += 1
        self.record.update(updates=self.updates, physical_ticks=self.ticks)
        write_json(
            self.root / "logs" / f"selection-{self.updates:06d}.json", self.values
        )
        self.save()  # Post-validation state, including patience and best selection.
        return copy.deepcopy(self.record)

    def finish(self):
        self.finish_calls += 1
        assert self.training_done
        if self.record["status"] != "early_stopped":
            self.record["status"] = "completed"
        write_json(self.root / "reports/training.json", {"updates": self.updates})
        return SimpleNamespace(run_dir=self.root)

    def restore(self, state):
        self.restore_calls += 1
        assert (
            self.threads_before_restore
            == self.prepared.config.runtime.numerical_threads
        )
        if state["scientific_sha256"] != self.prepared.scientific_sha256:
            raise ValueError("native state identity changed")
        self.updates, self.ticks = state["updates"], state["ticks"]
        self.values = copy.deepcopy(state["values"])
        self.rng.setstate(state["rng"])

    def close(self):
        self.closed = True


def finalizer(session):
    write_json(session.root / "evaluation/summary.json", {"done": True})


def session(tmp_path, *, prep=None, rec=None, **kwargs):
    return AdaptiveSession(
        prep or prepared(),
        tmp_path,
        rec or record(),
        training_session_factory=Native,
        finalizer=finalizer,
        **kwargs,
    )


def finish(wrapper):
    while not wrapper.step()["done"]:
        pass
    return wrapper.session.state()


def test_identity_waives_only_two_execution_fields():
    original = prepared()
    config = json.loads(original.config_json)
    config["runtime"].update(numerical_threads=8, max_concurrent=9)
    changed = replace(
        original,
        config_json=canonical_json(config),
        scientific_sha256="another-derived-hash",
    )
    assert scientific_identity(changed) == scientific_identity(original)
    for field, value in (("num_envs", 2), ("device", "cuda")):
        altered = copy.deepcopy(config)
        altered["runtime"][field] = value
        assert scientific_identity(
            replace(original, config_json=canonical_json(altered))
        ) != scientific_identity(original)
    for name in (
        "scenario_json",
        "composition_json",
        "parameters_json",
        "validation_json",
        "evaluation_json",
        "origins_json",
        "training_inputs_json",
        "policies_json",
    ):
        assert scientific_identity(
            replace(original, **{name: '{"changed":true}'})
        ) != scientific_identity(original)
    config["logging"]["level"] = "debug"
    assert scientific_identity(
        replace(original, config_json=canonical_json(config))
    ) != scientific_identity(original)
    effective = effective_prepared(original, 3)
    assert json.loads(effective.config_json)["runtime"]["numerical_threads"] == 3
    assert effective.scientific_sha256 == scientific_identity(original)
    assert original.scientific_sha256 == "strict-native-identity"


@pytest.mark.parametrize("threads", (0, -1, 1.5, True))
def test_allocation_rejects_invalid_threads(threads):
    with pytest.raises(ValueError, match="positive integer"):
        effective_prepared(prepared(), threads)


def test_checkpoint_post_validation_state_and_repeat_save(tmp_path):
    wrapper = session(tmp_path / "run")
    metrics = wrapper.step()
    path = Path(wrapper.save_checkpoint(tmp_path / "ray"))
    state = pickle.loads((path / "continuation.pkl").read_bytes())
    assert state["values"]["best_update"] == 1
    assert state["values"]["best_score"] is not None
    assert state["values"]["replay"] and state["values"]["optimizer"]["moments"]
    marker = verify_commit(path)
    assert metrics["experiment_id"] == marker["experiment_id"]
    assert metrics["commit_id"] == marker["commit_id"]
    assert metrics["checkpoint_digest"] == marker["checkpoint_digest"]
    calls = wrapper.session.save_calls
    (tmp_path / "ray/.metadata").write_text("Ray bookkeeping")
    assert verify_commit(tmp_path / "ray") == marker
    assert wrapper.save_checkpoint(tmp_path / "ray") == str(path)
    wrapper.save_checkpoint(tmp_path / "ray-again")
    assert wrapper.session.save_calls == calls
    assert (
        json.loads((wrapper.root / "checkpoints/last.json").read_text())["checkpoint"]
        == "update-000001"
    )


@pytest.mark.parametrize("change", ("weights", "missing", "marker", "extra", "symlink"))
def test_checkpoint_integrity_rejects_mutation(tmp_path, change):
    wrapper = session(tmp_path / "run")
    wrapper.step()
    path = Path(wrapper.save_checkpoint(tmp_path / "ray"))
    if change == "weights":
        (path / "groups/learner/weights.pt").write_bytes(b"changed")
    elif change == "missing":
        (path / "continuation.pkl").unlink()
    elif change == "marker":
        marker = json.loads((path / "commit.json").read_text())
        marker["physical_ticks"] += 1
        write_json(path / "commit.json", marker)
    elif change == "extra":
        (path / "support/commit.json").write_text("nested markers are hashed")
    else:
        (path / "extra-link").symlink_to(path / "snapshot.json")
    with pytest.raises(ValueError):
        verify_commit(path)


@pytest.mark.parametrize("change", ("science", "source", "dependency", "experiment"))
def test_identity_checked_before_native_construction(tmp_path, change):
    prep, rec = prepared(), record()
    wrapper = session(tmp_path / "run", prep=prep, rec=rec)
    wrapper.step()
    saved = wrapper.save_checkpoint(tmp_path / "ray")
    incoming = record(continuation=saved)
    if change == "science":
        prep = replace(prep, parameters_json='{"learning_rate":2}')
    elif change == "source":
        incoming["source"]["commit"] = "different"
    elif change == "dependency":
        incoming["source"]["dependencies"]["torch"] = "different"
    else:
        incoming["tuning"]["experiment_id"] = "different"
    constructed = []

    def factory(*args, **kwargs):
        constructed.append(args)
        raise AssertionError("must never construct")

    with pytest.raises(ValueError, match="identity changed"):
        AdaptiveSession(
            prep, tmp_path / "new", incoming, training_session_factory=factory
        )
    assert not constructed


def test_verified_marker_cannot_be_tampered(tmp_path):
    wrapper = session(tmp_path / "run")
    wrapper.step()
    marker = verify_commit(wrapper.last_commit)
    marker["threads"] = 90
    with pytest.raises(ValueError, match="commit identity"):
        verify_identity(wrapper.original, wrapper.record, marker)


def test_pause_resume_preserves_full_state_and_current_allocation(tmp_path):
    uninterrupted = session(tmp_path / "continuous")
    expected = finish(uninterrupted)
    interrupted = session(tmp_path / "before")
    interrupted.step()
    original_bytes = (interrupted.root / "config/original-prepared.json").read_bytes()
    saved = interrupted.save_checkpoint(tmp_path / "ray")
    interrupted.close()
    resumed = session(
        tmp_path / "after",
        rec=record(continuation=saved),
        threads=4,
        allocation_epoch=1,
    )
    assert resumed.session.threads_before_restore == 4
    assert resumed.session.sampling_threads == 1
    assert resumed.session.restore_calls == 1
    assert (
        resumed.root / "config/original-prepared.json"
    ).read_bytes() == original_bytes
    assert resumed.record["source"] == record()["source"]
    assert finish(resumed) == expected
    assert verify_commit(resumed.last_commit)["threads"] == 4


def test_runtime_whitelist_preserves_original_preparation_bytes(tmp_path):
    original = prepared()
    wrapper = session(tmp_path / "before", prep=original)
    wrapper.step()
    saved = Path(wrapper.save_checkpoint(tmp_path / "ray"))
    original_bytes = (saved / "original-prepared.json").read_bytes()
    config = json.loads(original.config_json)
    config["runtime"].update(numerical_threads=8, max_concurrent=5)
    allocation_input = replace(original, config_json=canonical_json(config))
    resumed = session(
        tmp_path / "after",
        prep=allocation_input,
        rec=record(continuation=str(saved)),
        threads=8,
    )
    assert resumed.original == original
    assert (
        resumed.root / "config/original-prepared.json"
    ).read_bytes() == original_bytes
    assert resumed.session.threads_before_restore == 8


def test_training_complete_commit_survives_failed_finalizer(tmp_path):
    prep = prepared()
    config = json.loads(prep.config_json)
    config["training"]["updates"] = 1
    prep = replace(prep, config_json=canonical_json(config))

    def failing_finalizer(wrapper):
        write_json(wrapper.root / "evaluation/partial.json", {"phase": "evaluation"})
        raise RuntimeError("evaluation temporarily failed")

    wrapper = AdaptiveSession(
        prep,
        tmp_path / "before",
        record(),
        training_session_factory=Native,
        finalizer=failing_finalizer,
    )
    with pytest.raises(RuntimeError, match="temporarily failed"):
        wrapper.step()
    assert not wrapper.final_done
    assert verify_commit(wrapper.last_commit)["phase"] == "training_complete"
    saved = wrapper.save_checkpoint(tmp_path / "ray")
    resumed = session(
        tmp_path / "after", prep=prep, rec=record(continuation=saved), threads=2
    )
    assert resumed._training_finished and not resumed.final_done
    metrics = resumed.step()
    assert metrics["done"] and metrics["phase"] == "experiment_complete"
    assert resumed.session.updates == 1 and resumed.session.finish_calls == 0
    terminal = resumed.save_checkpoint(tmp_path / "terminal")
    restored = session(
        tmp_path / "terminal-run", prep=prep, rec=record(continuation=terminal)
    )
    original_state = restored.session.state()
    assert restored.step()["done"]
    assert restored.session.state() == original_state
    assert restored.session.finish_calls == 0 and restored.session.save_calls == 0


def test_terminal_commit_failure_does_not_claim_completion(tmp_path, monkeypatch):
    wrapper = session(tmp_path / "run")
    wrapper.step()
    wrapper.step()
    commit = wrapper._commit

    def fail(phase="training"):
        if phase == "experiment_complete":
            raise OSError("disk unavailable")
        return commit(phase)

    monkeypatch.setattr(wrapper, "_commit", fail)
    with pytest.raises(OSError):
        wrapper.step()
    assert not wrapper.final_done
    assert verify_commit(wrapper.last_commit)["phase"] == "training_complete"
    monkeypatch.setattr(wrapper, "_commit", commit)
    assert wrapper.step()["done"]
    assert wrapper.session.updates == 3


def test_incomplete_native_update_requires_complete_restore(tmp_path, monkeypatch):
    wrapper = session(tmp_path / "run")
    baseline = wrapper.save_checkpoint(tmp_path / "baseline")
    native_update = wrapper.session.step_update

    def incomplete():
        native_update()
        raise RuntimeError("validation interrupted")

    monkeypatch.setattr(wrapper.session, "step_update", incomplete)
    with pytest.raises(RuntimeError, match="validation interrupted"):
        wrapper.step()
    with pytest.raises(RuntimeError, match="restore the last complete"):
        wrapper.step()
    copied = wrapper.save_checkpoint(tmp_path / "still-baseline")
    assert verify_commit(copied)["updates"] == 0
    wrapper.load_checkpoint(baseline)
    assert wrapper.session.updates == 0
    monkeypatch.setattr(wrapper.session, "step_update", native_update)
    assert wrapper.step()["updates"] == 1


def test_initial_control_creates_initial_checkpoint_when_disabled(tmp_path):
    prep = prepared()
    config = json.loads(prep.config_json)
    config["training"]["record_initial"] = False
    prep = replace(prep, config_json=canonical_json(config))
    no_control = session(tmp_path / "no-control", prep=prep)
    assert no_control.session.save_calls == 0
    wrapper = session(
        tmp_path / "run", prep=prep, rec=record(control_spec={"names": ["initial"]})
    )
    assert (wrapper.root / "checkpoints/update-000000/continuation.pkl").is_file()


def test_portable_dependencies_best_initial_and_records(tmp_path):
    dependency = tmp_path / "old-model"
    dependency.mkdir()
    (dependency / "weights.pt").write_bytes(b"frozen weights")
    (dependency / "model.json").write_text("frozen metadata")
    prep = prepared(
        policies_json=canonical_json(
            {"partner": {"resolved_model": {"source": str(dependency)}}}
        )
    )
    wrapper = session(tmp_path / "before", prep=prep)
    wrapper.step()
    wrapper.step()
    saved = Path(wrapper.save_checkpoint(tmp_path / "ray"))
    assert (
        saved / "support/checkpoints/update-000001/groups/learner/weights.pt"
    ).is_file()
    assert (saved / "support/checkpoints/update-000000/continuation.pkl").is_file()
    assert (saved / "support/logs/selection-000001.json").is_file()
    original_bytes = (saved / "original-prepared.json").read_bytes()
    wrapper.close()
    shutil.rmtree(wrapper.root)
    shutil.rmtree(dependency)
    resumed = session(
        tmp_path / "after", prep=prep, rec=record(continuation=str(saved)), threads=2
    )
    source = Path(
        json.loads(resumed.prepared.policies_json)["partner"]["resolved_model"][
            "source"
        ]
    )
    assert (source / "weights.pt").read_bytes() == b"frozen weights"
    assert (
        resumed.root / "config/original-prepared.json"
    ).read_bytes() == original_bytes
    assert resumed.step()["done"]
    second = Path(resumed.save_checkpoint(tmp_path / "ray-2"))
    assert not list(second.rglob("support/support"))
    assert not list((second / "support/checkpoints").rglob("support"))
    assert (
        second / "support/checkpoints/update-000001/groups/learner/weights.pt"
    ).is_file()


def test_early_stop_continuation_does_not_advance(tmp_path):
    wrapper = session(tmp_path / "before")
    wrapper.step()
    wrapper.record["status"] = "early_stopped"
    wrapper.session.finish()
    wrapper._training_finished = True
    wrapper._commit("training_complete")
    saved = wrapper.save_checkpoint(tmp_path / "ray")
    resumed = session(tmp_path / "after", rec=record(continuation=saved))
    metrics = resumed.step()
    assert metrics["done"] and metrics["training_status"] == "early_stopped"
    assert resumed.session.updates == 1


def test_frozen_evaluation_case_validation():
    cases = [{"case": "e", "replication": 0, "seed": 3}]
    rows = [{"case_id": "e", "replication": 0, "seed": 3, "engineering_failure": False}]
    _validate_cases(rows, cases, "test")
    for changes in (
        {"seed": 4},
        {"replication": 1},
        {"case_id": "other"},
        {"engineering_failure": True},
    ):
        bad = [{**rows[0], **changes}]
        with pytest.raises(RuntimeError, match="frozen cases"):
            _validate_cases(bad, cases, "test")


@pytest.mark.parametrize("platform", ["native", "nt"])
def test_commit_fsync_uses_supported_descriptor_and_preserves_content(
    tmp_path, monkeypatch, platform
):
    import os
    import stat

    from smartsom.experiments import tuning_session

    file_handles = {}
    synced_files = []
    original_open = Path.open
    original_fsync = os.fsync
    selected_platform = os.name if platform == "native" else platform

    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        file_handles[stream.fileno()] = (path, stream)
        return stream

    def fsync(descriptor):
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            path, stream = file_handles[descriptor]
            assert not stream.closed
            if selected_platform == "nt":
                assert stream.writable(), "Windows fsync requires a writable descriptor"
            synced_files.append(path.name)
        original_fsync(descriptor)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(
        tuning_session,
        "os",
        SimpleNamespace(
            name=selected_platform,
            fsync=fsync,
            open=os.open,
            close=os.close,
            getpid=os.getpid,
            O_RDONLY=os.O_RDONLY,
        ),
    )
    wrapper = session(tmp_path)
    try:
        marker = wrapper._commit()
        assert verify_commit(wrapper.last_commit) == marker
        assert marker["directory_sync"] == (
            "unsupported_windows" if selected_platform == "nt" else "fsync"
        )
        assert "weights.pt" in synced_files and "commit.json" in synced_files
    finally:
        wrapper.close()


def test_windows_directory_sync_is_explicitly_unsupported(monkeypatch, tmp_path):
    from smartsom.experiments import tuning_session

    def forbidden(*args):
        pytest.fail("Windows must not attempt or pretend POSIX directory sync")

    monkeypatch.setattr(
        tuning_session, "os", SimpleNamespace(name="nt", open=forbidden)
    )
    assert tuning_session._fsync_directory(tmp_path) is False


@pytest.mark.parametrize("fails", [False, True])
def test_posix_directory_sync_closes_descriptor_and_propagates_failure(
    monkeypatch, tmp_path, fails
):
    from smartsom.experiments import tuning_session

    calls = []

    def opened(path, flags):
        calls.append(("open", path, flags))
        return 42

    def fsync(descriptor):
        calls.append(("fsync", descriptor))
        if fails:
            raise OSError("directory flush failure")

    monkeypatch.setattr(
        tuning_session,
        "os",
        SimpleNamespace(
            name="posix",
            open=opened,
            fsync=fsync,
            close=lambda descriptor: calls.append(("close", descriptor)),
            O_RDONLY=0,
        ),
    )
    if fails:
        with pytest.raises(OSError, match="directory flush failure"):
            tuning_session._fsync_directory(tmp_path)
    else:
        assert tuning_session._fsync_directory(tmp_path) is True
    assert calls == [("open", tmp_path, 0), ("fsync", 42), ("close", 42)]


@pytest.mark.parametrize("platform", ["native", "nt"])
@pytest.mark.parametrize(
    "boundary", ["payload", "marker", "rename", "verify", "pointer"]
)
def test_publication_failure_preserves_prior_valid_commit_and_recovery_pointer(
    tmp_path, monkeypatch, platform, boundary
):
    import os
    import stat

    from smartsom.experiments import tuning_session

    wrapper = session(tmp_path)
    wrapper.step()
    prior = wrapper.last_commit
    prior_marker = verify_commit(prior)
    pointer = tmp_path / "checkpoints/adaptive-recovery.json"
    pointer_bytes = pointer.read_bytes()
    original_verify = tuning_session.verify_commit
    original_write = tuning_session.write_json
    original_rename = Path.rename
    original_open = Path.open
    original_fsync = os.fsync
    handles = {}

    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        handles[stream.fileno()] = path
        return stream

    def fsync(descriptor):
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            marker = handles[descriptor].name == "commit.json"
            if (boundary == "marker" and marker) or (
                boundary == "payload" and not marker
            ):
                raise OSError("injected publication failure")
        original_fsync(descriptor)

    def renamed(path, target):
        if boundary == "rename" and path.name.startswith(".pending-"):
            raise OSError("injected publication failure")
        return original_rename(path, target)

    def verified(path):
        if boundary == "verify" and Path(path) != prior:
            raise OSError("injected publication failure")
        return original_verify(path)

    def written(path, value):
        if boundary == "pointer" and path.name == "adaptive-recovery.json":
            raise OSError("injected publication failure")
        return original_write(path, value)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(Path, "rename", renamed)
    monkeypatch.setattr(tuning_session, "verify_commit", verified)
    monkeypatch.setattr(tuning_session, "write_json", written)
    monkeypatch.setattr(
        tuning_session,
        "os",
        SimpleNamespace(
            name=os.name if platform == "native" else platform,
            fsync=fsync,
            open=os.open,
            close=os.close,
            getpid=os.getpid,
            O_RDONLY=os.O_RDONLY,
        ),
    )
    try:
        wrapper.session.step_update()
        with pytest.raises(OSError, match="injected publication failure"):
            wrapper._commit()
        assert wrapper.last_commit == prior
        assert pointer.read_bytes() == pointer_bytes
        assert original_verify(prior) == prior_marker
        assert not wrapper.final_done
    finally:
        wrapper.close()


def test_posix_directory_permission_failure_is_not_suppressed(monkeypatch, tmp_path):
    from smartsom.experiments import tuning_session

    def denied(*args):
        raise PermissionError("directory access denied")

    monkeypatch.setattr(
        tuning_session, "os", SimpleNamespace(name="posix", open=denied, O_RDONLY=0)
    )
    with pytest.raises(PermissionError, match="directory access denied"):
        tuning_session._fsync_directory(tmp_path)
