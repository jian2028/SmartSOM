"""Sampling ownership is complete at actual idle boundaries, not slot counts."""

import json
import sys
from types import SimpleNamespace

import pytest

from smartsom.experiments.tuning_session import AdaptiveSession


class ProcessError(Exception):
    pass


def runtime_session(tmp_path, *, pool=None, samplers=2, native=True):
    session = object.__new__(AdaptiveSession)
    session.root = tmp_path
    session.original = SimpleNamespace(
        config=SimpleNamespace(runtime=SimpleNamespace(sampling_processes=samplers))
    )
    session.record = {"tuning": {"ray_trial_id": "trial-1"}}
    session.experiment_id = "experiment-1"
    session.threads, session.epoch = 1, 2
    executor = SimpleNamespace(_processes=pool) if pool is not None else None
    session.session = (
        SimpleNamespace(executor=executor, ticks=1, updates=1) if native else None
    )
    return session


def processes(monkeypatch, children, *, error=None):
    def descendants(recursive=False):
        assert recursive
        if error:
            raise error("descendant inspection denied")
        return [
            SimpleNamespace(pid=pid, create_time=lambda stamp=stamp: stamp)
            for pid, stamp in children
        ]

    process = SimpleNamespace(create_time=lambda: 10.0, children=descendants)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: process, Error=ProcessError),
    )


def payload(session, phase):
    session._runtime(phase)
    return json.loads((session.root / "tuning-runtime.json").read_text())


def test_idle_lazy_pool_registers_actual_worker_and_descendants(tmp_path, monkeypatch):
    session = runtime_session(tmp_path, pool={20: object()})
    processes(monkeypatch, [(20, 20.0), (21, 21.0)])
    result = payload(session, "committed")
    assert result["children_complete"]
    assert result["child_processes"] == [
        {"pid": 20, "create_time": 20.0},
        {"pid": 21, "create_time": 21.0},
    ]
    assert result["pid_create_time"] == 10.0


@pytest.mark.parametrize("phase", ("initializing", "training", "evaluation"))
def test_busy_or_constructor_boundary_does_not_reuse_prior_certificate(
    phase, tmp_path, monkeypatch
):
    session = runtime_session(tmp_path, pool={20: object()})
    session._children_complete = True
    processes(monkeypatch, [(20, 20.0)])
    assert not payload(session, phase)["children_complete"]


def test_missing_actual_worker_is_incomplete_even_if_pid_was_previously_known(
    tmp_path, monkeypatch
):
    session = runtime_session(tmp_path, pool={20: object(), 30: object()})
    session._owned_children = {(30, 1.0): {"pid": 30, "create_time": 1.0}}
    session._children_complete = True
    processes(monkeypatch, [(20, 20.0)])
    assert not payload(session, "committed")["children_complete"]


@pytest.mark.parametrize("error", (ProcessError, PermissionError, OSError))
def test_inspection_failure_invalidates_previous_certificate(
    error, tmp_path, monkeypatch
):
    session = runtime_session(tmp_path, pool={20: object()})
    session._children_complete = True
    processes(monkeypatch, [], error=error)
    assert not payload(session, "committed")["children_complete"]


def test_no_sampler_does_not_certify_a_denied_process_scan(tmp_path, monkeypatch):
    session = runtime_session(tmp_path, samplers=0)
    processes(monkeypatch, [], error=PermissionError)
    assert not payload(session, "closed")["children_complete"]


def test_empty_restored_pool_is_complete_before_next_submit(tmp_path, monkeypatch):
    session = runtime_session(tmp_path, pool={})
    processes(monkeypatch, [])
    assert payload(session, "restored")["children_complete"]
    assert not payload(session, "training")["children_complete"]


def test_successful_close_retains_lifetime_registry_and_records_remaining_helpers(
    tmp_path, monkeypatch
):
    session = runtime_session(tmp_path)
    session._owned_children = {(20, 20.0): {"pid": 20, "create_time": 20.0}}
    processes(monkeypatch, [(21, 21.0)])
    result = payload(session, "closed")
    assert result["children_complete"]
    assert {p["pid"] for p in result["child_processes"]} == {20, 21}


@pytest.mark.parametrize("native", (False, True))
def test_constructor_or_shutdown_failure_is_fail_closed(native, tmp_path, monkeypatch):
    session = runtime_session(tmp_path, pool={}, native=native)
    processes(monkeypatch, [])
    assert not payload(session, "closed")["children_complete"]
