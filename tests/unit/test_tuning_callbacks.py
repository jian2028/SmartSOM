"""Unsafe missing process ownership is an explicit failure, never silent waiting."""

import json

import pytest

pytest.importorskip("ray")

from smartsom.experiments.tuning_callbacks import EvidenceCallback


def _frozen_plan(tmp_path, monkeypatch, *, live_source=None, live_code="frozen"):
    from smartsom.experiments import composable, tuning_callbacks

    source = {"git": {"commit": "start"}}
    (tmp_path / "plan.json").write_text(
        json.dumps({"source": source, "implementation_sha256": "frozen"})
    )
    monkeypatch.setattr(composable, "implementation_identity", lambda: live_code)
    monkeypatch.setattr(
        tuning_callbacks, "source_identity", lambda: live_source or source
    )


def test_incomplete_failed_actor_ownership_stops_segment(tmp_path, monkeypatch):
    _frozen_plan(tmp_path, monkeypatch)

    class Broker:
        def refresh(self):
            pass

        def unresolved_failures(self):
            return ("failed-sampler",)

    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(RuntimeError, match="reservation retained, batch stopped"):
        callback.on_step_begin(1, [])


@pytest.mark.parametrize(
    ("live_source", "live_code", "message"),
    [
        (None, "changed", "implementation changed after freezing"),
        ({"git": {"commit": "later"}}, "frozen", "checkout or dependencies changed"),
    ],
)
def test_source_change_aborts_before_more_trials(
    tmp_path, monkeypatch, live_source, live_code, message
):
    _frozen_plan(tmp_path, monkeypatch, live_source=live_source, live_code=live_code)

    class Broker:
        def refresh(self):
            raise AssertionError("changed source must stop before resource admission")

    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(RuntimeError, match=message):
        callback.on_step_begin(1, [])
