"""Unsafe missing process ownership is an explicit failure, never silent waiting."""

import pytest

pytest.importorskip("ray")

from smartsom.experiments.tuning_callbacks import EvidenceCallback


def test_incomplete_failed_actor_ownership_stops_segment(tmp_path):
    class Broker:
        def refresh(self):
            pass

        def unresolved_failures(self):
            return ("failed-sampler",)

    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(RuntimeError, match="reservation retained, batch stopped"):
        callback.on_step_begin(1, [])
