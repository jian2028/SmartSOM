"""The coordinator keeps physical progress even if a stream operation raises."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.config.codec import digest
from smartsom.config.experiment import load_config, prepare
from smartsom.learning.production_sampling import ProductionSamplingSpec

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def test_local_post_step_failure_keeps_original_exception_and_actual_trace(monkeypatch):
    pytest.importorskip("gymnasium")
    from smartsom.learning.sampling import OrderedSamplingPool

    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda _: {})
    config = load_config(ROOT / "configs/runs/sb3_production.yaml")
    resolved = ProductionSamplingSpec(prepare(config).resolved, config.seed)
    evidence = SimpleNamespace(episode=lambda _: None)
    with OrderedSamplingPool(resolved, 2, 0, evidence) as pool:
        pool.reset()
        env = pool.local[0].env
        original_step = env.step
        failure = OSError("after real physical transition")
        before = env.sim.tick
        action = next(i for i, legal in enumerate(env.action_masks()) if legal)

        def step_then_fail(action):
            original_step(action)
            raise failure

        monkeypatch.setattr(env, "step", step_then_fail)
        with pytest.raises(OSError) as raised:
            pool.step([action, action])
        assert raised.value is failure
        assert len(evidence.active_envs[0].steps) == 1
        assert digest(evidence.active_envs[0].trace) == digest(pool.local[0].trace)
        assert env.sim.tick == before + 1
        assert evidence.active_envs[1].steps == ()
