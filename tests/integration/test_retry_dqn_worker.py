"""Bounded real DQN retry worker; no production experiments."""

import json
from pathlib import Path

from smartsom.experiments import control
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.tuning_resources import ResourceSnapshot
from tests.integration.test_runtime_scheduler import _until
from tests.integration.test_runtime_scheduler import (
    scheduler_factory as scheduler_factory,
)


def _controlled_dqn_retry_worker(directory, parent_root, retry_once=False):
    """Real DQN checkpoint roundtrip inside a controlled test worker."""
    import hashlib
    import random

    from smartsom.learning import state as learner_state
    from smartsom.learning.dqn.learner import BaseDQN
    from tests.helpers.algorithm import algorithm_options
    from tests.integration.test_rule_learning import _equal_dqn_values, _packed_request

    root = Path(directory)
    scope = control.Scope()
    scope.inherit_driver(parent_root)
    scope.bind(root)
    status = "failed"
    try:
        options = algorithm_options(method="dqn", width=8, epochs=1, batch_size=2)
        continuation = root / "continuation"
        pointer = continuation / "current.json"
        if not retry_once:
            policy = BaseDQN(options, 29)
            for tick in range(4):
                policy.choose(_packed_request(tick))
            policy.update([-1.0] * 4, terminal=True)
            assert policy.updates > 0
            generation = continuation / "generation-dqn"
            generation.mkdir(parents=True)
            policy.save(generation / "model.pt")
            metadata = {
                "rows": [{}],
                "ticks": 10,
                "condition_rng": random.Random(7).getstate(),
            }
            (generation / "state.json").write_text(json.dumps(metadata))
            files = {
                name: hashlib.sha256((generation / name).read_bytes()).hexdigest()
                for name in ("state.json", "model.pt")
            }
            pointer.write_text(
                json.dumps({"generation": generation.name, "files": files})
            )
        else:
            marker = json.loads(pointer.read_text())
            model = continuation / marker["generation"] / "model.pt"
            expected = learner_state.load(model, options)
            restored = BaseDQN(options, 999)
            restored.load(model)
            _equal_dqn_values(expected, restored.get_state())
            assert restored.updates > 0
            assert sum(map(len, restored.replay.values())) > 0
            prior = restored.updates
            for tick in range(3):
                restored.choose(_packed_request(tick))
            restored.update([-2.0] * 3, terminal=True)
            assert restored.updates > prior
            control.write_json(
                root / "resumed-dqn.json",
                {
                    "prior_updates": prior,
                    "updates": restored.updates,
                    "retry_once": retry_once,
                },
            )
            status = "completed"
    finally:
        control.write_json(root / "batch.json", {"status": status})
        scope.finish(status)


def test_failed_retry_real_dqn_checkpoint_and_next_update(
    scheduler_factory, monkeypatch
):
    """Real spawned DQN restore through Retry; fixture author plans, not official learner session."""
    from types import SimpleNamespace

    from smartsom.config.codec import digest
    from smartsom.experiments import author_batch, batch_retry, reservation

    monkeypatch.setattr(runtime, "_RETRY_ENABLED", True)
    observer = SimpleNamespace(
        snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
    )
    scheduler = scheduler_factory(1, 1, keep_open=True, observer=observer)
    scheduler.worker = _controlled_dqn_retry_worker
    root = scheduler.root
    child = root / "t0"
    (child / "inputs").mkdir()
    (child / "inputs/experiment.yaml").write_text("run:\n  seed: 4103\n")
    (child / "trial.lock").touch()
    child_plan = {"seed": 4103}
    plan = {
        "files": [
            {
                "id": "t0",
                "child": "t0",
                "task": "train",
                "child_plan_sha256": digest(child_plan),
            }
        ]
    }
    monkeypatch.setattr(author_batch, "load", lambda owner: (owner, plan, {}))
    monkeypatch.setattr(reservation, "load", lambda path: child_plan)
    _until(scheduler, lambda _: scheduler.jobs["t0"]["status"] == "failed")
    before = (child / "continuation/current.json").read_bytes()
    preview = batch_retry.preview_failed(root, ["t0"], include_paused=True)
    assert preview["targets"][0]["check"] == "eligible_for_coordinator_revalidation"
    assert (
        batch_retry.submit_preview(root, preview)["outcomes"][0]["status"] == "received"
    )
    _until(scheduler, lambda _: scheduler.jobs["t0"]["status"] == "completed")
    result = json.loads((child / "resumed-dqn.json").read_text())
    assert (
        result["retry_once"] is True and result["updates"] > result["prior_updates"] > 0
    )
    assert (child / "continuation/current.json").read_bytes() == before
    assert len(scheduler.jobs["t0"]["retry_history"]) == 1
