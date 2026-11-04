"""Short isolated Retry roundtrip through the real official learner session."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from smartsom.experiments import author_batch, control, queue_admission
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.launch import compile_launch
from smartsom.experiments.runtime_workers import NativeScheduler
from smartsom.experiments.tuning_resources import ResourceSnapshot
from tests.integration.test_runtime_scheduler import _until
from tests.unit.test_algorithm_components import recipe
from tests.unit.test_incremental_batch import batch


def _inject_official_failure(directory, parent_root, retry_once=False):
    """Inject once after a real committed episode; run the actual official worker."""
    from smartsom.experiments import reservation_runtime

    original = reservation_runtime.episode
    calls = 0

    def bounded_episode(*args, **kwargs):
        nonlocal calls
        calls += 1
        if not retry_once and calls == 2:
            raise RuntimeError("fixture failure after first committed episode")
        kwargs["max_ticks"] = min(kwargs.get("max_ticks") or 40, 40)
        return original(*args, **kwargs)

    reservation_runtime.episode = bounded_episode
    if retry_once:
        import torch

        from smartsom.experiments import reservation_session
        from tests.integration.test_rule_learning import _equal_dqn_values

        original_init = reservation_session.ReservationSession.__init__

        def checked_init(session, args):
            original_init(session, args)
            pointer = json.loads(
                (Path(directory) / "continuation/current.json").read_text()
            )
            generation = Path(directory) / "continuation" / pointer["generation"]
            expected = torch.load(
                generation / "model.pt", map_location="cpu", weights_only=False
            )["state"]
            _equal_dqn_values(expected, session.policy.get_state())
            saved = json.loads((generation / "state.json").read_text())
            assert session.rng.getstate() == reservation_session.tuples(
                saved["condition_rng"]
            )
            control.write_json(
                Path(directory) / "fixture-restored.json",
                {"ticks": session.ticks, "state_equal": True},
            )

        reservation_session.ReservationSession.__init__ = checked_init
    # One attempt only for the initial fault and for the user's manual retry.
    return author_batch._quiet_worker(directory, parent_root, retry_once=True)


def test_retry_runs_official_dqn_session_from_committed_checkpoint(
    tmp_path, monkeypatch
):
    import torch

    folder = batch(tmp_path)
    algorithm = recipe("dqn")
    params = algorithm["production"]["policies"]["dqn"]["learner"]["params"]
    params.update(width=8, batch_size=2, epochs=1)
    (folder / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    source = folder / "medium-base.yaml"
    document = yaml.safe_load(source.read_text())
    document.update(train_ticks=120, final_evaluation=False)
    document["run"]["seconds"] = 120
    source.write_text(yaml.safe_dump(document))
    compiled = compile_launch(
        folder,
        ids=["medium-base"],
        first_train_seeds=1,
        operations={"output_root": str(tmp_path / "results")},
    )
    root = Path(author_batch.allocate(compiled))
    _, plan, state = author_batch.load(root)
    key = state["selection"]["tasks"][0]
    row = next(item for item in plan["files"] if item["id"] == key)
    child = root / row["child"]
    monkeypatch.setattr(runtime, "_RETRY_ENABLED", True)
    scope = control.Scope()
    token = control.CURRENT.set(scope)
    scheduler = None
    try:
        scope.bind(root)
        author_batch._runtime_initialize(root, plan, state, limit=1, keep_open=True)
        queue_admission.initialize(root, supported=True)
        observer = SimpleNamespace(
            snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
        )
        scheduler = NativeScheduler(
            root,
            {key: {"root": str(child), "status": "queued"}},
            worker=_inject_official_failure,
            observer=observer,
        )
        _until(scheduler, lambda _: scheduler.jobs[key]["status"] == "failed")
        before = json.loads((child / "continuation/current.json").read_text())
        generation = child / "continuation" / before["generation"]
        saved = json.loads((generation / "state.json").read_text())
        assert len(saved["rows"]) == 1 and saved["ticks"] == 40
        model_path = generation / "model.pt"
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        assert checkpoint["state"]["learner"]["updates"] > 0
        assert any(
            checkpoint["state"]["learner"]["optimizers"][group]["state"]
            for group in checkpoint["state"]["learner"]["optimizers"]
        )
        from smartsom.experiments import batch_retry

        preview = batch_retry.preview_failed(root, [key], include_paused=True)
        assert preview["targets"][0]["continuation"]["ticks"] == 40
        assert (
            batch_retry.submit_preview(root, preview)["outcomes"][0]["status"]
            == "received"
        )
        _until(
            scheduler,
            lambda _: scheduler.jobs[key]["status"] == "completed",
            timeout=60,
        )
        after = json.loads((child / "continuation/current.json").read_text())
        final = json.loads(
            (child / "continuation" / after["generation"] / "state.json").read_text()
        )
        assert after["generation"] != before["generation"] and final["ticks"] == 120
        assert final["rows"][0] == saved["rows"][0] and len(final["rows"]) == 3
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
        attempts = json.loads((child / "attempts.json").read_text())
        assert len(attempts) == 1 and "fixture failure" in attempts[0]["message"]
        assert len(scheduler.jobs[key]["retry_history"]) == 1
        assert json.loads((child / "fixture-restored.json").read_text()) == {
            "ticks": 40,
            "state_equal": True,
        }
    finally:
        if scheduler is not None:
            scheduler.close()
        scope.finish("completed")
        control.CURRENT.reset(token)
