"""Native metrics use actual losses without changing the learning trajectory."""

import copy
import json
import pickle
from types import SimpleNamespace

import pytest

from smartsom.learning import production_diagnostics as diagnostics


def test_weighted_summary_is_bounded_and_unavailable_is_null():
    learner = SimpleNamespace()
    for _ in range(1000):
        diagnostics._add(learner, {"td_loss": (2.0, 2), "entropy": (0.0, 0)})
    diagnostics._add(learner, {"td_loss": (8.0, 1000)})
    row = diagnostics.summarize(learner.training_diagnostics)
    assert row["metrics"]["td_loss"]["mean"] == 4.0
    assert row["metrics"]["td_loss"]["weight"] == 3000
    assert row["metrics"]["entropy"]["mean"] is None
    assert row["metrics"]["entropy"]["unavailable_minibatches"] == 1000
    assert len(json.dumps(learner.training_diagnostics)) < 400


def test_k3_uses_only_valid_actor_rows_and_detaches():
    torch = pytest.importorskip("torch")
    learner = SimpleNamespace()
    log_ratio = torch.tensor([0.0, 1000.0, 0.5], requires_grad=True)
    active = torch.tensor([1.0, 0.0, 1.0])
    scalar = torch.tensor(2.0, requires_grad=True)
    diagnostics.record_ppo(
        learner, scalar, scalar, scalar, scalar, log_ratio, active, active
    )
    result = diagnostics.summarize(learner.training_diagnostics)["metrics"]
    assert result["approx_kl_k3"]["mean"] == pytest.approx(
        (__import__("math").exp(0.5) - 1.5) / 2
    )
    assert result["approx_kl_k3"]["weight"] == 2
    assert log_ratio.grad is None and scalar.grad is None
    diagnostics.record_ppo(
        learner, scalar, scalar, scalar, scalar, log_ratio, active * 0, active * 0
    )
    assert (
        diagnostics.summarize(learner.training_diagnostics)["metrics"]["entropy"][
            "unavailable_minibatches"
        ]
        == 1
    )


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn", "central_sb3_ppo"])
def test_native_diagnostics_preserve_trajectory_reports_and_resume(
    name, tmp_path, monkeypatch
):
    pytest.importorskip("ray")
    torch = pytest.importorskip("torch")
    if "sb3" in name:
        pytest.importorskip("sb3_contrib")
    from test_composable_learning import tiny

    from smartsom.experiments.composable import TrainingSession, allocate

    prepared = tiny(name, tmp_path, ticks=16)
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    try:
        initial = copy.deepcopy(session.state_dict())
        empty = session.learner_diagnostics()
        assert all(
            m["mean"] is None
            for g in empty["groups"].values()
            for m in g["metrics"].values()
        )
        session.step_update()
        with (root / "checkpoints/update-000001/continuation.pkl").open("rb") as stream:
            midpoint = pickle.load(stream)
        middle = copy.deepcopy(session.learner_diagnostics())
        session.step_update()
        expected = copy.deepcopy(session.learner_diagnostics())
        actions = copy.deepcopy(session.actions)
        counts = copy.deepcopy(session.collector.counts)
        weights = {
            g: copy.deepcopy(p.network.state_dict())
            for g, p in session.policies.items()
            if hasattr(p, "network")
        }
        assert (
            json.loads((root / "run.json").read_text())["learner_diagnostics"]
            == expected
        )
        assert session.history[-1]["learner_diagnostics"] == expected
        assert any(g["observed_minibatches"] > 0 for g in expected["groups"].values())
        for group, row in expected["groups"].items():
            assert row["optimization_steps"] == session.optimizations[group]
            assert row["collector"] == session.collector.counts.get(group, {})
            if name == "train_all_dqn":
                assert row["replay_length"] == len(session.replays[group].rows)
                assert row["target_clock"] == session.target_clock[group]
        session.close()
        session = TrainingSession(prepared, root, record)
        session.restore(midpoint)
        assert session.learner_diagnostics() == middle
        session.step_update()
        assert session.learner_diagnostics() == expected
        session.close()
        session = TrainingSession(prepared, root, record)
        # Pre-diagnostics checkpoints legitimately have no historical metrics.
        initial.pop("learner_diagnostics")
        session.restore(initial)
        monkeypatch.setattr(diagnostics, "record_ppo", lambda *args: None)
        monkeypatch.setattr(diagnostics, "record_dqn", lambda *args: None)
        session.step_update()
        session.step_update()
        assert session.actions == actions
        assert session.collector.counts == counts
        for group, expected_weights in weights.items():
            for key, tensor in expected_weights.items():
                torch.testing.assert_close(
                    session.policies[group].network.state_dict()[key],
                    tensor,
                    rtol=0,
                    atol=0,
                )
        assert all(
            g["observed_minibatches"] == 0
            for g in session.learner_diagnostics()["groups"].values()
        )
    finally:
        session.close()


@pytest.mark.parametrize("value", [-1e-4, 1e-4, -1e-5, 1e-5, 0.0])
def test_k3_near_zero_is_stable_nonnegative(value):
    import math
    from decimal import Decimal, localcontext

    torch = pytest.importorskip("torch")
    learner = SimpleNamespace()
    log_ratio = torch.tensor([value], dtype=torch.float32)
    scalar = torch.tensor(0.0)
    mask = torch.tensor([1.0])
    diagnostics.record_ppo(
        learner, scalar, scalar, scalar, scalar, log_ratio, mask, mask
    )
    result = diagnostics.summarize(learner.training_diagnostics)["metrics"][
        "approx_kl_k3"
    ]["mean"]
    x = float(log_ratio.item())
    assert result >= 0 and math.isfinite(result)
    # An independent reference retains the exact float32 input, rather than
    # repeating the float64 subtraction whose last bits vary across backends.
    with localcontext() as context:
        context.prec = 80
        exact = Decimal.from_float(x)
        reference = float(exact.exp() - 1 - exact)
    # expm1(x) is O(x), while k3 is O(x**2). Budget two ULPs at the
    # expm1 scale plus one at the result scale for subtraction/reference
    # rounding; a fixed relative tolerance on k3 ignores this cancellation.
    error_budget = 2 * math.ulp(math.expm1(x)) + math.ulp(reference)
    assert result == pytest.approx(reference, rel=0, abs=error_budget)
    if x:
        assert result > 0  # Clamping the old negative float32 result is wrong too.
    else:
        assert result == 0


def test_k3_nonfinite_is_unavailable():
    torch = pytest.importorskip("torch")
    learner = SimpleNamespace()
    scalar = torch.tensor(0.0)
    mask = torch.tensor([1.0])
    diagnostics.record_ppo(
        learner, scalar, scalar, scalar, scalar, torch.tensor([1000.0]), mask, mask
    )
    result = diagnostics.summarize(learner.training_diagnostics)["metrics"][
        "approx_kl_k3"
    ]
    assert result == {"mean": None, "weight": 0, "unavailable_minibatches": 1}
