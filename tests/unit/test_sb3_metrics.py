"""Undefined variance is distinct from invalid learner state or loss."""

import importlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
sb3_update_metrics = importlib.import_module(
    "smartsom.learning.production"
).sb3_update_metrics


def test_constant_finite_targets_mark_explained_variance_unavailable():
    metrics = {"train/explained_variance": float("nan"), "train/loss": 0.5}
    actual = sb3_update_metrics(metrics, np.zeros(4), np.arange(4.0))
    assert actual == {"train/loss": 0.5, "train/explained_variance_defined": 0}
    assert np.isnan(metrics["train/explained_variance"])


@pytest.mark.parametrize(
    "returns,values",
    [([0, 1], [0, 0]), ([0, float("nan")], [0, 0]), ([0, 0], [0, float("inf")])],
)
def test_invalid_variance_or_state_is_not_suppressed(returns, values):
    actual = sb3_update_metrics(
        {"train/explained_variance": float("nan")},
        np.array(returns),
        np.array(values),
    )
    assert np.isnan(actual["train/explained_variance"])
    assert "train/explained_variance_defined" not in actual


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_invalid_loss_remains_visible_even_with_constant_targets(loss):
    actual = sb3_update_metrics(
        {"train/explained_variance": float("nan"), "train/loss": loss},
        np.zeros(4),
        np.zeros(4),
    )
    assert not np.isfinite(actual["train/loss"])


def test_finite_variance_diagnostic_is_unchanged():
    metrics = {"train/explained_variance": 0.75}
    assert sb3_update_metrics(metrics, np.arange(4.0), np.arange(4.0)) == metrics
