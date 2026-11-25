"""PPO learning-rate decay and KL early stop: default-off learner options."""

import pytest

from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    BaseWorkload,
    LearnerParameters,
)
from smartsom.config.base_layout import base_factory
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options

torch = pytest.importorskip("torch")

NEW_KEYS = ("lr_final_fraction", "lr_decay_updates", "target_kl")
WORKLOAD = BaseWorkload(
    schema="smartsom.base-workload/v1",
    orders=8,
    initial_orders=4,
    mean_interarrival=4,
    base_ticks=dict.fromkeys("ABCD", 2),
)


def without_new_keys(options):
    """The same algorithm document with the new learner keys absent."""
    data = options.model_dump(mode="json", by_alias=True)
    for spec in data["production"]["policies"].values():
        for key in NEW_KEYS:
            del spec["learner"]["params"][key]
    return BaseAlgorithm.model_validate(data)


def trained(options, updates=2):
    from smartsom.learning.ppo.learner import BasePPO

    policy = BasePPO(options, 1)
    losses = []
    for seed in range(123, 123 + updates):
        report, rewards, critic = episode(
            base_factory("small"), WORKLOAD, options, policy, seed, "normal", 1000
        )
        losses.append(
            policy.update(
                rewards, terminal=report["status"] == "completed", final_critic=critic
            )
        )
    return policy, losses


def options(**params):
    return algorithm_options(epochs=3, batch_size=16, **params)


def assert_same_parameters(left, right):
    state = right.models.state_dict()
    for key, value in left.models.state_dict().items():
        assert torch.equal(value, state[key]), key


def test_defaults_and_inactive_options_train_bit_identically():
    absent = without_new_keys(options())
    assert "target_kl" not in absent.learning_policy.learner.params
    plain, plain_losses = trained(absent)
    for params in (
        {},
        {"target_kl": 1e9},
        {"lr_final_fraction": 1.0, "lr_decay_updates": 5},
    ):
        other, losses = trained(options(**params))
        assert_same_parameters(plain, other)
        assert other.updates == plain.updates
    assert all(
        "learning_rate" not in g and "epochs_run" not in g
        for loss in plain_losses
        for g in loss.values()
    )


def test_lr_decay_reaches_final_fraction_and_stays():
    policy, losses = trained(
        options(learning_rate=0.001, lr_final_fraction=0.1, lr_decay_updates=4), 6
    )
    per_update = [{g["learning_rate"] for g in loss.values()} for loss in losses]
    assert all(len(values) == 1 for values in per_update)
    rates = [values.pop() for values in per_update]
    expected = [0.001 * (1 - min(1, n / 4) * 0.9) for n in range(6)]
    assert rates == pytest.approx(expected)
    assert rates[4:] == pytest.approx([0.0001, 0.0001])
    assert policy.update_calls == 6
    for optimizer in policy.optimizers.values():
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)


def test_kl_early_stop_skips_remaining_epochs_only_when_triggered():
    _, losses = trained(options(target_kl=1e-12))
    runs = [
        n for loss in losses for g in loss.values() for n in g["epochs_run"].values()
    ]
    # Each minibatch measures KL before its own step, so a one-minibatch first
    # epoch has zero KL; the stop then triggers after at most two epochs.
    assert runs and max(runs) < 3
    _, losses = trained(options(target_kl=1e9))
    runs = [
        n for loss in losses for g in loss.values() for n in g["epochs_run"].values()
    ]
    assert set(runs) == {3}


def test_schedule_clock_survives_checkpoint_and_old_checkpoints_load(tmp_path):
    from smartsom.learning.ppo.learner import BasePPO

    config = options(lr_final_fraction=0.5, lr_decay_updates=10)
    policy, _ = trained(config)
    policy.save(tmp_path / "last.pt")
    restored = BasePPO(config, 9)
    restored.load(tmp_path / "last.pt")
    assert restored.update_calls == policy.update_calls == 2
    old = policy.get_state()
    del old["update_calls"]
    legacy = BasePPO(config, 9)
    legacy.set_state(old)
    assert legacy.update_calls == 0


@pytest.mark.parametrize(
    "params",
    [
        {"lr_final_fraction": 0.1},
        {"lr_decay_updates": 10},
        {"lr_final_fraction": 0.0, "lr_decay_updates": 10},
        {"lr_final_fraction": 1.5, "lr_decay_updates": 10},
        {"lr_final_fraction": 0.5, "lr_decay_updates": 0},
        {"target_kl": 0.0},
        {"target_kl": -0.01},
        {"target_kl": float("inf")},
    ],
)
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        LearnerParameters(**params)


@pytest.mark.parametrize(
    "params", [{"target_kl": 0.02}, {"lr_final_fraction": 0.5, "lr_decay_updates": 3}]
)
def test_options_are_ppo_only(params):
    with pytest.raises(ValueError, match="PPO settings"):
        algorithm_options(method="dqn", **params)
