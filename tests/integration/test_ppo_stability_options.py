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
    return algorithm_options(**{"epochs": 3, "batch_size": 16, **params})


def record_step_rates(policy):
    """Record the learning rate in effect at every optimiser step."""
    rates = []
    for optimizer in policy.optimizers.values():
        step = optimizer.step

        def recorded(*args, _optimizer=optimizer, _step=step, **kwargs):
            rates.append(_optimizer.param_groups[0]["lr"])
            return _step(*args, **kwargs)

        optimizer.step = recorded
    return rates


def run_updates(policy, config, seeds):
    for seed in seeds:
        report, rewards, critic = episode(
            base_factory("small"), WORKLOAD, config, policy, seed, "normal", 1000
        )
        policy.update(
            rewards, terminal=report["status"] == "completed", final_critic=critic
        )


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
        n
        for loss in losses
        for g in loss.values()
        for n in (g["epochs_run"]["min"], g["epochs_run"]["max"])
    ]
    # Each minibatch measures KL before its own step, so a one-minibatch first
    # epoch has zero KL; the stop then triggers after at most two epochs.
    assert runs and max(runs) < 3
    _, losses = trained(options(target_kl=1e9))
    runs = [
        n
        for loss in losses
        for g in loss.values()
        for n in (g["epochs_run"]["min"], g["epochs_run"]["max"])
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


def test_rate_in_effect_at_every_optimizer_step():
    from smartsom.learning.ppo.learner import BasePPO

    config = options(learning_rate=0.001, lr_final_fraction=0.1, lr_decay_updates=2)
    policy = BasePPO(config, 1)
    rates = record_step_rates(policy)
    for n, expected in enumerate((0.001, 0.00055, 0.0001, 0.0001)):
        rates.clear()
        run_updates(policy, config, [123 + n])
        assert rates, n
        assert rates == pytest.approx([expected] * len(rates)), n


def test_mid_decay_resume_applies_the_uninterrupted_rate(tmp_path):
    from smartsom.learning.ppo.learner import BasePPO

    config = options(learning_rate=0.001, lr_final_fraction=0.1, lr_decay_updates=4)
    straight = BasePPO(config, 1)
    run_updates(straight, config, [123, 124])
    straight.save(tmp_path / "mid.pt")
    expected = record_step_rates(straight)
    run_updates(straight, config, [125])
    resumed = BasePPO(config, 7)
    resumed.load(tmp_path / "mid.pt")
    actual = record_step_rates(resumed)
    run_updates(resumed, config, [125])
    assert actual and expected
    rate = 0.001 * (1 - 0.5 * 0.9)
    assert actual == pytest.approx([rate] * len(actual))
    assert expected == pytest.approx([rate] * len(expected))


def test_kl_stop_after_first_epoch_with_several_minibatches():
    _, losses = trained(options(batch_size=4, target_kl=1e-12))
    runs = [
        n
        for loss in losses
        for g in loss.values()
        for n in (g["epochs_run"]["min"], g["epochs_run"]["max"])
    ]
    # Later minibatches of epoch 1 follow optimiser steps, so their KL is positive.
    assert 1 in runs


def test_clock_does_not_advance_without_an_optimizer_step():
    from smartsom.learning.ppo.learner import BasePPO

    config = options(lr_final_fraction=0.5, lr_decay_updates=4)
    policy = BasePPO(config, 1)
    policy.ownership.is_frozen = lambda key: True
    run_updates(policy, config, [123])
    assert policy.updates == 0 and policy.update_calls == 0


def test_per_role_override_cannot_enable_ppo_options_on_dqn():
    data = algorithm_options(method="dqn").model_dump(mode="json", by_alias=True)
    data["production"]["policies"]["dqn"]["role_params"] = {
        "machine": {"target_kl": 0.02}
    }
    with pytest.raises(ValueError, match="PPO settings"):
        BaseAlgorithm.model_validate(data)


def test_epochs_run_is_a_group_summary_without_entity_keys():
    data = options(target_kl=1e9).model_dump(mode="json", by_alias=True)
    for spec in data["production"]["policies"].values():
        spec["parameter_sharing"] = {"roles": {"machine": "independent"}}
    independent = BaseAlgorithm.model_validate(data)
    # The compiler normally attaches these stable physical identities.
    independent._owner_ids = {
        "machine": tuple(m.machine_id for m in base_factory("small").machines)
    }
    policy, losses = trained(independent)
    owner = independent._owner_ids["machine"][0].encode().hex()
    assert any(key.endswith(owner) for key in policy.optimizers)
    for loss in losses:
        for g in loss.values():
            assert g["epochs_run"] == {"min": 3, "max": 3}
