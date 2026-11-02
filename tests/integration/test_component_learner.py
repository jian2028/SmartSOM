"""A file-defined learner uses real shared sampling, updates and full restore."""

from pathlib import Path

import pytest
import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation_runtime import episode
from tests.unit.test_algorithm_components import author, recipe

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("changed", ["workload.yaml", "../execution.json"])
def test_prepared_training_inputs_retain_frozen_content_checks(
    tmp_path, monkeypatch, changed
):
    from smartsom.experiments import reservation, reservation_session

    path = author(tmp_path, recipe("dqn"))
    plan = compile_reservation(
        path,
        task="train",
        sets=["experiment.train_ticks=2", "experiment.drain_frozen=true"],
    )
    root = reservation.allocate(plan, root=tmp_path / "run")
    args, _ = reservation.arguments(root)

    def repeated_parse(*args):
        raise AssertionError("prepared input was reparsed")

    monkeypatch.setattr(reservation_session, "load_inputs", repeated_parse)
    session = reservation_session.ReservationSession(args)
    session.close()
    workload = root / "inputs" / changed
    workload.write_text(workload.read_text() + "\n")
    with pytest.raises(ValueError, match="prepared input identity changed"):
        reservation_session.ReservationSession(args)


def test_replay_checkpoint_retains_all_fields_precision_and_shared_requests(tmp_path):
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
    from smartsom.learning.dqn.replay import CheckpointReplay

    request = BaseDecision(
        7,
        "machine",
        "owner",
        "process",
        (BaseCandidate("rule", ("job", "machine"), vector(1.0000000000000002)),),
        vector(0.12345678901234567),
        ((), (), (), ()),
    )
    raw = {
        "machine": [
            {
                "request": request,
                "next": request,
                "index": 0,
                "reward": 0.12345678901234567,
                "physical_dt": 0,
            }
        ]
    }
    path = tmp_path / "state.pt"
    torch.save({"replay": CheckpointReplay(raw)}, path, pickle_protocol=5)
    restored = torch.load(path, weights_only=False)["replay"]
    assert type(restored) is dict
    assert restored == raw
    assert restored["machine"][0]["request"] is restored["machine"][0]["next"]
    assert (
        restored["machine"][0]["request"].candidates[0].features
        == request.candidates[0].features
    )


def test_custom_observation_replay_keeps_torch_serialization(monkeypatch):
    from types import SimpleNamespace

    from tests.helpers.algorithm import algorithm_options

    learner = build_policy(algorithm_options(method="dqn"), 1).learner
    monkeypatch.setattr(
        learner, "options", SimpleNamespace(observation=SimpleNamespace(target=None))
    )
    assert learner.get_state()["replay"] is learner.replay


def test_checkpoint_reads_old_pickle_and_preserves_complete_state(tmp_path):
    import random

    from smartsom.learning import state

    class Options:
        def science_identity(self):
            return {"algorithm": "test-complete-state"}

    options = Options()
    tensor = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    saved = {
        "models": tensor,
        "targets": tensor.clone(),
        "adam": {"step": torch.tensor(3), "exp_avg": tensor / 2},
        "replay": [{"reward": 0.125, "action": ("工件", 2), "next": None}],
        "rng": random.Random(42).getstate(),
        "generator": torch.Generator().manual_seed(42).get_state(),
    }
    current, previous = tmp_path / "current.pt", tmp_path / "previous.pt"
    state.save(current, options, saved)
    torch.save(
        {
            "contract": "smartsom.component-checkpoint/v2",
            "identity": options.science_identity(),
            "state": saved,
        },
        previous,
        pickle_protocol=2,
    )
    for path in (current, previous):
        restored = state.load(path, options)
        for name in ("models", "targets", "generator"):
            assert torch.equal(restored[name], saved[name])
        assert torch.equal(restored["adam"]["exp_avg"], saved["adam"]["exp_avg"])
        assert restored["adam"]["step"] == saved["adam"]["step"]
        assert restored["replay"] == saved["replay"]
        assert restored["rng"] == saved["rng"]


def test_custom_learner_factory_updates_and_restores(tmp_path):
    data = recipe("ppo")
    data["production"]["policies"]["ppo"]["learner"] = {
        "path": str(ROOT / "examples/current/learner.py"),
        "factory": "create_learner",
        "params": {"epochs": 1},
    }
    path = author(tmp_path, data)
    plan = compile_reservation(
        path,
        sets=[
            "workload.orders=8",
            "workload.initial_orders=4",
            "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        ],
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    _, factory, workload, options, _ = load_inputs(frozen / "experiment.yaml")
    policy = build_policy(options, 8)
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        9,
        "normal",
        500,
        max_ticks=100,
        metrics_level="off",
    )
    policy.update(rewards, terminal=False, final_critic=critic)
    assert policy.last_losses["custom"]["batch_ticks"] == report["tick"]
    assert policy.updates > 0
    file = tmp_path / "model.pt"
    policy.save(file)
    restored = build_policy(options, 99)
    restored.load(file)
    assert restored.updates == policy.updates
    assert all(
        torch.equal(v, restored.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )
    # Sampler transfer is explicit; no runtime dependency on a .models attribute.
    sampler = build_policy(options, 1, training=False)
    sampler.learner.load_sampling_state(policy.learner.sampling_state())
    sampler.seed_sampling(123)
    assert all(
        torch.equal(v, sampler.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )
