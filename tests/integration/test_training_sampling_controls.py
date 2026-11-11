"""Current public compilation and native sampling/update acceptance."""

import copy
import json

import pytest
import torch
import yaml

from smartsom.config.base_diagnostic import LearnerParameters
from smartsom.learning.update_schedule import UpdateClock
from tests.unit.test_algorithm_components import author, recipe


def test_update_clocks_count_new_experience_and_preserve_fractional_credit():
    settings = LearnerParameters(
        update_trigger="transitions",
        update_every=4,
        update_amount="ratio",
        updates_per_transition=0.3,
    )
    clock = UpdateClock()
    assert clock.advance(3, settings, episode_end=False) == 0
    assert clock.advance(1, settings, episode_end=False) == 1
    restored = UpdateClock.restore(clock.state())
    assert restored.advance(4, settings, episode_end=False) == 1
    assert restored.advance(4, settings, episode_end=False) == 1
    assert restored.advance(4, settings, episode_end=False) == 1
    assert restored.advance(4, settings, episode_end=False) == 2
    assert clock.inserted == 4


def test_warmup_and_exploration_use_per_network_insertions():
    settings = LearnerParameters(
        warmup_transitions=4,
        exploration="linear",
        epsilon=1,
        epsilon_final=0.1,
        epsilon_decay_transitions=10,
    )
    first, second = UpdateClock(), UpdateClock()
    assert first.advance(3, settings, episode_end=True) == 0
    assert second.epsilon(settings) == 1
    assert first.advance(1, settings, episode_end=True) == 3
    assert first.epsilon(settings) == pytest.approx(0.64)
    assert second.inserted == 0


def prepare(
    tmp_path, method, *, processes=0, trigger="episode", independent=False, ratio=False
):
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation

    document = recipe(method)
    document["production"]["policies"][method]["learner"]["params"].update(
        update_trigger=trigger,
        update_every=2,
        epochs=1,
        batch_size=8,
    )
    definition = document["production"]["policies"][method]
    if independent:
        definition["parameter_sharing"] = {"default": "independent"}
    if ratio:
        definition["learner"]["params"].update(
            update_amount="ratio",
            updates_per_transition=0.3,
            exploration="linear",
            epsilon=1.0,
            epsilon_final=0.1,
            epsilon_decay_transitions=100,
        )
    path = author(tmp_path, document)
    ex = yaml.safe_load(path.read_text())
    ex.update(
        train_ticks=256,
        drain_frozen=True,
        final_evaluation=False,
        sampling=dict(num_envs=processes or 1, processes=processes, fragment_ticks=4),
    )
    path.write_text(yaml.safe_dump(ex))
    plan = compile_reservation(path, task="train")
    root = reservation.allocate(plan, root=tmp_path / "run")
    args, _ = reservation.arguments(root)
    return args, root


@pytest.mark.parametrize(
    "method,trigger",
    [
        ("dqn", "transitions"),
        ("ppo", "fragment"),
        ("dqn", "episode"),
        ("ppo", "episode"),
    ],
)
@pytest.mark.parametrize("processes", [0, 2])
def test_current_session_sampling_and_complete_restore(
    tmp_path, method, trigger, processes
):
    from smartsom.experiments.reservation_session import ReservationSession

    args, root = prepare(tmp_path, method, processes=processes, trigger=trigger)
    session = ReservationSession(args)
    try:
        while not session.step()["done"]:
            pass
        assert session.ticks == 256
        assert len(session.rows) == (2 if processes else 1)
        assert session.policy.updates > 0
        expected = copy.deepcopy(session.policy.get_state())
        updates = session.policy.updates
        if processes:
            assert len(expected["sampler_states"]) == 2
            assert all(p.is_alive() for _, p in session.samplers.workers)
    finally:
        session.close()
    restored = ReservationSession(args)
    try:
        assert restored.ticks == 256
        assert restored.policy.updates == updates
        actual = restored.policy.get_state()
        for key, tensor in expected["learner"]["models"].items():
            assert torch.equal(actual["learner"]["models"][key], tensor)
        assert restored.step()["done"]
    finally:
        restored.close()
    assert json.loads((root / "continuation/current.json").read_text())["files"]


@pytest.mark.parametrize("amount", ["fixed", "ratio"])
def test_warmup_threshold_is_independent_of_packet_partition(amount):
    settings = LearnerParameters(
        warmup_transitions=4,
        update_trigger="transitions",
        update_every=1,
        update_amount=amount,
        epochs=1,
        updates_per_transition=1,
    )
    one, split = UpdateClock(), UpdateClock()
    assert one.advance(4, settings, episode_end=False) == 1
    assert sum(split.advance(n, settings, episode_end=False) for n in (3, 1)) == 1
    assert one.state() == split.state()


def test_sparse_dqn_tails_wait_for_real_next_request():
    from types import SimpleNamespace

    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
    from smartsom.domain.training import RewardTimeline
    from smartsom.learning.dqn.replay import transitions
    from smartsom.learning.sampling import take_rollout

    def row(tick, owner):
        return {
            "request": BaseDecision(
                tick, "machine", owner, "process", (BaseCandidate("j", ("j",)),)
            ),
            "index": 0,
        }

    learner = SimpleNamespace(
        records=[row(0, "a"), row(1, "b"), row(4, "a")], needs_next_request=True
    )
    learner.discard_rollout = lambda: setattr(learner, "records", [])
    policy = SimpleNamespace(
        learner=learner,
        options=SimpleNamespace(learning_policy=SimpleNamespace(split_stages=())),
    )
    rewards = RewardTimeline()
    rewards.extend([1.0] * 5)
    records, timeline, _, _ = take_rollout(
        policy, rewards, None, terminal=False, episode_end=False
    )
    pending, omitted = transitions(
        records, timeline, terminal=False, scale={"machine": 1.0}
    )
    assert omitted == 0
    assert [
        (r["request"].owner, r["reward"], r["next"].tick) for r in pending["machine"]
    ] == [("a", 4, 4)]
    learner.records += [row(8, "a"), row(9, "b")]
    rewards.extend([1.0] * 5)
    records, timeline, _, _ = take_rollout(
        policy, rewards, None, terminal=False, episode_end=False
    )
    pending, omitted = transitions(
        records, timeline, terminal=False, scale={"machine": 1.0}
    )
    assert omitted == 0
    assert sorted(
        (r["request"].owner, r["reward"], r["next"].tick) for r in pending["machine"]
    ) == [("a", 4, 8), ("b", 8, 9)]


def test_ppo_returns_do_not_join_matching_subjects_from_different_worlds():
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
    from smartsom.domain.training import RewardTimeline
    from tests.helpers.algorithm import algorithm_options

    policy = build_policy(algorithm_options(method="ppo"), 101)
    learner = policy.learner
    request = BaseDecision(
        0, "machine", "same-owner", "process", (BaseCandidate("job", ("job",)),)
    )
    packets = []
    for reward in (1.0, 100.0):
        timeline = RewardTimeline()
        timeline.extend([reward])
        packets.append(([{"request": request, "value": 0.0}], timeline, True, None))
    captured = []
    learner.update = lambda *a, **kw: captured.extend(learner.records)
    learner.update_rollouts(packets)
    assert [row["return"] for row in captured] == [1.0, 100.0]


def assert_state_equal(left, right):
    from smartsom.learning.dqn.replay import PackedRows

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple, PackedRows)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "method,trigger", [("dqn", "transitions"), ("ppo", "fragment")]
)
def test_resume_continues_second_wave_identically(tmp_path, method, trigger):
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation
    from smartsom.experiments.reservation_session import ReservationSession

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _, _ = prepare(inputs, method, processes=2, trigger=trigger)
    path = inputs / "experiment.yaml"
    ex = yaml.safe_load(path.read_text())
    ex.update(
        train_ticks=512,
        periodic_evaluation={
            "unit": "physical_ticks",
            "milestones": [256],
            "selection": "last",
        },
    )
    path.write_text(yaml.safe_dump(ex))
    plan = compile_reservation(path, task="train")
    results = []
    for restart in (False, True):
        root = reservation.allocate(plan, root=tmp_path / str(restart))
        args, _ = reservation.arguments(root)
        session = ReservationSession(args)
        # The milestone is only a durable wave boundary here, not a panel test.
        session.evaluate_periodically = lambda: None
        try:
            assert not session.step()["done"]
            assert session.ticks == 256
            if restart:
                session.close()
                # Avoid invoking the unrelated panel on restoration.
                import unittest.mock

                with unittest.mock.patch.object(
                    ReservationSession, "evaluate_periodically", lambda self: None
                ):
                    session = ReservationSession(args)
                session.evaluate_periodically = lambda: None
            assert session.step()["done"]
            results.append(copy.deepcopy(session.policy.get_state()))
        finally:
            session.close()
    assert_state_equal(results[0], results[1])


@pytest.mark.parametrize("worker_deadline", [False, True])
def test_parallel_wall_budget_rolls_back_without_failed_training(
    tmp_path, monkeypatch, worker_deadline
):
    import time

    from smartsom.experiments import training_sampling
    from smartsom.experiments.reservation_session import ReservationSession

    args, root = prepare(tmp_path, "dqn", processes=2, trigger="transitions")
    session = ReservationSession(args)
    generation = (root / "continuation/current.json").read_bytes()
    expected = copy.deepcopy(session.policy.get_state())
    workers = []
    original = training_sampling.Samplers.__init__

    def capture(self, *args):
        original(self, *args)
        workers.extend(p for _, p in self.workers)

    monkeypatch.setattr(training_sampling.Samplers, "__init__", capture)
    if worker_deadline:

        def receive_worker(self, index, session):
            connection, _ = self.workers[index]
            assert connection.poll(30)
            kind, value = training_sampling._receive(connection)
            assert kind == "timeout", (kind, value)
            raise TimeoutError("worker wall budget")

        monkeypatch.setattr(training_sampling.Samplers, "receive", receive_worker)
    session.deadline = time.time() + 0.01
    try:
        result = session.step()
        assert result["done"]
        assert result["status"] == "wall_budget_reached"
        assert (root / "continuation/current.json").read_bytes() == generation
        assert_state_equal(session.policy.get_state(), expected)
        assert workers and all(not p.is_alive() for p in workers)
    finally:
        session.close()


def test_frozen_only_ppo_fragment_does_not_require_an_optimizer_step():
    from types import SimpleNamespace

    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
    from smartsom.learning.ppo.learner import BasePPO

    learner = BasePPO.__new__(BasePPO)
    learner.groups = ("dispatcher",)
    learner.ownership = SimpleNamespace(
        options={"dispatcher": LearnerParameters()},
        keys={"dispatcher": ("pickup", "delivery")},
        is_frozen=lambda key: key == "pickup",
        key=lambda request: request.stage,
    )
    learner.options = SimpleNamespace(
        learning_policy=SimpleNamespace(split_stages=("dispatcher",))
    )
    learner.records = [
        {
            "request": BaseDecision(
                0, "dispatcher", "v", "pickup", (BaseCandidate("j", ("j",)),)
            ),
            "raw_advantage": 1.0,
            "return": 1.0,
        }
    ]
    result = learner.update(None, terminal=False, final_critic=None, prepared=True)
    assert result["dispatcher"]["decisions"] == 1
    assert learner.records == []


def _crashing_worker(connection, options, seed, index):
    from smartsom.experiments.training_sampling import _receive, _send

    _receive(connection)
    _send(connection, ("error", "injected worker failure"))
    connection.close()


def test_worker_error_is_not_hidden_and_children_exit(tmp_path, monkeypatch):
    from smartsom.experiments import training_sampling
    from smartsom.experiments.reservation_session import ReservationSession

    args, root = prepare(tmp_path, "dqn", processes=2)
    session = ReservationSession(args)
    original = training_sampling.Samplers.__init__
    workers = []

    def capture(self, *args):
        original(self, *args)
        workers.extend(p for _, p in self.workers)

    monkeypatch.setattr(training_sampling.Samplers, "__init__", capture)
    monkeypatch.setattr(training_sampling, "_worker", _crashing_worker)
    generation = (root / "continuation/current.json").read_bytes()
    try:
        with pytest.raises(RuntimeError, match="injected worker failure"):
            session.step()
        assert workers and all(not p.is_alive() for p in workers)
        assert (root / "continuation/current.json").read_bytes() == generation
    finally:
        session.close(failed=True)


def test_first_recording_is_selected_per_condition_in_parallel(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.experiments import training_sampling
    from smartsom.experiments.reservation_session import ReservationSession

    args, _ = prepare(tmp_path, "ppo", processes=2)
    session = ReservationSession(args)
    requests = []
    # Two prior A episodes must not suppress the first B recording.
    session.rows = [{"condition": "normal"}, {"condition": "normal"}]
    session.rng = SimpleNamespace(choice=lambda _: "high")
    session.ex = session.ex.model_copy(
        update={"replay": session.ex.replay.model_copy(update={"train": "first"})}
    )

    class Samplers:
        workers = [(None, None), (None, None)]

        def receive(self, index, session):
            raise RuntimeError("stop after dispatch")

        def close(self):
            pass

    session.samplers = Samplers()
    monkeypatch.setattr(
        training_sampling, "_send", lambda connection, value: requests.append(value)
    )
    try:
        with pytest.raises(RuntimeError, match="stop after dispatch"):
            session.step()
        assert requests[0][3]["replay_directory"] is not None
        assert requests[1][3]["replay_directory"] is None
    finally:
        session.close()


def test_evaluation_keeps_sampler_memory_before_during_and_after(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.experiments import phase_resources, queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_resources import ResourceSnapshot

    unit = 256 * 1024**2
    child = tmp_path / "child"
    runtime_control.initialize(tmp_path, 8)
    with runtime_control.transaction(tmp_path) as state:
        state.update(tasks={"a": {"root": str(child)}}, budget={"cpus": 8})
    write_json(
        tmp_path / "control/admission.json",
        {
            "schema": queue_admission.SCHEMA,
            "owner_id": "test",
            "slots": {"native:a": {"cpus": 3, "memory": int(3 * unit * 1.25)}},
        },
    )
    monkeypatch.setattr(
        "smartsom.experiments.tuning_resources.ResourceMonitor.snapshot",
        lambda self: ResourceSnapshot(16, 32 * 1024**3, 30 * 1024**3),
    )
    monkeypatch.setattr(
        "psutil.Process.memory_info", lambda self: SimpleNamespace(rss=unit)
    )
    monkeypatch.setattr(
        "psutil.Process.children",
        lambda self, recursive: [
            SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=unit))
            for _ in range(2)
        ],
    )
    token = phase_resources.BATCH.set((tmp_path, child))
    try:
        lease = phase_resources.EvaluationLease(4, baseline_cpus=3)
        assert lease.sampler_memory == 2 * unit
        assert lease.target(4) == 4
        during = queue_admission.snapshot(tmp_path)["slots"]["native:a"]
        assert during["memory"] >= (3 + 4) * unit * 1.25
        lease.close()
        after = queue_admission.snapshot(tmp_path)["slots"]["native:a"]
        assert after["cpus"] == 3
        assert after["memory"] >= 3 * unit * 1.25
    finally:
        phase_resources.BATCH.reset(token)


@pytest.mark.parametrize(
    "method,trigger", [("dqn", "transitions"), ("ppo", "fragment")]
)
def test_parallel_independent_networks_and_real_ratio_updates(
    tmp_path, method, trigger
):
    from fractions import Fraction

    from smartsom.experiments.reservation_session import ReservationSession

    args, _ = prepare(
        tmp_path,
        method,
        processes=2,
        trigger=trigger,
        independent=True,
        ratio=method == "dqn",
    )
    session = ReservationSession(args)
    try:
        assert session.step()["done"]
        learner = session.policy.learner
        assert len(learner.models) > len(learner.groups)
        assert learner.updates > 0
        if method == "dqn":
            for key, clock in learner.update_clocks.items():
                assert (
                    Fraction(learner.key_updates[key]) + clock.credit
                    == Fraction(3, 10) * clock.inserted
                )
            assert len({c.inserted for c in learner.update_clocks.values()}) > 1
            for group, loss in learner.last_losses.items():
                for key, value in loss["owner_epsilon"].items():
                    assert value == learner.update_clocks[key].epsilon(
                        learner.ownership.options[group]
                    )
    finally:
        session.close()


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    "clock_mode", ["global_gradient_steps", "network_gradient_steps"]
)
def test_target_clock_and_actual_replay_statistics_restore(
    tmp_path, independent, clock_mode
):
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
    from smartsom.experiments.reservation_runtime import episode
    from tests.helpers.algorithm import algorithm_document

    document = algorithm_document(
        dict(
            method="dqn",
            epochs=3,
            batch_size=100000,
            target_update_interval=2,
            target_update_clock=clock_mode,
        )
    )
    if independent:
        document["production"]["policies"]["dqn"]["parameter_sharing"] = {
            "default": "independent"
        }
    options = BaseAlgorithm.model_validate(document)
    _, factory, workload, _, _ = load_inputs(
        "experiments/examples/small/default/experiment.yaml"
    )
    options._owner_ids = {
        "machine": tuple(m.machine_id for m in factory.machines),
        "dispatcher": tuple(a.agv_id for a in factory.agvs),
        **{
            g: tuple(b.buffer_id for b in factory.buffers if b.role == role)
            for g, role in (
                ("pre", "machine_pre"),
                ("post", "machine_post"),
                ("input", "system_input"),
            )
        },
    }
    policy = build_policy(
        options, 101, training=True, selection="sample", numerical_threads=1
    )
    _, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        101,
        "normal",
        2000,
        max_ticks=64,
        metrics_level="summary",
    )
    policy.update(rewards, terminal=True, final_critic=critic)
    learner = policy.learner
    assert learner.updates > 0
    for group in learner.groups:
        for key in learner.ownership.keys[group]:
            steps = learner.key_updates[key]
            expected = (
                learner.updates if clock_mode == "global_gradient_steps" else steps
            ) // 2
            assert learner.target_updates[key] == expected
            # Oversized batch samples all available rows, never the configured size.
            assert learner.sampled_rows[key] == steps * len(learner.replay[key])
            stats = learner.last_losses[group]["networks"][key]
            assert stats["inserted_transitions"] == len(learner.replay[key])
            assert stats["sampled_rows"] == learner.sampled_rows[key]
            assert stats["replay_ratio"] == (steps if learner.replay[key] else 0)
    path = tmp_path / "checkpoint.pt"
    learner.save(path)
    restored = build_policy(
        options, 999, training=True, selection="sample", numerical_threads=1
    ).learner
    restored.load(path)
    assert restored.sampled_rows == learner.sampled_rows
    assert restored.target_updates == learner.target_updates
    for key, value in learner.targets.state_dict().items():
        assert torch.equal(restored.targets.state_dict()[key], value)
    # Continue from identical optimizer/replay/RNG state without new transitions.
    learner.update(rewards, terminal=True)
    restored.update(rewards, terminal=True)
    assert restored.target_updates == learner.target_updates
    for key, value in learner.targets.state_dict().items():
        assert torch.equal(restored.targets.state_dict()[key], value)
    assert restored.sampled_rows == learner.sampled_rows
    for key, value in learner.models.state_dict().items():
        assert torch.equal(restored.models.state_dict()[key], value)


def test_target_clock_rejects_unknown_mode():
    with pytest.raises(ValueError, match="target_update_clock"):
        LearnerParameters(target_update_clock="wall_minutes")


def test_mixed_target_clocks_frozen_stage_and_retained_replay():
    from collections import defaultdict
    from pathlib import Path

    import torch

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
    from smartsom.experiments.reservation_runtime import episode

    document = yaml.safe_load(Path("configs/algorithms/v2-base-ddqn.yaml").read_text())
    definition = document["production"]["policies"][document["production"]["default"]]
    definition["learner"]["params"].update(
        epochs=3,
        batch_size=99,
        replay_capacity=2,
        target_update_interval=2,
        target_update_clock="network_gradient_steps",
    )
    definition["role_params"] = {
        "dispatcher": {
            "target_update_clock": "global_gradient_steps",
            "target_update_interval": 3,
        },
        "machine": {"epochs": 1, "target_update_interval": 1},
    }
    definition["split_stages"] = ["dispatcher"]
    definition["frozen_stages"] = ["dispatcher.pickup"]
    options = BaseAlgorithm.model_validate(document)
    _, factory, workload, _, _ = load_inputs(
        "experiments/examples/small/default/experiment.yaml"
    )
    policy = build_policy(
        options, 101, training=True, selection="sample", numerical_threads=1
    )
    _, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        101,
        "normal",
        2000,
        max_ticks=64,
        metrics_level="summary",
    )
    learner = policy.learner
    calls = defaultdict(list)
    for key, target in learner.targets.items():
        old = target.load_state_dict

        def capture(state, *a, key=key, old=old, **kw):
            calls[key].append((learner.updates, learner.key_updates[key]))
            for n, v in learner.models[key].state_dict().items():
                if not isinstance(v, torch.nn.parameter.UninitializedParameter):
                    assert torch.equal(v, state[n])
            return old(state, *a, **kw)

        target.load_state_dict = capture
    policy.update(rewards, terminal=True, final_critic=critic)
    for group in learner.groups:
        settings = learner.ownership.options[group]
        for key in learner.ownership.keys[group]:
            steps = learner.key_updates[key]
            expected = (
                0
                if learner.ownership.is_frozen(key)
                else (
                    learner.updates
                    if settings.target_update_clock == "global_gradient_steps"
                    else steps
                )
                // settings.target_update_interval
            )
            assert len(calls[key]) == expected == learner.target_updates[key], (
                key,
                calls[key],
                expected,
            )
            assert all(
                (
                    global_n
                    if settings.target_update_clock == "global_gradient_steps"
                    else net_n
                )
                % settings.target_update_interval
                == 0
                for global_n, net_n in calls[key]
            )
            assert learner.sampled_rows[key] == steps * len(learner.replay[key])
            stats = learner.last_losses[group]["networks"][key]
            assert stats["replay_ratio"] == (
                learner.sampled_rows[key] / stats["inserted_transitions"]
                if stats["inserted_transitions"]
                else 0
            )

    assert any(c.inserted > 2 for c in learner.update_clocks.values())
