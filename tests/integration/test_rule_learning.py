"""Rule actions use the same native update, ownership and complete-state path."""

import copy

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import BaseAlgorithm, BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options
from tests.helpers.observations import numeric_candidate, numeric_request


@pytest.mark.parametrize("method", ["ppo", "dqn"])
@pytest.mark.parametrize("arm", ["base", "all"])
def test_rule_learning_update_evaluate_and_restore(tmp_path, method, arm):
    import torch

    options = algorithm_options(
        method=method, social_arm=arm, epochs=1, batch_size=16, width=8
    )
    data = options.model_dump(mode="json", by_alias=True)
    data["action_representation"] = {
        "default": "rules",
        "stages": {"pre.service": "direct"},
    }
    data["production"]["policies"][method]["parameter_sharing"] = {
        "default": "shared",
        "roles": {"machine": "independent"},
    }
    options = BaseAlgorithm.model_validate(data)
    factory = base_factory("small")
    options._owner_ids = {
        "machine": tuple(m.machine_id for m in factory.machines),
        "dispatcher": tuple(v.agv_id for v in factory.agvs),
    }
    # The actual compiler normally provides these stable physical identities.
    from smartsom.domain.production import ProductionScenario
    from smartsom.engine.production import ProductionSimulator

    sim = ProductionSimulator(
        ProductionScenario(factory, (), tick_limit=10), contract="base"
    )
    options._owner_ids.update(
        {
            g: tuple(b for b, r in sim.roles.items() if r == role)
            for g, role in (
                ("pre", "machine_pre"),
                ("post", "machine_post"),
                ("input", "system_input"),
            )
        }
    )
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = build_policy(options, 101)
    report, rewards, critic = episode(
        factory, workload, options, policy, 17, "normal", 100, max_ticks=160
    )
    assert report["rule_actions"]
    assert all(r["request"].rule_audit is None for r in policy.records)
    policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    assert policy.updates > 0
    model = tmp_path / "last.pt"
    policy.save(model)
    restored = build_policy(options, 999)
    restored.load(model)
    assert restored.updates == policy.updates
    for key, value in policy.learner.models.state_dict().items():
        assert torch.equal(value, restored.learner.models.state_dict()[key])
    if method == "dqn":
        assert sum(map(len, policy.learner.replay.values())) > 0
        assert {k: len(v) for k, v in restored.learner.replay.items()} == {
            k: len(v) for k, v in policy.learner.replay.items()
        }
        for key, value in policy.learner.targets.state_dict().items():
            assert torch.equal(value, restored.learner.targets.state_dict()[key])
    reports = []
    for learner in (policy, restored):
        row, reward, critic = episode(
            factory, workload, options, learner, 18, "high", 100, max_ticks=160
        )
        learner.update(
            reward, terminal=row["status"] == "completed", final_critic=critic
        )
        reports.append((row["protocol"], row["rule_actions"], reward))
    assert reports[0] == reports[1]
    for key, value in policy.learner.models.state_dict().items():
        assert torch.equal(value, restored.learner.models.state_dict()[key])
    evaluation = build_policy(options, 101, training=False, selection="argmax")
    evaluation.load(model)
    report, _, _ = episode(
        factory, workload, options, evaluation, 17, "normal", 100, max_ticks=160
    )
    assert report["rule_actions"] and not evaluation.records
    changed = copy.deepcopy(data)
    changed["action_representation"]["default"] = "direct"
    incompatible = BaseAlgorithm.model_validate(changed)
    incompatible._owner_ids = options._owner_ids
    with pytest.raises(ValueError, match="contract mismatch"):
        build_policy(incompatible).load(model)


@pytest.mark.parametrize("mode, expected", [("off", 0), ("first", 4), ("all", 8)])
def test_evaluation_replay_modes_and_resumed_case_navigation(tmp_path, mode, expected):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation_runtime as rt
    from smartsom.experiments.catalog import describe

    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/examples/small/default/experiment.yaml"
    plan = compile_reservation(
        source,
        sets=[
            "scenario.jobs.orders.total=4",
            "scenario.jobs.orders.initial=2",
            "scenario.jobs.arrivals.mean_interval_ticks=2",
            "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
            "experiment.drain_frozen=true",
            "experiment.drain_ticks=100",
            "experiment.run.task=evaluate",
            "experiment.run.methods=[rule]",
            "experiment.test_cases=8",
            "experiment.tracking.mode=disabled",
            f"experiment.replay.evaluation={mode}",
            "algorithm.action_representation.default=rules",
        ],
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    entry = plan.materialize(inputs)
    output = tmp_path / "evaluation"
    args = dict(
        experiment=entry,
        output=output,
        calibration=None,
        checkpoint=None,
        method=["rule"],
        selection="sample",
        split="test",
        cases=8,
        action_seeds=1,
        seconds=120,
        native=True,
    )
    rt.evaluate(SimpleNamespace(**args))
    recordings = sorted(output.glob("recordings/*/attempt-*/replay/run.json"))
    assert len(recordings) == expected
    assert len(describe(output)["replay_commands"]) == expected
    before = {str(p): p.read_bytes() for p in recordings}
    rt.evaluate(SimpleNamespace(**args))
    assert before == {str(p): p.read_bytes() for p in recordings}
    cases = [
        json.loads(row) for row in (output / "cases.jsonl").read_text().splitlines()
    ]
    assert len(cases) == 8 and all(row["rule_actions"] for row in cases)


def test_prespecified_test_nodes_no_wall_limit_and_train_replay(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation_session import ReservationSession

    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/examples/small/default/experiment.yaml"
    plan = compile_reservation(
        source,
        sets=[
            "scenario.jobs.orders.total=4",
            "scenario.jobs.orders.initial=2",
            "scenario.jobs.arrivals.mean_interval_ticks=2",
            "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
            "experiment.drain_frozen=true",
            "experiment.drain_ticks=40",
            "experiment.train_episodes=2",
            "experiment.training_conditions=[normal]",
            "experiment.run.task=train",
            "experiment.run.seconds=null",
            "experiment.validation.milestones=[]",
            "experiment.validation.sample_action_seeds=1",
            "experiment.final_evaluation=true",
            "experiment.evaluation.checkpoints=[{episode: 0, modes: [sample]}, {episode: 2, modes: [argmax]}]",
            "experiment.test_cases=4",
            "experiment.tracking.mode=disabled",
            "experiment.replay.train=first",
            "experiment.replay.evaluation=off",
            "algorithm.action_representation.default=rules",
            "algorithm.production.policies.ppo.learner.params.epochs=1",
            "algorithm.production.policies.ppo.learner.params.width=8",
        ],
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    entry = plan.materialize(inputs)
    args = SimpleNamespace(
        experiment=entry,
        output=tmp_path / "run",
        seed=101,
        calibration=None,
        ticks=None,
        seconds=None,
        numerical_threads=1,
    )
    session = ReservationSession(args)
    try:
        session.step()
        assert session.progress()["deadline_unix"] is None
    finally:
        session.close()
    restored = ReservationSession(args)
    try:
        assert len(restored.rows) == 1
        restored.step()
        restored.final_evaluate()
    finally:
        restored.close()
    assert (
        len(
            list(
                (tmp_path / "run/recordings/train").glob("*/attempt-*/replay/run.json")
            )
        )
        == 1
    )
    completed = json.loads((tmp_path / "run/evaluation/completed.json").read_text())
    assert len(completed["panels"]) == 2
    assert not (tmp_path / "run/validation").exists()
    for node, mode in ((0, "sample"), (2, "argmax")):
        rows = [
            json.loads(line)
            for line in (
                tmp_path / f"run/evaluation/episode-{node:08d}/{mode}/cases.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        assert len(rows) == 4
        assert all(
            r["evaluation_identity"]["training_progress"]["episodes"] == node
            for r in rows
        )


@pytest.mark.parametrize("panel_expired", [False, True])
def test_case_wall_protection_commits_but_panel_interruption_does_not(
    tmp_path, monkeypatch, panel_expired
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation_runtime as rt

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    plan = compile_reservation(
        source,
        sets=[
            "scenario.jobs.orders.total=4",
            "scenario.jobs.orders.initial=2",
            "experiment.drain_frozen=true",
            "experiment.drain_ticks=40",
            "experiment.run.task=evaluate",
            "experiment.run.methods=[rule]",
            "experiment.run.seconds=null",
            "experiment.test_cases=4",
            "experiment.tracking.mode=disabled",
            "experiment.evaluation.case_seconds={sample: 0.0000001}",
        ],
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output = tmp_path / "evaluation"
    args = dict(
        experiment=plan.materialize(inputs),
        output=output,
        native=True,
        calibration=None,
        checkpoint=None,
        method=["rule"],
        selection="sample",
        split="test",
        cases=4,
        action_seeds=1,
        seconds=None,
        deadline_unix=0 if panel_expired else None,
    )
    if panel_expired:
        with pytest.raises(TimeoutError, match="evaluation wall budget"):
            rt.evaluate(SimpleNamespace(**args))
        row = json.loads((output / "interrupted-case.json").read_text())
        assert row["wall_limit_scope"] == "panel"
        assert not (output / "cases.jsonl").exists()
        return
    rt.evaluate(SimpleNamespace(**args))
    rows = [json.loads(s) for s in (output / "cases.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["condition"] for r in rows} == {"normal", "low", "medium", "high"}
    for row in rows:
        assert row["wall_limit_scope"] == "case"
        assert row["status"] == "truncated"
        assert row["wall_clock_fragment"]
        assert row["termination_reason"] == "wall_clock_interrupted"
        assert row["joint_cost"] is None
        assert row["common_window"]["status"] == "censored"

    def unexpected_retry(*args, **kwargs):
        pytest.fail("a receipted case protection outcome must not execute again")

    monkeypatch.setattr(rt, "episode", unexpected_retry)
    rt.evaluate(SimpleNamespace(**args))
    assert [
        json.loads(s) for s in (output / "cases.jsonl").read_text().splitlines()
    ] == rows


def _packed_request(tick, owner="car"):
    from smartsom.domain.observation_data import vector

    return numeric_request(
        tick,
        "dispatcher",
        owner,
        "pickup",
        (
            numeric_candidate("a", ("a",), vector(1)),
            numeric_candidate("b", ("b",), vector(2)),
        ),
        vector(),
        ((),) * 4,
    )


def _equal_dqn_values(a, b):
    from collections.abc import Sequence

    import torch

    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal_dqn_values(a[key], b[key])
    elif isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _equal_dqn_values(x, y)
    else:
        assert a == b


def test_packed_dqn_and_custom_observation_storage():
    from smartsom.learning.dqn.learner import BaseDQN
    from smartsom.learning.dqn.replay import PackedRows

    request = _packed_request(0)
    rows = PackedRows([{"request": request, "index": 0}])
    assert type(rows.entries[0]) is bytes
    assert rows[0] == rows[0] and rows[0] is not rows[0]
    options = algorithm_options(method="dqn", width=8)
    data = options.model_dump(mode="json", by_alias=True)
    data["production"]["policies"]["dqn"]["observation"]["target"] = (
        "tests.helpers.example_observation.create_observation"
    )
    # Direct learner construction does not invoke a custom observation factory.
    learner = BaseDQN(BaseAlgorithm.model_validate(data), 11)
    assert not learner.packed_replay
    assert type(learner.records) is list
    assert all(type(rows) is list for rows in learner.replay.values())


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("capacity", [2, 50])
def test_packed_dqn_capacity_order_counts_rng_and_update(terminal, capacity):
    from smartsom.learning.dqn.learner import BaseDQN

    options = algorithm_options(
        method="dqn", width=8, epochs=1, batch_size=2, replay_capacity=capacity
    )
    packed, raw = BaseDQN(options, 17), BaseDQN(options, 17)
    raw.packed_replay = False
    raw.records, raw.replay = [], {key: [] for key in raw.models}
    # Existing replay plus more transitions than capacity, two owners, same ticks.
    for policy in (packed, raw):
        for tick in range(3):
            policy.choose(_packed_request(tick))
        policy.update([-1.0] * 3, terminal=True)
        for tick in range(5):
            for owner in ("car", "other"):
                policy.choose(_packed_request(tick, owner))
    assert packed.update([-1.0] * 5, terminal=terminal) == raw.update(
        [-1.0] * 5, terminal=terminal
    )
    _equal_dqn_values(packed.get_state(), raw.get_state())
    assert packed.truncated_tails == (0 if terminal else 2)
    assert packed.last_losses["dispatcher"]["transitions"] == (10 if terminal else 8)
    assert [r["request"].tick for r in packed.replay["dispatcher"]] == [
        r["request"].tick for r in raw.replay["dispatcher"]
    ]


def test_packed_dqn_plain_checkpoint_restore_and_next_update(tmp_path):
    import torch

    from smartsom.learning.dqn.learner import BaseDQN
    from smartsom.learning.dqn.replay import PackedRows

    options = algorithm_options(method="dqn", width=8, epochs=1, batch_size=2)
    raw = BaseDQN(options, 29)
    raw.packed_replay = False
    raw.records, raw.replay = [], {key: [] for key in raw.models}
    for tick in range(4):
        raw.choose(_packed_request(tick))
    raw.update([-1.0] * 4, terminal=True)
    checkpoint = tmp_path / "plain.pt"
    raw.save(checkpoint)
    expected_rng = torch.get_rng_state().clone()
    packed = BaseDQN(options, 999)
    packed.load(checkpoint)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(isinstance(rows, PackedRows) for rows in packed.replay.values())
    _equal_dqn_values(raw.get_state(), packed.get_state())
    for tick in range(3):
        assert raw.choose(_packed_request(tick)) == packed.choose(_packed_request(tick))
    assert raw.update([-2.0] * 3, terminal=False) == packed.update(
        [-2.0] * 3, terminal=False
    )
    _equal_dqn_values(raw.get_state(), packed.get_state())
    packed.save(tmp_path / "packed.pt")
    restored = BaseDQN(options, 0)
    restored.load(tmp_path / "packed.pt")
    _equal_dqn_values(packed.get_state(), restored.get_state())


def test_packed_dqn_independent_owner_capacity_is_per_key():
    from types import SimpleNamespace

    from smartsom.domain.base_decisions import GROUPS
    from smartsom.learning.dqn.replay import (
        PackedRows,
        bounded_transitions,
        transitions,
    )

    ownership = SimpleNamespace(
        keys={g: (("a", "b") if g == "dispatcher" else (g,)) for g in GROUPS},
        options={g: SimpleNamespace(replay_capacity=2) for g in GROUPS},
        key=lambda request: (
            request.owner if request.group == "dispatcher" else request.group
        ),
    )
    records = [
        {"request": _packed_request(tick, owner), "index": 0}
        for tick in range(5)
        for owner in ("a", "b")
    ]
    raw, raw_omitted = transitions(records, [-1.0] * 5, terminal=True, scale=1.0)
    packed, omitted, counts = bounded_transitions(
        PackedRows(records), [-1.0] * 5, terminal=True, scale=1.0, ownership=ownership
    )
    assert omitted == raw_omitted == 0
    for owner in ("a", "b"):
        expected = [r for r in raw["dispatcher"] if r["request"].owner == owner]
        assert counts[owner] == len(expected) == 5
        assert list(packed[owner]) == expected[:2]
        assert [r["request"].tick for r in packed[owner]] == [4, 3]


@pytest.mark.parametrize("packed_donor", [False, True])
def test_packed_dqn_set_state_preserves_borrowed_payload_and_live_donor(packed_donor):
    from smartsom.learning.dqn.learner import BaseDQN

    options = algorithm_options(method="dqn", width=8, epochs=1, batch_size=2)
    donor = BaseDQN(options, 73)
    if not packed_donor:
        donor.packed_replay = False
        donor.records, donor.replay = [], {key: [] for key in donor.models}
    for tick in range(4):
        donor.choose(_packed_request(tick))
    donor.update([-1.0] * 4, terminal=True)
    payload = donor.get_state()
    expected = copy.deepcopy(payload)
    recipients = [BaseDQN(options, seed) for seed in (74, 75)]
    for recipient in recipients:
        recipient.set_state(payload)
        _equal_dqn_values(payload, expected)
        _equal_dqn_values(donor.get_state(), expected)
        _equal_dqn_values(recipient.get_state(), expected)
    left, right = recipients
    assert left.replay["dispatcher"].entries is not right.replay["dispatcher"].entries
    if packed_donor:
        assert (
            left.replay["dispatcher"].entries is not donor.replay["dispatcher"].entries
        )
        assert (
            left.replay["dispatcher"].entries[0]
            is donor.replay["dispatcher"].entries[0]
        )
    left.replay["dispatcher"].entries.clear()
    assert len(donor.replay["dispatcher"]) == len(right.replay["dispatcher"]) == 4
    _equal_dqn_values(payload, expected)
