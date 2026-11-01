"""Rule actions use the same native update, ownership and complete-state path."""

import copy

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import BaseAlgorithm, BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options


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
            "workload.orders=4",
            "workload.initial_orders=2",
            "workload.mean_interarrival=2",
            "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
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
            "workload.orders=4",
            "workload.initial_orders=2",
            "workload.mean_interarrival=2",
            "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
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
