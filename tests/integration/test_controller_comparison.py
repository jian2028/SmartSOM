"""Central/Justin adapters share the current entry, rules, worlds and physics."""

import copy
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.components import resolve
from smartsom.config.reservation import compile_reservation
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
from smartsom.domain.public_facts import freeze
from smartsom.experiments.reservation_runtime import episode

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/controller-comparison/narrow-layout"


def inputs(size="small", method="central-ppo"):
    return load_inputs(STUDY / f"{size}-{method}.yaml")


def module(name):
    options = inputs()[3]
    factory = resolve(options.learning_policy.learner)
    return importlib.import_module(factory.__module__ + "." + name)


def request(tick=0, group="machine", count=2, owner="machine_1"):
    candidates = tuple(
        BaseCandidate(f"rule:{i}", (f"job-{i}", "normal"), vector(i))
        for i in range(count)
    )
    return BaseDecision(
        tick,
        group,
        owner,
        "process" if group == "machine" else "accept",
        candidates,
        vector(),
        ((), (), (), ()),
        inputs=freeze(
            {
                "context": (1.0, 2.0, 3.0),
                "token": (1.0, 0.0),
                "layout": "test-fixed-layout",
                "prefix": (),
                "prefix_actions": (),
            }
        ),
    )


def bounded_run(policy, values, ticks=12):
    ex, factory, workload, options, _ = values
    return episode(
        factory,
        workload,
        options,
        policy,
        7,
        "normal",
        ex.drain_ticks,
        max_ticks=ticks,
        metrics_level="off",
    )


@pytest.mark.parametrize("size", ["small", "medium"])
@pytest.mark.parametrize("method", ["central-ppo", "justin"])
def test_two_maps_real_decisions_and_updates(size, method):
    values = inputs(size, method)
    policy = build_policy(values[3], 4101, training=method == "central-ppo")
    result, rewards, critic = bounded_run(policy, values)
    assert result["tick"] == 12
    assert result["decisions"] > 0
    if policy.training:
        assert len(policy.records) == result["decisions"]
        before = copy.deepcopy(policy.learner.model.state_dict())
        losses = policy.update(rewards, terminal=False, final_critic=critic)
        assert policy.updates > 0
        assert losses["central"]["parameters"] > 0
        assert any(
            not torch.equal(v, policy.learner.model.state_dict()[k])
            for k, v in before.items()
        )
    else:
        assert not policy.training and not policy.records


def test_role_returns_and_same_tick_credit_have_independent_expected_values():
    records = [
        dict(request=request(t, g), value=v)
        for t, g, v in [
            (0, "machine", 0.0),
            (0, "machine", 2.0),
            (1, "pre", 0.0),
            (2, "pre", 1.0),
        ]
    ]
    module("rollout").prepare_returns(records, [1.0, 2.0, 3.0], 0.5, 4.0)
    assert [r["return"] for r in records] == [7.0, 7.0, 6.5, 5.5]
    assert [r["advantage"] for r in records] == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_per_decision_clipping_and_forced_actor_mask():
    rows = [
        dict(
            index=0,
            log_prob=float(torch.tensor(0.5).log()),
            advantage=1.0,
            actor_mask=True,
            **{"return": 0.0},
        ),
        dict(
            index=0, log_prob=0.0, advantage=100.0, actor_mask=False, **{"return": 0.0}
        ),
    ]
    logits = torch.tensor([[0.8, 0.2], [1.0, 0.0]]).log().requires_grad_()
    loss, actor, critic = module("rollout").objective(
        logits, torch.zeros(2), rows, 0.2, 0.0
    )
    assert float(actor.detach()) == pytest.approx(-1.2)
    assert float(critic) == 0
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_prefix_is_frozen_replayed_and_reset_even_at_same_episode_start_tick():
    learner = build_policy(inputs()[3], 4).learner
    first = request()
    learner.choose(first, vector())
    learner.choose(replace(first, owner="machine_2"), vector())
    rows = learner.records
    assert not rows[0]["request"].inputs["prefix"]
    assert len(rows[1]["request"].inputs["prefix"]) == 1
    previous = rows[1]["request"].inputs["prefix_actions"][0]
    assert previous[:3] == ("machine", "machine_1", "process")
    assert previous[3] == rows[0]["action"]
    batch = learner.pack(
        [r["request"] for r in rows],
        critics=[vector(), vector()],
        options=learner.options,
    )
    logits = learner.model.actor(batch)
    recomputed = logits.log_softmax(-1)[
        torch.arange(2), torch.tensor([r["index"] for r in rows])
    ]
    assert recomputed.detach().tolist() == pytest.approx(
        [r["log_prob"] for r in rows], abs=1e-6
    )
    learner.choose(request(tick=1), vector())
    assert not learner.records[-1]["request"].inputs["prefix"]
    learner.begin_episode()
    learner.choose(request(tick=1), vector())
    assert not learner.records[-1]["request"].inputs["prefix"]


def test_forced_actions_record_prefix_but_have_no_actor_loss_and_eval_skips_critic(
    monkeypatch,
):
    options = inputs()[3]
    learner = build_policy(options, 4).learner
    forced = request(count=1)
    learner.choose(forced, vector())
    assert not learner.records[0]["actor_mask"]
    inference = build_policy(options, 9, training=False).learner
    inference.load_sampling_state(learner.sampling_state())

    def forbidden(*args):
        raise AssertionError("critic used during inference")

    monkeypatch.setattr(inference.model.critic, "forward", forbidden)
    inference.choose(forced)
    inference.choose(request())
    assert len(inference.prefix) == 2 and not inference.records


def test_complete_checkpoint_restore_matches_next_update_and_initial_state(tmp_path):
    options = inputs()[3]
    original = build_policy(options, 11)
    empty = tmp_path / "empty.pt"
    original.save(empty)
    restored = build_policy(options, 99)
    restored.load(empty)
    for policy in (original, restored):
        policy.choose(request(), vector())
        policy.update([1.0, -2.0], terminal=True)
    for key, value in original.learner.model.state_dict().items():
        assert torch.equal(value, restored.learner.model.state_dict()[key])
    saved = tmp_path / "trained.pt"
    original.save(saved)
    restored = build_policy(options, 33)
    restored.load(saved)
    for policy in (original, restored):
        policy.begin_episode()
        for tick in range(3):
            policy.choose(request(tick=tick), vector())
        policy.update([2.0, -3.0, 1.0], terminal=False, final_critic=vector())
    assert original.updates == restored.updates
    for key, value in original.learner.model.state_dict().items():
        assert torch.equal(value, restored.learner.model.state_dict()[key])


def test_rejects_same_width_different_layout():
    learner = build_policy(inputs()[3], 1).learner
    learner.choose(request(), vector())
    changed = dict(request().inputs)
    changed["layout"] = "different-owner-identities"
    with pytest.raises(ValueError, match="layout mismatch"):
        learner.choose(replace(request(), inputs=freeze(changed)), vector())


def test_config_inventory_and_exact_rule_protocol_alignment():
    for size in ("small", "medium"):
        base = load_inputs(
            ROOT / f"experiments/rule-dqn/narrow-layout/{size}-base.yaml"
        )
        central, justin = inputs(size), inputs(size, "justin")
        assert central[3].action_representation == base[3].action_representation
        for variant in (central, justin):
            assert variant[3].decision_protocol == base[3].decision_protocol
            assert variant[3].traffic == base[3].traffic
            assert variant[0].case_set_sha256 == base[0].case_set_sha256
            assert variant[0].condition_scenarios == base[0].condition_scenarios
        assert central[0].train_seeds == base[0].train_seeds == (4101, 4102, 4103)
        assert central[0].train_episodes == 1000
        raw = yaml.safe_load((STUDY / f"{size}-justin.yaml").read_text())
        assert "train_seeds" not in raw and raw["run"]["task"] == "evaluate"


def test_frozen_components_report_global_owner_and_run_after_relocation(tmp_path):
    plan = compile_reservation(STUDY / "small-central-ppo.yaml")
    assert set(plan.component_report["network_ownership"]) == {"central"}
    folder = tmp_path / "relocated"
    folder.mkdir()
    values = load_inputs(plan.materialize(folder))
    policy = build_policy(values[3], 9)
    result, rewards, critic = bounded_run(policy, values, ticks=2)
    policy.update(rewards, terminal=False, final_critic=critic)
    assert result["tick"] == 2 and policy.updates


def test_extension_import_and_initialization_do_not_advance_global_randomness():
    import random

    state, tensor_state = random.getstate(), torch.get_rng_state().clone()
    policy = build_policy(inputs()[3], 7)
    policy.choose(request(), vector())
    assert random.getstate() == state
    assert torch.equal(torch.get_rng_state(), tensor_state)


def test_public_topology_is_opt_in_and_future_jobs_do_not_change_observation():
    from smartsom.config.base_diagnostic import BaseWorkload
    from smartsom.config.base_layout import base_factory
    from smartsom.engine.production import ProductionSimulator
    from smartsom.engine.public_facts import public_facts
    from smartsom.experiments.reservation_runtime import generate_scenario

    factory = base_factory("small")
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=10,
    )
    world = generate_scenario(factory, workload, 33, "normal", 2000)
    first = ProductionSimulator(world)
    assert "public_topology" not in first.protocol.__dict__
    ordinary = public_facts(first)
    assert "topology" not in ordinary
    assert "public_topology" not in first.protocol.__dict__
    public = public_facts(first, topology=True)
    assert "topology" in public
    assert public["topology"] is public_facts(first, topology=True)["topology"]
    with pytest.raises(TypeError):
        public["topology"]["width"] = 9
    # Changing unreleased due dates does not change the released public state.
    second = ProductionSimulator(world)
    unreleased = set(second.demands) - second.released
    assert unreleased
    for identity in unreleased:
        second.demands[identity] = replace(second.demands[identity], due_at=999999)
    observation = module("observation").create_observation()
    assert observation(public, request()) == observation(
        public_facts(second, topology=True), request()
    )


def test_native_base_and_sl_never_request_topology(monkeypatch):
    from smartsom.engine import public_facts

    def forbidden(*args):
        raise AssertionError("native Base/SL requested central topology")

    monkeypatch.setattr(public_facts, "topology_facts", forbidden)
    for method in ("base", "sl"):
        values = load_inputs(
            ROOT / f"experiments/rule-dqn/narrow-layout/small-{method}.yaml"
        )
        policy = build_policy(values[3], 4101)
        result, rewards, critic = bounded_run(policy, values)
        policy.update(rewards, terminal=False, final_critic=critic)
        assert result["tick"] == 12


def test_justin_wip_preferences_normal_mode_and_default_acceptance():
    policy = build_policy(inputs(method="justin")[3], training=False)
    facts = freeze(
        {
            "vehicles": ({"id": "agv", "job": None, "target": None, "cell": (0, 0)},),
            "buffers": tuple(
                {
                    "id": owner,
                    "ready": (job,),
                    "quota": 1,
                    "capacity": {"pool": 2},
                    "inventory": {"pool": [job]},
                }
                for owner, job in (("input", "new"), ("post", "wip"))
            ),
            "jobs": ({"id": "new", "step": 0}, {"id": "wip", "step": 2}),
            "topology": {"ports": {"input": ((1, 0),), "post": ((5, 0),)}},
        }
    )
    pickup = BaseDecision(
        0,
        "dispatcher",
        "agv",
        "pickup",
        (
            BaseCandidate("near", ("input",), vector(1)),
            BaseCandidate("wip", ("post",), vector(5)),
        ),
        vector(),
        ((), (), (), ()),
        inputs=facts,
    )
    assert policy.choose(pickup) == ("post",)
    machine = replace(
        request(),
        candidates=(
            BaseCandidate("fast", ("a", "fast"), vector()),
            BaseCandidate("normal", ("b", "normal"), vector()),
        ),
    )
    assert policy.choose(machine) == ("b", "normal")
    from smartsom.policies.rule import BaseRule

    acceptance = request(group="pre")
    assert policy.choose(acceptance) == BaseRule().choose(acceptance)


def test_batch_first_wave_contains_one_learning_seed_and_one_rule_evaluation():
    from smartsom.experiments.cli import _parser
    from smartsom.experiments.launch import compile_launch

    args = _parser().parse_args(
        [
            "check",
            str(STUDY),
            "--id",
            "medium-central-ppo",
            "--select",
            "train_seeds: [4101]",
            "--id",
            "medium-justin",
        ]
    )
    plan = compile_launch(STUDY, ids=args.batch_ids, selectors=args.batch_selectors)
    assert len(plan.files) == 8
    selected = [row for row in plan.files if row["id"] in plan.selection["tasks"]]
    assert len(selected) == 2
    assert [
        (row["path"].stem, row["compiled"].experiment.run.task) for row in selected
    ] == [("medium-central-ppo", "train"), ("medium-justin", "evaluate")]
    assert selected[0]["compiled"].experiment.run.seed == 4101


def test_central_uses_native_session_transactions_and_restores_completed_budget(
    tmp_path,
):
    from smartsom.experiments import reservation
    from smartsom.experiments.reservation_session import ReservationSession

    plan = compile_reservation(
        STUDY / "small-central-ppo.yaml",
        sets=[
            "experiment.train_episodes=null",
            "experiment.train_ticks=20",
            "experiment.budget_boundary=cut",
            "experiment.evaluation.checkpoints=[]",
            "experiment.final_evaluation=false",
            "experiment.tracking.mode=disabled",
        ],
    )
    root = reservation.allocate(plan, root=tmp_path / "central-session")
    args, _ = reservation.arguments(root)
    session = ReservationSession(args)
    assert session.policy.learner.model is None  # The initial durable commit is valid.
    result = session.step()
    assert result["done"] and session.ticks == 20
    expected = copy.deepcopy(session.policy.learner.model.state_dict())
    updates = session.policy.updates
    session.close()
    restored = ReservationSession(args)
    try:
        assert restored.ticks == 20 and restored.policy.updates == updates > 0
        assert restored.step()["done"]
        assert len(restored.rows) == 1
        for key, value in expected.items():
            assert torch.equal(value, restored.policy.learner.model.state_dict()[key])
    finally:
        restored.close()


def test_public_smoke_entry_executes_central_and_justin(tmp_path):
    import json

    from smartsom.experiments import reservation

    for method in ("central-ppo", "justin"):
        output = tmp_path / method
        plan = compile_reservation(
            STUDY / f"small-{method}.yaml",
            task="smoke",
            sets=[
                "experiment.evaluation.checkpoints=[]",
                "experiment.final_evaluation=false",
                "experiment.run.smoke_ticks=5",
                "experiment.run.seconds=60",
                "experiment.resources.guard=null",
                "experiment.tracking.mode=disabled",
                f'experiment.run.output="{output}"',
            ],
        )
        result = reservation.run(
            plan, display_options={"progress": "off", "verbose": False}
        )
        assert result["status"] == "completed"
        rows = json.loads((output / "summary.json").read_text())["results"]
        assert len(rows) == 1 and rows[0]["tick"] == 5
        assert (output / "last.pt").exists() == (method == "central-ppo")


def test_tracking_uses_central_ownership_not_reservation_protocol(tmp_path):
    from smartsom.telemetry.tracking_metadata import metadata

    for method, expected_control, expected_method in (
        ("central-ppo", "centralized", "morris_central_ppo"),
        ("justin", "distributed", "justin"),
    ):
        folder = tmp_path / method
        (folder / "inputs").mkdir(parents=True)
        plan = compile_reservation(STUDY / f"small-{method}.yaml")
        plan.materialize(folder / "inputs")
        state = torch.get_rng_state().clone()
        report = metadata(folder, {}, seed=4101)
        assert report["comparison"]["method"] == expected_method
        assert report["comparison"]["control"] == expected_control
        assert f"control:{expected_control}" in report["tags"]
        assert torch.equal(state, torch.get_rng_state())


def test_cli_resume_continues_an_incomplete_central_session(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    from smartsom.experiments import reservation
    from smartsom.experiments.reservation_session import ReservationSession

    plan = compile_reservation(
        STUDY / "small-central-ppo.yaml",
        sets=[
            "experiment.train_episodes=null",
            "experiment.train_ticks=20",
            "experiment.budget_boundary=cut",
            "experiment.evaluation.checkpoints=[]",
            "experiment.final_evaluation=false",
            "experiment.tracking.mode=disabled",
        ],
    )
    root = reservation.allocate(plan, root=tmp_path / "partial-central")
    args, _ = reservation.arguments(root)
    session = ReservationSession(args)
    # A bounded first rollout creates a real incomplete native transaction stream.
    monkeypatch.setattr(session, "rollout_limit", lambda: 10)
    progress = session.step()
    assert not progress["done"] and session.ticks == 10
    assert progress["network_count"] == 1 and progress["model_parameters"] > 0
    updates = session.policy.updates
    first = json.loads((root / "continuation/current.json").read_text())["generation"]
    session.close()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "resume",
            str(root),
            "--foreground",
            "--progress",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    progress = json.loads((root / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["physical_ticks"] == 20
    assert progress["episodes"] == 2 and progress["updates"] > updates
    assert (
        json.loads((root / "continuation/current.json").read_text())["generation"]
        != first
    )
    assert len((root / "episodes.jsonl").read_text().splitlines()) == 2


def test_real_multistage_next_rollout_and_update_survive_state_transfer():
    values = inputs()
    policy = build_policy(values[3], 713)
    _, rewards, critic = bounded_run(policy, values, ticks=250)
    points = {
        row["request"].group + "." + row["request"].stage for row in policy.records
    }
    assert {
        "machine.process",
        "dispatcher.pickup",
        "dispatcher.delivery",
        "input.service",
        "pre.accept",
        "pre.service",
        "post.accept",
        "post.service",
    } <= points
    policy.update(rewards, terminal=False, final_critic=critic)
    restored = build_policy(values[3], 999)
    restored.set_state(copy.deepcopy(policy.get_state()))
    trajectories = []
    for candidate in (policy, restored):
        _, rewards, critic = bounded_run(candidate, values, ticks=250)
        trajectories.append(
            [
                (r["request"].tick, r["action"], r["log_prob"], r["value"])
                for r in candidate.records
            ]
        )
        candidate.update(rewards, terminal=False, final_critic=critic)
    assert trajectories[0] == trajectories[1]
    for key, value in policy.learner.model.state_dict().items():
        assert torch.isfinite(value).all()
        assert torch.equal(value, restored.learner.model.state_dict()[key])
