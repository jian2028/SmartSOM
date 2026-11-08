"""Independent arithmetic and real centralized/custom component contracts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
from smartsom.domain.training import LearningContext, RewardTimeline
from smartsom.experiments.reservation_runtime import episode
from smartsom.learning.returns import prepare_returns

ROOT = Path(__file__).resolve().parents[2]


def configuration(method):
    return BaseAlgorithm.model_validate(
        yaml.safe_load(
            (ROOT / f"configs/algorithms/e1a-current-{method}.yaml").read_text()
        )
    )


def run(options, policy, *, ticks=30, recording=None):
    ex, f, w, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    return episode(
        f,
        w,
        options,
        policy,
        19,
        "normal",
        ex.drain_ticks,
        max_ticks=ticks,
        metrics_level="off",
        recording=recording,
    )


def test_reward_timeline_records_training_cut_without_changing_terminal():
    options = configuration("dqn")
    policy = build_policy(options, 101)
    report, rewards, _ = run(options, policy, ticks=30)
    assert report["status"] == "truncated"
    assert rewards.end_reason == report["termination_reason"] == "training_budget_cut"


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_centralized_update_and_complete_restore(method, tmp_path):
    options = configuration(method)
    policy = build_policy(options, 101)
    report, rewards, final = run(options, policy)
    assert report["decisions"] > 0
    rows = list(policy.records)
    assert all(r["request"].group == "central" for r in rows)
    assert any(a["request"].tick == b["request"].tick for a, b in zip(rows, rows[1:]))
    policy.update(rewards, terminal=False, final_critic=final)
    assert policy.updates > 0
    path = tmp_path / "state.pt"
    policy.save(path)
    restored = build_policy(options, 101)
    restored.load(path)
    original_rng = torch.get_rng_state()
    a = run(options, policy)
    policy.update(a[1], terminal=False, final_critic=a[2])
    torch.set_rng_state(original_rng)
    b = run(options, restored)
    restored.update(b[1], terminal=False, final_critic=b[2])
    assert a[1] == b[1]
    for name, tensor in policy.learner.models.state_dict().items():
        other = restored.learner.models.state_dict()[name]
        assert (
            torch.equal(tensor, other)
            if isinstance(tensor, torch.Tensor)
            else tensor == other
        ), name
    if method == "dqn":
        assert policy.learner.truncated_tails == 0
        assert any(r["physical_dt"] == 0 for r in policy.learner.replay["central"])


def test_hand_gae_same_tick_and_time_discount():
    class Model:
        def value(self, inputs):
            return torch.tensor([3.0 for _ in inputs])

    rows = [
        dict(request=LearningContext(t, "central", "factory"), value=v)
        for t, v in [(0, 1.0), (0, 2.0), (2, 4.0)]
    ]
    settings = SimpleNamespace(
        gamma=0.5, learning_reward_scale=1.0, return_estimator="gae", gae_lambda=0.8
    )
    # rewards [2,4,6]: last delta=6+.5*3-4=3.5;
    # second delta=2+.5*4+.25*4-2=3; A=3+.25*.8*3.5=3.7;
    # first delta=2-1=1; A=1+.8*3.7=3.96.
    prepare_returns(
        rows,
        [2.0, 4.0, 6.0],
        settings,
        terminal=False,
        final_critic={},
        model_for=lambda r: Model(),
    )
    assert [r["raw_advantage"] for r in rows] == pytest.approx([3.96, 3.7, 3.5])
    assert [r["return"] for r in rows] == pytest.approx([4.96, 5.7, 7.5])


def test_reward_subjects_do_not_follow_parameter_sharing():
    rewards = RewardTimeline()
    rewards.append_result(
        {"team": 1.0, "roles": {"machine": 2.0}, "agents": {("machine", "m1"): 3.0}}, 1
    )
    rewards.append_result({"team": 4.0}, 2)
    assert rewards.for_subject("machine", "m1") == [6.0, 0.0, 4.0]
    assert rewards.for_subject("machine", "m2") == [3.0, 0.0, 4.0]
    assert rewards.for_subject("central", "factory") == [1.0, 0.0, 4.0]
    with pytest.raises(ValueError):
        rewards.append_result({"team": float("nan")}, 1)


def test_gae_separates_agent_sequences():
    class Model:
        def value(self, inputs):
            raise AssertionError("terminal must not bootstrap")

    rows = [
        dict(request=LearningContext(0, "machine", owner), value=value)
        for owner, value in [("a", 1.0), ("b", 100.0)]
    ]
    settings = SimpleNamespace(
        gamma=1.0, learning_reward_scale=1.0, return_estimator="gae", gae_lambda=0.95
    )
    prepare_returns(
        rows,
        [5.0],
        settings,
        terminal=True,
        final_critic=None,
        model_for=lambda r: Model(),
    )
    assert [r["return"] for r in rows] == [5.0, 5.0]


def test_recorded_reward_equals_unrecorded_and_replay_metadata(tmp_path):
    import json

    from smartsom.trace.current import CurrentRecording

    options = configuration("ppo")
    a = run(options, build_policy(options, 7, training=False))
    with CurrentRecording(tmp_path / "recorded") as rec:
        b = run(options, build_policy(options, 7, training=False), recording=rec)
    assert a[1] == b[1]
    rows = [
        json.loads(s)
        for s in (tmp_path / "recorded/rewards.jsonl").read_text().splitlines()
    ]
    assert sum(r["team"] for r in rows) == a[0]["learning_return"]
    assert rows[0]["start"] == 0 and rows[-1]["end"] == a[0]["tick"]
    from smartsom.trace.production import Playback

    assert (
        Playback(tmp_path / "recorded").learning_return(a[0]["tick"])
        == a[0]["learning_return"]
    )
    manifest = json.loads((tmp_path / "recorded/run.json").read_text())
    assert manifest["learning_components"]["reward"]["target"].endswith(
        "e1a.reward.create_reward"
    )


def test_external_named_critic_stateful_reward_and_custom_learner(tmp_path):
    from smartsom.config.components import freeze_components, validate_components

    extension = tmp_path / "custom.py"
    extension.write_text("""
import torch
from torch import nn
from smartsom.controllers.e1a.network import Network, pack
from smartsom.learning.ppo.learner import BasePPO

class CustomNetwork(Network):
    def __init__(self):
        super().__init__(False)
        self.critic = nn.Linear(3,1)
    def value(self, inputs):
        return self.critic(torch.tensor([v['counts'] for v in inputs],dtype=torch.float32)).squeeze(-1)
def create_network(*,group,options,value_head):
    return CustomNetwork()
create_network.INPUT_SCHEMA='smartsom.e1a-current/actor-v2'
create_network.CRITIC_SCHEMA='custom.critic/v1'
create_network.pack=pack

def create_critic():
    def observe(facts, context):
        assert not hasattr(context,'candidates')
        return {'counts':(facts['tick']/100, len(facts['jobs'])/100, float(context.bootstrap))}
    return observe
create_critic.OUTPUT_SCHEMA='custom.critic/v1'

class Reward:
    public_snapshots=False
    def __init__(self): self.total=0
    def begin_episode(self): self.episode=0
    def __call__(self, transition):
        self.total+=1; self.episode+=1
        previous=getattr(self,'previous',None)
        if previous is not None: assert previous['failed'] >= 0
        self.previous=transition.outcomes
        try: transition.outcomes['failed']=123
        except TypeError: pass
        else: raise AssertionError('mutable outcomes')
        assert transition.end > transition.start
        assert transition.actions
        assert all(a.group == "central" and a.stage == "assign" and a.identity == a.action for a in transition.actions)
        return {'team':1.,'roles':{'central':2.},'agents':{('central','factory'):3.}}
    def get_state(self): return {'total':self.total,'previous':getattr(self,'previous',None)}
    def set_state(self,state):
        self.total=state['total']; self.previous=state['previous']
def create_reward(*,orders,quality_penalty): return Reward()
create_reward.CONTRACT='smartsom.reward/v1'

class Learner(BasePPO):
    def update(self,rewards,*,terminal,final_critic):
        from smartsom.learning.transitions import decision_transitions
        transitions=decision_transitions(self.records,rewards,terminal=terminal)
        assert transitions and all(t.physical_dt>=0 for t in transitions)
        return super().update(rewards,terminal=terminal,final_critic=final_critic)
def create_learner(*,options,seed,training,selection,numerical_threads,**params):
    return Learner(options,seed,training=training,selection=selection,numerical_threads=numerical_threads)
create_learner.CONTRACT='smartsom.learner/v1'
""")
    data = configuration("ppo").model_dump(mode="json", by_alias=True)

    def spec(name):
        return {"path": str(extension), "factory": name}

    data["reward"] = spec("create_reward")
    policy = data["production"]["policies"]["ppo"]
    policy["critic_observation"] = spec("create_critic")
    policy["network"] = spec("create_network")
    policy["learner"] = {
        **spec("create_learner"),
        "params": policy["learner"]["params"],
    }
    options = BaseAlgorithm.model_validate(data)
    validate_components(options)
    frozen, files, report = freeze_components(options, tmp_path)
    assert "reward" in report and "production.policies.ppo.critic_observation" in report
    a = build_policy(options, 9)
    result = run(options, a)
    assert result[1].for_subject("central", "factory") == [6 * v for v in result[1]]
    assert all(len(v["counts"]) == 3 for v in result[2].values())
    a.update(result[1], terminal=False, final_critic=result[2])
    a.save(tmp_path / "saved.pt")
    b = build_policy(options, 9)
    b.load(tmp_path / "saved.pt")
    assert a.training_components.reward_state == b.training_components.reward_state
    run(options, b)
    assert (
        b.training_components.reward_state["total"]
        == 2 * a.training_components.reward_state["total"]
    )
    changed = deepcopy(data)
    changed["reward"]["params"] = {"unexpected": 1}
    with pytest.raises(ValueError, match="contract mismatch"):
        build_policy(BaseAlgorithm.model_validate(changed), 9).load(
            tmp_path / "saved.pt"
        )


def test_custom_transition_keeps_truncation_with_bootstrap_candidate():
    from smartsom.engine.central_control import ADVANCE, CentralDecision
    from smartsom.learning.transitions import decision_transitions

    rewards = RewardTimeline()
    rewards.append_result({"team": 2.0}, 1)
    rewards.final_requests["central", "factory"] = CentralDecision(1, (ADVANCE,), None)
    request = CentralDecision(0, (ADVANCE,), None)
    (transition,) = decision_transitions(
        [{"request": request, "index": 0}], rewards, terminal=False
    )
    assert transition.truncated and not transition.terminated
    assert transition.next_request.tick == 1
    assert transition.reward == 2.0


def test_reward_events_exclude_unrealized_processing_draws():
    from smartsom.domain.training import public_event

    result = public_event(
        {
            "tick": 1,
            "kind": "processing_started",
            "job": "j1",
            "nominal_ticks": 10,
            "actual_ticks": 17,
            "defect": True,
            "future_draw": 0.9,
        }
    )
    assert dict(result) == {
        "tick": 1,
        "kind": "processing_started",
        "job": "j1",
        "nominal_ticks": 10,
    }


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_real_natural_terminal_central_episode(method, monkeypatch):
    from smartsom.config.base_diagnostic import generate_scenario

    ex, f, w, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    w = w.model_copy(
        update={
            "orders": 2,
            "initial_orders": 2,
            "families": (("A",),),
            "optional_repeat_probability": 0.0,
        }
    )
    options = configuration(method)
    policy = build_policy(options, 13)
    scenario = generate_scenario(f, w, 19, "normal", 2000)
    report, rewards, critic = episode(
        f,
        w,
        options,
        policy,
        19,
        "normal",
        2000,
        scenario=scenario,
        metrics_level="off",
    )
    assert report["status"] == "completed" and report["completed"] == 2
    assert not rewards.final_requests
    if method == "ppo":

        def forbidden(*args, **kwargs):
            raise AssertionError("true terminal must not evaluate bootstrap")

        monkeypatch.setattr("smartsom.learning.returns.values", forbidden)
    policy.update(rewards, terminal=True, final_critic=critic)
    assert policy.updates > 0


def test_numpy_reward_recording_uses_same_normalized_values(tmp_path):
    import json

    import numpy as np

    from smartsom.trace.current import CurrentRecording

    rewards = RewardTimeline()
    result = rewards.append_result(
        {
            "team": np.float32(1.5),
            "roles": {"central": np.float32(2.5)},
            "agents": {("central", "factory"): np.float32(3.5)},
        },
        1,
    )
    with (tmp_path / "rewards.jsonl").open("w") as output:
        recorder = CurrentRecording(tmp_path / "unused")
        recorder.reward_stream = output
        recorder.reward(0, 1, result)
    row = json.loads((tmp_path / "rewards.jsonl").read_text())
    assert (
        row["team"] + row["roles"]["central"] + row["agents"][0]["value"]
        == rewards.for_subject("central", "factory")[0]
    )


def test_monitor_keeps_learning_return_separate_from_physical_objective():
    from smartsom.telemetry.experiment_tracking import cloud_fields

    assert cloud_fields(
        {
            "learning_return": 12.0,
            "observed_return": -3.0,
            "agent_returns": {"machine/m1": 5.0},
        }
    ) == {"learning_return": 12.0, "observed_return": -3.0}


def test_illegal_candidate_is_rejected_before_reward_commit():
    options = configuration("dqn")
    policy = build_policy(options, 101)
    policy.choose = lambda request, critic: "invalid-candidate"
    with pytest.raises(ValueError, match="illegal"):
        run(options, policy, ticks=2)


def test_e1a_compact_identity_affine_matches_dense_values_and_gradients():
    from smartsom.controllers.e1a.network import Network

    torch.manual_seed(719)
    identities = torch.tensor(
        [[[0, 1, -1], [2, -1, 3], [-1, -1, -1]], [[3, 2, 1], [1, 1, 0], [2, 3, 0]]]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    batch = {
        "global": torch.randn(2, 7),
        "candidates": torch.randn(2, 3, 15),
        "associations": identities,
        "owner_count": 4,
        "mask": mask,
        "layouts": ["same", "same"],
    }
    compact = Network(False)
    actual = compact.actor(batch)
    dense = deepcopy(compact)
    one_hot = torch.zeros(2, 3, 3, 4)
    for b in range(2):
        for c in range(3):
            for slot in range(3):
                owner = int(identities[b, c, slot])
                if owner >= 0:
                    one_hot[b, c, slot, owner] = 1
    features = torch.cat(
        (
            batch["global"].unsqueeze(1).expand(-1, 3, -1),
            batch["candidates"],
            one_hot.flatten(-2),
        ),
        -1,
    )
    expected = dense.scorer(features).squeeze(-1).masked_fill(~mask, -torch.inf)
    assert torch.allclose(actual[mask], expected[mask], atol=1e-6, rtol=1e-5)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    actual[mask].square().mean().backward()
    expected[mask].square().mean().backward()
    for (name, left), (_, right) in zip(
        compact.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5), name
