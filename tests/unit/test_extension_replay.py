"""Actual checkpoint-policy public inputs, state evolution, and failure prefixes."""

import copy
import json
from dataclasses import replace

import pytest
from test_extensions import CountingEncoder, CountingReward, require_optional
from test_resource_training import bundle as resource_bundle
from test_resource_training import spt_proposals
from test_training import bundle as central_bundle

from smartsom.config.codec import digest, primitive
from smartsom.config.extensions import ExtensionRef, ExtensionSpec, RewardSpec
from smartsom.config.training import episode_input
from smartsom.dispatch import DecisionContext
from smartsom.engine import Simulator
from smartsom.experiments.evidence import write_json
from smartsom.learning.checkpoint import (
    CheckpointPolicy,
    file_hash,
    structural_identity,
)
from smartsom.learning.extension_evidence import wire_observation
from smartsom.learning.extension_replay import replay_extensions
from smartsom.learning.extensions import (
    ExtensionsRuntime,
    bind_extensions,
    central_observation,
    register_extension,
    resource_observation,
)
from smartsom.learning.projection import LearningProjection
from smartsom.learning.resource_policy import ResourceCheckpointPolicy
from smartsom.learning.resources import ResourceProjection


def extended_bundle(tmp_path, resource, *, stateful=False, problem=None):
    resolved, checkpoint = (resource_bundle if resource else central_bundle)(tmp_path)
    if problem is not None:
        factory, workload = problem
        resolved = replace(
            resolved,
            factory=factory,
            workload=workload,
            arrivals=None,
            processing_times=None,
            machine_events=None,
            quality=None,
            holding_buffer_enabled=False,
            scenario=resolved.scenario.model_copy(
                update={"arrivals": None, "quality": None, "holding_buffer": None}
            ),
        )
    spec = resolved.algorithm.algorithm
    if stateful:
        register_extension(
            "observation", "tests.replay-counter", "1", CountingEncoder, stateful=True
        )
        register_extension(
            "reward", "tests.replay-reward", "1", CountingReward, stateful=True
        )
    extensions = bind_extensions(
        ExtensionSpec(
            observation=ExtensionRef(
                name="tests.replay-counter" if stateful else "builtin.dict", version="1"
            ),
            reward=RewardSpec(
                team=ExtensionRef(name="tests.replay-reward", version="1")
                if stateful
                else ExtensionRef(
                    name="builtin.reward_scale",
                    version="1",
                    parameters={"scale": 2.0, "terminal_offset": 5.0},
                ),
                roles={
                    "machine_policy": ExtensionRef(
                        name="builtin.reward_scale",
                        version="1",
                        parameters={"scale": 3.0, "terminal_offset": 7.0},
                    )
                }
                if resource
                else {},
                learner_scale=0.1,
            ),
        ),
        spec.provider,
    )
    context = DecisionContext(0, (), (), ())
    if resource:
        projection = ResourceProjection(
            resolved.factory,
            spec.projection,
            transport_enabled=resolved.transport_enabled,
        )
        decision = projection.project(context)
        layouts = {
            view.role: resource_observation(context, view).layout()
            for view in decision.views
        }
    else:
        projection = LearningProjection(resolved.factory, spec.projection)
        layouts = {
            None: central_observation(context, projection.project(context)).layout()
        }
    runtime = ExtensionsRuntime(
        extensions,
        spec.provider,
        layouts,
        learner_scale=spec.parameters.learner_reward_scale if resource else 1.0,
    )
    write_json(checkpoint / "extension_state.json", runtime.state_dict())
    entry = {
        "path": "extension_state.json",
        "sha256": file_hash(checkpoint / "extension_state.json"),
    }
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    manifest.update(extensions=primitive(extensions), extension_state=entry)
    manifest["structure_sha256"] = structural_identity(resolved, spec.projection)
    if resource:
        manifest["role_mapping"] = [
            (view.agent_id, view.role) for view in decision.views
        ]
        manifest["agent_steps"] = manifest["environment_steps"] * len(decision.views)
    manifest["files"].append(entry)
    write_json(checkpoint / "checkpoint.json", manifest)
    algorithm = resolved.algorithm.model_copy(
        update={
            "algorithm": spec.model_copy(
                update={
                    "extensions": extensions,
                    "checkpoint_sha256": file_hash(checkpoint / "checkpoint.json"),
                }
            )
        }
    )
    return replace(resolved, algorithm=algorithm)


@pytest.mark.parametrize("resource", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_extension_policy_actual_input_replay_and_tamper(
    tmp_path, monkeypatch, resource, stateful, complete
):
    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )
    resolved = extended_bundle(tmp_path, resource, stateful=stateful)
    if not complete:
        resolved = replace(
            resolved,
            run=resolved.run.model_copy(
                update={
                    "budget": resolved.run.budget.model_copy(
                        update={"max_decisions": 3}
                    )
                }
            ),
        )
    records, rewards = [], []
    if resource:
        policy = ResourceCheckpointPolicy(
            resolved,
            predictor=lambda decision: spt_proposals(decision.mapping),
            on_extension=records.append,
            on_reward=rewards.append,
        )
    else:
        policy = CheckpointPolicy(
            resolved,
            predictor=lambda view: next(
                i for i, action in enumerate(view.actions) if action is not None
            ),
            on_extension=records.append,
            on_reward=rewards.append,
        )
    inp = episode_input(resolved)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    try:
        while sim.current_decision is not None:
            if resource:
                policy.observed_trace_end = len(sim.trace)
            outcome = sim.step(policy.select_action(sim.current_decision))
            if resource:
                policy.check_outcome(outcome, trace_end=len(sim.trace))
            else:
                policy.check_outcome(outcome)
    except RuntimeError as exc:
        assert not complete and "budget_exhausted" in str(exc)
    assert bool(sim.current_decision is None) == complete
    records = primitive(records)
    rewards = primitive(rewards)
    trace = primitive(sim.trace)
    result = replay_extensions(
        resolved, records, trace, complete=complete, reward_records=rewards
    )
    assert result["status"] == ("passed" if complete else "partial_verified")
    assert result["decisions"] == len(records) > 0
    assert result["rewards"] == len(rewards) == len(records)
    assert rewards[-1]["reason"] == ("completed" if complete else "budget_exhausted")
    assert sum(row["team"]["raw"] for row in rewards) == policy.total_reward
    assert rewards[-1]["team"]["research"] != rewards[-1]["team"]["raw"]
    assert records[0]["views"][0]["observations_sha256"]
    if stateful:
        assert records[0]["state_before_sha256"] != records[0]["state_after_sha256"]
        assert records[1]["state_before_sha256"] == rewards[0]["state_after_sha256"]
        assert rewards[0]["state_before_sha256"] == records[0]["state_after_sha256"]
    for key in (
        "context_sha256",
        "checkpoint_sha256",
        "extensions_sha256",
        "state_before_sha256",
        "state_after_sha256",
    ):
        changed = copy.deepcopy(records)
        changed[0][key] = "0" * 64
        with pytest.raises(ValueError, match="replay mismatch"):
            replay_extensions(resolved, changed, trace, complete=complete)
    for key in ("observations_sha256", "mask_sha256"):
        changed = copy.deepcopy(records)
        changed[0]["views"][0][key] = "0" * 64
        with pytest.raises(ValueError, match="replay mismatch"):
            replay_extensions(resolved, changed, trace, complete=complete)
    with pytest.raises(ValueError, match="physical trace|coverage"):
        replay_extensions(resolved, records[:-1], trace, complete=complete)
    changed = copy.deepcopy(records)
    changed[0]["views"].append(changed[0]["views"][0])
    with pytest.raises(ValueError, match="duplicate|exactly one"):
        replay_extensions(resolved, changed, trace, complete=complete)
    changed_rewards = copy.deepcopy(rewards)
    changed_rewards[-1]["team"]["learner"] += 1
    with pytest.raises(ValueError, match="reward replay mismatch"):
        replay_extensions(
            resolved, records, trace, complete=complete, reward_records=changed_rewards
        )
    with pytest.raises(ValueError, match="reward replay mismatch|reward coverage"):
        replay_extensions(
            resolved, records, trace, complete=complete, reward_records=rewards[:-1]
        )


def test_float32_evidence_hashes_the_wire_value_and_rejects_overflow():
    value = {"public": (0.1, 16777217)}
    assert wire_observation(value) == {"public": (0.10000000149011612, 16777216.0)}
    assert digest(wire_observation(value)) != digest(value)
    for bad in (float("inf"), float("nan"), 1e100):
        with pytest.raises(ValueError, match="float32"):
            wire_observation([bad])


def test_outer_run_auditor_requires_and_checks_extension_evidence(
    tmp_path, monkeypatch
):
    from smartsom.experiments.audit import audit_run
    from smartsom.experiments.evidence import artifact_digests
    from smartsom.experiments.runner import run_one

    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )
    resolved = extended_bundle(tmp_path, False)
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(update={"output_root": str(tmp_path / "runs")}),
    )
    records, rewards = [], []
    original = CheckpointPolicy.__init__

    def initialize(self, resolved, **kwargs):
        kwargs.setdefault(
            "predictor",
            lambda view: next(
                i for i, action in enumerate(view.actions) if action is not None
            ),
        )
        kwargs.setdefault("on_extension", records.append)
        kwargs.setdefault("on_reward", rewards.append)
        original(self, resolved, **kwargs)

    monkeypatch.setattr(CheckpointPolicy, "__init__", initialize)
    result = run_one(resolved)
    directory = result.run_dir
    ledger = directory / "extension_decisions.jsonl"
    # The integration runner owns file wiring; exercise the same artifact contract
    # in this isolated extension checkout until that independent patch is merged.
    if not ledger.exists():
        ledger.write_text("".join(json.dumps(primitive(row)) + "\n" for row in records))
    reward_ledger = directory / "extension_rewards.jsonl"
    if not reward_ledger.exists():
        reward_ledger.write_text(
            "".join(json.dumps(primitive(row)) + "\n" for row in rewards)
        )

    def refresh_inventory():
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["artifacts"] = artifact_digests(directory)
        write_json(directory / "manifest.json", manifest)

    refresh_inventory()
    audited = audit_run(directory)
    assert audited["status"] == "passed"
    assert audited["extension_replay"]["status"] == "passed"
    raw = ledger.read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows[0]["views"][0]["observations_sha256"] = "0" * 64
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    refresh_inventory()
    with pytest.raises(ValueError, match="extension decision replay mismatch"):
        audit_run(directory)
    ledger.unlink()
    refresh_inventory()
    with pytest.raises(ValueError, match="missing extension"):
        audit_run(directory)


@pytest.mark.parametrize("resource", [False, True])
def test_real_deadlock_rewards_match_training_environment_and_replay(
    tmp_path, monkeypatch, resource
):
    require_optional("gymnasium")
    if resource:
        require_optional("pettingzoo")
    from test_buffers import dispatch, move, vehicle_case
    from test_resource_projection import indices

    from smartsom.domain import MachineBuffers
    from smartsom.engine import DeadlockError
    from smartsom.learning.gymnasium import SchedulingEnv
    from smartsom.learning.pettingzoo import SmartSOMParallelEnv

    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )
    factory, workload, _ = vehicle_case()
    factory = replace(
        factory,
        buffers=(MachineBuffers("M1", 0, 0),),
        transport=replace(factory.transport, agvs=factory.transport.agvs[:1]),
    )
    resolved = extended_bundle(
        tmp_path, resource, stateful=True, problem=(factory, workload)
    )
    spec = resolved.algorithm.algorithm
    actions = (move("A", "M1", "V1"), dispatch("A"), move("B", "M1", "V1"))
    selected = iter(actions)
    records, rewards = [], []
    policy_type = ResourceCheckpointPolicy if resource else CheckpointPolicy
    policy = policy_type(
        resolved,
        predictor=(lambda view: indices(view.mapping, next(selected)))
        if resource
        else (lambda view: view.actions.index(next(selected))),
        on_extension=records.append,
        on_reward=rewards.append,
    )
    inp = episode_input(resolved)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    with pytest.raises(DeadlockError):
        while sim.current_decision is not None:
            before = len(sim.trace)
            context = sim.current_decision
            try:
                outcome = sim.step(policy.select_action(context))
                if resource:
                    policy.check_outcome(outcome, trace_end=len(sim.trace))
                else:
                    policy.check_outcome(outcome)
            except DeadlockError as exc:
                policy.fail(
                    exc,
                    tick=max(row.simulation_time for row in sim.trace_since(before)),
                    trace_end=len(sim.trace),
                )
                raise
    assert rewards[-1]["reason"] == "deadlock"
    assert (
        replay_extensions(
            resolved,
            primitive(records),
            primitive(sim.trace),
            complete=False,
            reward_records=primitive(rewards),
        )["status"]
        == "partial_verified"
    )
    if resource:
        env = SmartSOMParallelEnv(
            inp,
            spec.projection,
            limits=resolved.run.budget.limits(),
            extensions=spec.extensions,
            learner_scale=spec.parameters.learner_reward_scale,
        )
    else:
        env = SchedulingEnv(
            inp,
            spec.projection,
            limits=resolved.run.budget.limits(),
            observation_kind="plain",
            provider=spec.provider,
            extensions=spec.extensions,
        )
    env.reset()
    for action, expected in zip(actions, rewards, strict=True):
        choice = (
            indices(env.projected, action)
            if resource
            else env.projected.actions.index(action)
        )
        env.step(choice)
        step = env.steps[-1]
        if resource:
            assert primitive(step.reward_values.team) == primitive(expected["team"])
            assert primitive(step.reward_values.roles) == primitive(expected["roles"])
        else:
            assert (step.reward, step.research_reward, step.learner_reward) == tuple(
                getattr(expected["team"], key) for key in ("raw", "research", "learner")
            )
    assert env.reason == "deadlock" and env.total_reward == -10001
    assert primitive(env.simulator.trace) == primitive(sim.trace)


def test_resource_stalled_reward_is_terminal_and_replays(tmp_path, monkeypatch):
    from smartsom.learning.joint import PolicyStalledError

    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )
    resolved = extended_bundle(tmp_path, True, stateful=True)
    records, rewards = [], []
    policy = ResourceCheckpointPolicy(
        resolved,
        predictor=lambda decision: {view.agent_id: 0 for view in decision.views},
        on_extension=records.append,
        on_reward=rewards.append,
    )
    inp = episode_input(resolved)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    with pytest.raises(PolicyStalledError):
        while sim.current_decision is not None:
            policy.observed_trace_end = len(sim.trace)
            outcome = sim.step(policy.select_action(sim.current_decision))
            policy.check_outcome(outcome, trace_end=len(sim.trace))
    assert rewards[-1]["reason"] == "policy_stalled"
    assert (
        replay_extensions(
            resolved,
            primitive(records),
            primitive(sim.trace),
            complete=False,
            reward_records=primitive(rewards),
        )["status"]
        == "partial_verified"
    )


@pytest.mark.parametrize("resource", [False, True])
def test_initial_failure_notification_is_recorded_once_and_not_falsely_replayed(
    tmp_path, monkeypatch, resource
):
    from smartsom.engine import DeadlockError

    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend", lambda provider: {}
    )
    resolved = extended_bundle(tmp_path, resource, stateful=True)
    rewards = []
    policy = (ResourceCheckpointPolicy if resource else CheckpointPolicy)(
        resolved, predictor=lambda view: None, on_reward=rewards.append
    )
    error = DeadlockError("initialization failed")
    policy.fail(error, tick=0, trace_end=0)
    policy.fail(error, tick=0, trace_end=0)
    assert len(rewards) == 1
    assert rewards[0]["decision_index"] is None
    assert rewards[0]["reason"] == "deadlock"
    assert rewards[0]["team"].raw == policy.total_reward == -10001
    # This fixture's actual world can initialize: a claimed initialization
    # deadlock must therefore be rejected by physical replay.
    with pytest.raises(ValueError, match="reward coverage|physical trace"):
        replay_extensions(
            resolved, [], [], complete=False, reward_records=primitive(rewards)
        )
