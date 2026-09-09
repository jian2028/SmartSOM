"""Resource checkpoint, shared runner and ledger gates without learner imports."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_learning_projection import COMBINATIONS, module_case
from test_resource_projection import SPEC, indices

from smartsom.algorithms import SPTPolicy
from smartsom.config import resolve_training_run
from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.models import EpisodeBudget
from smartsom.config.training import episode_input, framework_seed
from smartsom.engine import SimulationResult, Simulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.evidence import write_json
from smartsom.experiments.resource_audit import audit_joint_run
from smartsom.learning.checkpoint import (
    CheckpointFile,
    ResourceCheckpointManifest,
    RoleWeights,
    file_hash,
    resolve_checkpoint_reference,
    structural_identity,
    validate_checkpoint,
)
from smartsom.learning.joint import JointActionCoordinator
from smartsom.learning.joint_evidence import round_record
from smartsom.learning.joint_replay import replay_joint
from smartsom.learning.resource_policy import ResourceCheckpointPolicy
from smartsom.learning.resources import ResourceProjection

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def base_only(monkeypatch):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda p: {})


def training():
    return resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")


def test_resource_scientific_sequence_matches_both_central_backends():
    resource = training()
    for name in ("rllib", "sb3"):
        central = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
        for index in range(5):
            assert resource.episode(index) == central.episode(index)
        assert resource.framework_seed != central.framework_seed
    assert resource.framework_seed == framework_seed(101, "rllib.resource_ppo")
    assert resource.run.budget.environment_steps == 4096


def bundle(tmp_path):
    r = training()
    spec = r.algorithm.algorithm
    projection = ResourceProjection(
        r.base.factory, spec.projection, transport_enabled=True
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    member = checkpoint / "weights.fixture"
    member.write_bytes(b"preflight fixture, not an inference model")
    weights = tuple(
        RoleWeights(role=role, initial_sha256="1" * 64, final_sha256="2" * 64)
        for role in ("agv_policy", "machine_policy")
    )
    metadata = ResourceCheckpointManifest(
        schema="smartsom.resource-checkpoint/v1",
        provider=spec.provider,
        projection=spec.projection,
        parameters=spec.parameters,
        structure_sha256=structural_identity(r.base, spec.projection),
        files=(CheckpointFile(path=member.name, sha256=file_hash(member)),),
        dependencies=(),
        initial_weights_sha256=digest({w.role: w.initial_sha256 for w in weights}),
        final_weights_sha256=digest({w.role: w.final_sha256 for w in weights}),
        role_weights=weights,
        role_mapping=tuple(
            (a, "machine_policy" if a.startswith("machine:") else "agv_policy")
            for a in projection.agents
        ),
        environment_steps=4096,
        agent_steps=4096 * 12,
        physical_actions=1,
        learner_updates=16,
        framework_seed=r.framework_seed,
    )
    write_json(checkpoint / "checkpoint.json", metadata)
    algorithm = r.algorithm.model_copy(
        update={"algorithm": spec.model_copy(update={"checkpoint": str(checkpoint)})}
    )
    algorithm = resolve_checkpoint_reference(algorithm, tmp_path / "algorithm.yaml")
    resolved = replace(
        r.base,
        algorithm=algorithm,
        run=r.base.run.model_copy(
            update={"budget": EpisodeBudget(), "output_root": str(tmp_path / "runs")}
        ),
    )
    return resolved, checkpoint


@pytest.mark.parametrize("field", ["role_mapping", "role_weights", "agent_steps"])
def test_checkpoint_roles_updates_and_counts_checked_before_execution(tmp_path, field):
    resolved, checkpoint = bundle(tmp_path)
    assert validate_checkpoint(resolved)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    metadata[field] = 1 if field == "agent_steps" else []
    write_json(checkpoint / "checkpoint.json", metadata)
    resolved = replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={
                "algorithm": resolved.algorithm.algorithm.model_copy(
                    update={
                        "checkpoint_sha256": file_hash(checkpoint / "checkpoint.json")
                    }
                )
            }
        ),
    )
    with pytest.raises(ConfigurationError, match="role/update/count"):
        run_one(resolved)
    assert not Path(resolved.run.output_root).exists()


def test_checkpoint_random_input_compatible_structure_and_corruption_rejected(tmp_path):
    resolved, checkpoint = bundle(tmp_path)
    other = training().episode(3).input
    assert validate_checkpoint(
        replace(resolved, arrivals=other.arrivals, quality=other.quality)
    )
    changed = replace(resolved.factory, machines=resolved.factory.machines[::-1])
    assert validate_checkpoint(replace(resolved, factory=changed))
    changed = replace(resolved.algorithm.algorithm.projection, time_scale=101)
    incompatible = replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={
                "algorithm": resolved.algorithm.algorithm.model_copy(
                    update={"projection": changed}
                )
            }
        ),
    )
    with pytest.raises(ConfigurationError, match="incompatible"):
        validate_checkpoint(incompatible)
    (checkpoint / "weights.fixture").write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="digest"):
        run_one(resolved)
    assert not Path(resolved.run.output_root).exists()


def inject_predictor(monkeypatch, predictor):
    original = ResourceCheckpointPolicy.__init__
    monkeypatch.setattr(
        ResourceCheckpointPolicy,
        "__init__",
        lambda self, resolved: original(self, resolved, predictor=predictor),
    )


def spt_proposals(decision):
    return indices(decision, SPTPolicy().select_action(decision.context))


def test_shared_runner_joint_replay_and_tamper_detection(tmp_path, monkeypatch):
    resolved, _ = bundle(tmp_path)
    inject_predictor(monkeypatch, spt_proposals)
    result = run_one(resolved)
    audited = audit_joint_run(result.run_dir)
    assert audited["status"] == "passed"
    assert audited["team_return"] == -result.simulation_result.makespan
    ledger = result.run_dir / "joint_decisions.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    for field, bad in (("reward", 100), ("trace_end", 100), ("round", 10)):
        changed = copy.deepcopy(records)
        changed[0][field] = bad
        with pytest.raises(ValueError, match="joint replay record"):
            replay_joint(
                episode_input(resolved),
                resolved.algorithm.algorithm.projection,
                changed,
            )
    changed = copy.deepcopy(records)
    changed[0]["agents"][0]["observations_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="joint replay record"):
        replay_joint(
            episode_input(resolved), resolved.algorithm.algorithm.projection, changed
        )


def test_runner_stalled_and_writer_failure_preserve_original_cause(
    tmp_path, monkeypatch
):
    resolved, _ = bundle(tmp_path)
    inject_predictor(monkeypatch, lambda d: {v.agent_id: 0 for v in d.views})
    with pytest.raises(RunFailedError, match="policy_stalled") as failure:
        run_one(resolved)
    path = failure.value.run_dir
    records = [
        json.loads(s) for s in (path / "joint_decisions.jsonl").read_text().splitlines()
    ]
    audited = replay_joint(
        episode_input(resolved), resolved.algorithm.algorithm.projection, records
    )
    assert audited.reason == "policy_stalled" and audited.total_reward == -10001
    assert json.loads((path / "summary.json").read_text())["makespan"] is None
    # A failed ledger writer does not replace the actual core trace with success.
    monkeypatch.setattr(
        "smartsom.experiments.runner.append_json",
        lambda *a: (_ for _ in ()).throw(OSError("joint writer failed")),
    )
    with pytest.raises(RunFailedError, match="joint writer failed") as failure:
        run_one(resolved)
    assert (failure.value.run_dir / "failure.json").exists()


@pytest.mark.parametrize("flags", COMBINATIONS)
def test_all_combinations_framework_free_joint_ledger(flags):
    inp = module_case(*flags)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    projection = ResourceProjection(
        inp.factory, SPEC, transport_enabled=inp.transport_enabled
    )
    records, cursor, previous = [], 0, 0
    while sim.current_decision is not None:
        decision = projection.project(sim.current_decision)
        choices = spt_proposals(decision)
        coordinator = JointActionCoordinator(decision, choices)
        outcome = coordinator.execute(sim)
        done = isinstance(outcome, SimulationResult)
        tick = outcome.makespan if done else outcome.simulation_time
        end = cursor + len(sim.trace_since(cursor))
        records.append(
            round_record(
                round_index=len(records),
                decision=decision,
                indices=choices,
                proposals=tuple(coordinator.records),
                actions=tuple(coordinator.actions),
                reward=float(previous - tick),
                tick=tick,
                reason="completed" if done else None,
                trace_start=cursor,
                trace_end=end,
            )
        )
        cursor, previous = end, tick
    audited = replay_joint(inp, SPEC, primitive(records))
    assert audited.result == outcome and audited.total_reward == -outcome.makespan
