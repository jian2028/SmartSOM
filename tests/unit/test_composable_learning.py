"""Physical-time math and real backend/continuation integration."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json, primitive
from smartsom.config.experiment import prepare
from smartsom.experiments.composable import (
    TrainingSession,
    allocate,
    evaluation_recipe,
    policies_for,
    prepared_from_run,
)
from smartsom.learning.production_collection import Replay, physical_gae

ROOT = Path(__file__).resolve().parents[2]


def test_physical_time_gae_termination_and_bootstrap():
    rows = [
        {"reward": 2.0, "value": 1.0, "dt": 5, "terminated": False},
        {"reward": 3.0, "value": 4.0, "dt": 1, "terminated": True},
    ]
    result = physical_gae(rows, 1000, 0.5, 1)
    assert result[1]["advantage"] == -1
    assert result[0]["return"] == pytest.approx(2 + 0.5**5 * 3)
    truncated = physical_gae(
        [{"reward": 2.0, "value": 1.0, "dt": 1, "terminated": False}], 4, 0.5, 1
    )
    assert truncated[0]["return"] == 4


def test_uniform_replay_restores_sampling_order():
    replay = Replay(3, 23)
    for i in range(6):
        replay.add({"x": i})
    restored = Replay(3, 1)
    restored.load_state_dict(replay.state_dict())
    assert replay.sample(3) == restored.sample(3)
    assert {r["x"] for r in replay.rows} == {3, 4, 5}


def tiny(name, root, ticks=24, *, envs=1, sampling=0):
    config = api.load_config(ROOT / "configs/test/runs" / (name + ".yaml"))
    config.training.total_ticks = ticks
    config.training.ticks_per_update = ticks // 2
    config.validation.enabled = False
    config.output.root = str(root)
    config.scenario_overrides["tick_limit"] = 48
    config.runtime.num_envs = envs
    config.runtime.sampling_processes = sampling
    prepared = prepare(config)
    parameters = json.loads(prepared.parameters_json)
    if config.training.algorithm == "dqn":
        parameters.update(batch_size=2, warmup_ticks=0, target_update_ticks=4)
    else:
        parameters.update(batch_size=4, n_epochs=1)
    return replace(prepared, parameters_json=canonical_json(parameters))


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
def test_parallel_sampling_uses_two_envs_and_restores_complete_wave(name, tmp_path):
    pytest.importorskip("ray")
    prepared = tiny(name, tmp_path, ticks=8, envs=2, sampling=2)
    root, record, frozen = allocate(prepared, "training")
    first = TrainingSession(frozen, root, record)
    try:
        first.step_update()
        assert first.parallel_sampling
        assert [row["env"] for row in first.actions] == [0, 1, 0, 1]
        with (root / "checkpoints/update-000001/continuation.pkl").open("rb") as stream:
            saved = __import__("pickle").load(stream)
        first.execute()
        expected = copy.deepcopy(first.actions)
        expected_states = copy.deepcopy(first.sampler_states)
    finally:
        first.close()
    resumed = TrainingSession(prepared_from_run(root), root, record)
    try:
        resumed.restore(saved)
        resumed.execute()
        assert resumed.actions == expected
        assert len(resumed.sampler_states) == 2
        for original, restored in zip(
            expected_states, resumed.sampler_states, strict=True
        ):
            assert original.keys() == restored.keys()
    finally:
        resumed.close()


@pytest.mark.parametrize(
    "name", ["train_all_ppo", "train_all_dqn", "central_rllib_ppo", "central_sb3_ppo"]
)
def test_real_update_package_roundtrip_and_exact_resume(name, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    if "sb3" in name:
        pytest.importorskip("sb3_contrib")
    prepared = tiny(name, tmp_path)
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    partial = session.execute(stop_after_updates=1)
    assert partial.status == "interrupted"
    assert sum(session.optimizations.values()) > 0
    saved_path = partial.last_checkpoint / "continuation.pkl"
    import pickle

    with saved_path.open("rb") as stream:
        saved = pickle.load(stream)
    session.execute()
    continuous_actions = copy.deepcopy(session.actions)
    continuous_weights = {
        g: {k: v.clone() for k, v in p.network.state_dict().items()}
        for g, p in session.policies.items()
        if hasattr(p, "network")
    }
    resumed = TrainingSession(prepared_from_run(root), root, record)
    resumed.restore(saved)
    resumed.execute()
    assert resumed.actions == continuous_actions
    for g, weights in continuous_weights.items():
        assert all(
            torch.equal(v, resumed.policies[g].network.state_dict()[k])
            for k, v in weights.items()
        )
    assert resumed.target_clock == session.target_clock
    assert resumed.optimizations == session.optimizations
    if session.replays:
        for g in session.replays:
            assert resumed.replays[g].state_dict() == session.replays[g].state_dict()
    assert all(
        resumed.policies[g].fingerprint() == original
        for g, original in resumed.frozen.items()
    )
    from smartsom.experiments.composable import checkpoint_path, export

    checkpoint = checkpoint_path(root, "last")
    infer, _ = policies_for(evaluation_recipe(frozen, checkpoint))
    for g, policy in infer.items():
        if hasattr(policy, "network"):
            assert all(
                torch.equal(v, policy.network.state_dict()[k])
                for k, v in continuous_weights[g].items()
            )
    group = None if name.startswith("central") else "mover"
    exported = export(root, tmp_path / (name + ".zip"), group=group)
    from smartsom.learning.production_inference import read_package

    metadata, weights, encoder = read_package(exported)
    assert metadata["action_contract"].endswith("/v3")
    assert weights and encoder["schema"]
    from smartsom.experiments.evidence import source_identity

    assert record["source"]["git"] == source_identity()["git"]


def test_double_dqn_masks_targets_and_physical_intervals():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from smartsom.learning.production_rllib_v3 import PhysicalDQNLearner

    class Scores:
        def __init__(self, value):
            self.value = torch.tensor(value, dtype=torch.float32)

        def __call__(self, obs):
            return self.value, None

    class Module:
        network = Scores([[9, -1e30, 2]] * 4)
        target = Scores([[4, 1e30, 7]] * 4)

    class Metrics:
        def log_dict(self, *args, **kwargs):
            pass

    learner = object.__new__(PhysicalDQNLearner)
    learner._module = {"default_policy": Module()}
    learner.metrics = Metrics()
    q = torch.tensor([[0.0]] * 4, requires_grad=True)
    batch = {
        "actions": torch.tensor([0] * 4),
        "next_obs": {},
        "rewards": torch.tensor([1.0] * 4),
        "discount": torch.tensor([0.5**d for d in (0, 1, 5, 1)]),
        "terminated": torch.tensor([False, False, False, True]),
    }
    loss = learner.compute_loss_for_module(
        module_id="default_policy", config=None, batch=batch, fwd_out={"q": q}
    )
    expected = torch.nn.functional.smooth_l1_loss(
        q[:, 0], torch.tensor([5.0, 3.0, 1.125, 1.0])
    )
    assert loss == expected


def test_mixed_checkpoints_zip_and_frozen_partners(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    import yaml

    from smartsom.experiments.composable import checkpoint_path, export

    prepared = tiny("train_all_ppo", tmp_path / "source")
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    session.execute()
    zip_path = export(root, tmp_path / "dispatcher.zip", group="dispatcher")
    composition = json.loads(frozen.composition_json)
    composition.pop("matching")
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    selections = {
        "machine": {
            "source": str(root),
            "checkpoint": "update-000001",
            "group": "machine",
        },
        "buffer": {"source": str(root), "checkpoint": "last", "group": "buffer"},
        "dispatcher": {"source": str(zip_path)},
    }
    for role, selector in selections.items():
        path = policy_dir / (role + ".yaml")
        path.write_text(
            yaml.safe_dump(
                {
                    "schema": "smartsom.policy/v1",
                    "role": role,
                    "implementation": {"kind": "model", "model": selector},
                }
            )
        )
        composition["groups"][role]["policy"] = str(path)
    composition["groups"]["mover"]["policy"] = str(
        ROOT / "configs/test/policies/mover_new_ppo.yaml"
    )
    composition["pickup_matching"] = str(
        ROOT / "configs/test/rules/pickup_global_optimal.yaml"
    )
    comp_path = tmp_path / "mixed.yaml"
    comp_path.write_text(yaml.safe_dump(composition))
    config = prepared.config
    config.composition = str(comp_path)
    config.training.groups = ("mover",)
    config.output.root = str(tmp_path / "mixed")
    mixed = prepare(config)
    mixed = replace(mixed, parameters_json=prepared.parameters_json)
    target, meta, mixed = allocate(mixed, "training")
    training = TrainingSession(mixed, target, meta)
    before = {g: training.policies[g].fingerprint() for g in selections}
    training.execute()
    assert all(training.policies[g].fingerprint() == before[g] for g in selections)
    assert meta["changed_weights"]["mover"]
    # The run has its own dependency closure; it no longer reads source files.
    import shutil

    shutil.rmtree(root)
    zip_path.unlink()
    inference, _ = policies_for(
        evaluation_recipe(prepared_from_run(target), checkpoint_path(target, "last"))
    )
    assert all(inference[g].fingerprint() == before[g] for g in selections)
    resumed = api.resume(target)
    assert resumed.environment_steps == training.ticks
    assert resumed.status == "completed"


def test_ppo_prefix_probability_is_joint_and_clipped_once():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    import numpy as np

    from smartsom.config.experiment_v3 import PPOParameters
    from smartsom.learning.production_models import default_network, packet_arrays
    from smartsom.learning.production_rllib_v3 import build_learner

    params = primitive(PPOParameters()).copy()
    params.update(entropy_coefficient=0.0, value_coefficient=0.0)
    learner = build_learner(1, default_network("ppo"), "ppo", params)

    class Scores(torch.nn.Module):
        def forward(self, obs):
            logits = obs["candidates"][:, :, 0].clone().requires_grad_(True)
            return logits.masked_fill(~obs["mask"], -torch.inf), obs["context"][:, 0]

    module = learner.module["default_policy"]
    module.network = Scores()

    def encoded(values):
        candidates = np.zeros((2, 16), np.float32)
        candidates[:, 0] = np.log(values)
        return {
            "context": [0.0],
            "candidates": candidates.tolist(),
            "prefix": [],
            "mask": [True, True],
        }

    expected = __import__("math").log(2 / 5) + __import__("math").log(7 / 12)
    row = {
        "inputs": [encoded([2, 3]), encoded([5, 7])],
        "actions": [0, 1],
        "value_input": encoded([1, 1]),
        "log_probability": expected - __import__("math").log(2),
        "actor_mask": True,
        "advantage": 1.0,
        "return": 0.0,
    }
    raw = packet_arrays([row])
    batch = {
        k: {n: torch.as_tensor(v) for n, v in val.items()}
        if isinstance(val, dict)
        else torch.as_tensor(val)
        for k, val in raw.items()
    }
    output = module._forward_train(batch)
    assert float(output["joint_log_probability"][0].detach()) == pytest.approx(expected)
    loss = learner.compute_loss_for_module(
        module_id="default_policy", config=learner.config, batch=batch, fwd_out=output
    )
    assert float(loss.detach()) == pytest.approx(-1.2)


def test_candidate_padding_does_not_truncate_and_illegal_high_q_is_masked():
    pytest.importorskip("torch")
    import numpy as np

    from smartsom.learning.production_models import (
        CandidateNetwork,
        default_network,
        tensor_inputs,
    )

    values = {
        "context": [0.0],
        "candidates": np.zeros((137, 16)).tolist(),
        "prefix": np.zeros((35, 16)).tolist(),
        "mask": [True] * 136 + [False],
    }
    encoded = tensor_inputs([values])
    assert encoded["candidates"].shape == (1, 137, 16)
    assert encoded["prefix"].shape == (1, 35, 16)
    network = CandidateNetwork(1, default_network("dqn"), "rllib.resource_dqn", "dqn")
    scores, _ = network(encoded)
    assert scores[0, -1] < -1e30
    assert int(scores.argmax(-1)) != 136


def test_public_encoder_distinguishes_routes_and_resource_semantics():
    pytest.importorskip("torch")
    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.engine.production import ProductionSimulator
    from smartsom.learning.production_models import PublicEncoder

    prepared = (
        compile_experiment(ROOT / "configs/test/runs/all_rules_v4.yaml")
        .entries[0]
        .prepared
    )
    sim = ProductionSimulator(prepared.scenario, contract="v3")
    encoder = PublicEncoder(
        sim.factory,
        {"time_scale": 100.0, "count_scale": 100.0, "max_jobs": 64},
        role="buffer",
    )
    jobs = list(sim.jobs)[:2]
    view = sim.protocol.public_view()
    left = encoder.semantic_features(jobs[0], view)
    right = encoder.semantic_features(jobs[1], view)
    assert left != right
    assert encoder.candidate_width > 16
    assert encoder.capacities[sim.pre[next(iter(sim.machines))]] == 4


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("active", [False, True])
def test_machine_samples_and_real_updates_are_reported_honestly(
    tmp_path, algorithm, active
):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    prepared = tiny("train_machine_" + algorithm, tmp_path, ticks=24 if active else 2)
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    if active:
        # Hand-placed PRE inventory exercises learning without depending on
        # transport exploration. Ordinary authoring still starts with empty PRE.
        sim = session.sims[0]
        job = next(iter(sim.jobs))
        operation = sim.demands[sim.jobs[job]["demand"]].steps[0].operation_type
        machine = next(
            m for m, r in sim.machines.items() if operation in r.operation_types
        )
        jobs = [
            j
            for j, row in sim.jobs.items()
            if sim.demands[row["demand"]].steps[0].operation_type == operation
        ]
        assert len(jobs) >= 3
        for j in jobs[:3]:
            sim._remove(j)
            sim._place(j, sim.pre[machine], sim._free(sim.pre[machine]))
    session.execute()
    counts = session.collector.counts["machine"]
    if active:
        assert counts["decisions"] >= 3
        assert counts["training_samples"] >= 2
        assert session.optimizations["machine"] > 0
        assert record["changed_weights"]["machine"]
    else:
        assert counts["decisions"] == counts["training_samples"] == 0
        assert session.optimizations["machine"] == 0
        assert not record["changed_weights"]["machine"]


@pytest.mark.parametrize("sampling", [0, 2])
def test_terminal_tick_ledger_async_envs_resume_and_partial_episode(tmp_path, sampling):
    pytest.importorskip("ray")
    prepared = tiny("train_all_ppo", tmp_path, ticks=8, envs=2, sampling=sampling)
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    for index, sim in enumerate(session.sims):
        sim.scenario = replace(sim.scenario, tick_limit=index + 2)
    try:
        session.step_update()
        # Env 0 ends on its second tick; env 1's partial episode is not archived.
        assert [
            (r["env"], r["episode"], r["training_physical_ticks"])
            for r in session.episode_results
        ] == [(0, 0, 3)]
        saved = copy.deepcopy(session.state_dict())
        session.execute()
        expected = copy.deepcopy(session.episode_results)
        assert [
            (r["env"], r["episode"], r["training_physical_ticks"]) for r in expected
        ] == [(0, 0, 3), (1, 0, 6)]
        assert all(r["truncated"] and not r["completed"] for r in expected)
        assert (
            json.loads((root / "reports/training-episodes.json").read_text())
            == expected
        )
    finally:
        session.close()
    resumed = TrainingSession(prepared_from_run(root), root, record)
    try:
        resumed.restore(saved)
        resumed.execute()
        assert resumed.episode_results == expected
        # Restored terminal env 0 resets without archiving the same episode again.
        assert len({(r["env"], r["episode"]) for r in resumed.episode_results}) == len(
            expected
        )
    finally:
        resumed.close()


def test_ledger_survives_optimizer_failure_and_last_tick(tmp_path, monkeypatch):
    pytest.importorskip("ray")
    prepared = tiny("train_all_ppo", tmp_path, ticks=4)
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    session.sims[0].scenario = replace(session.sims[0].scenario, tick_limit=2)

    def fail():
        raise RuntimeError("optimizer failed after terminal tick")

    monkeypatch.setattr(session, "optimize_ppo", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        session.execute()
    assert session.record["status"] == "failed"
    rows = json.loads((root / "reports/training-episodes.json").read_text())
    assert len(rows) == 1 and rows[0]["training_physical_ticks"] == 2


@pytest.mark.parametrize("sampling", [0, 2])
def test_ledger_includes_episode_ending_on_final_training_tick(tmp_path, sampling):
    pytest.importorskip("ray")
    prepared = tiny("train_all_ppo", tmp_path, ticks=8, envs=2, sampling=sampling)
    config = prepared.config
    config.training.max_ticks = 2
    prepared = replace(prepared, config_json=canonical_json(primitive(config)))
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    session.execute()
    assert [
        (r["env"], r["episode"], r["training_physical_ticks"])
        for r in session.episode_results
    ] == [(0, 0, 3), (1, 0, 4), (0, 1, 7), (1, 1, 8)]
    assert (
        json.loads((root / "reports/training-episodes.json").read_text())
        == session.episode_results
    )
