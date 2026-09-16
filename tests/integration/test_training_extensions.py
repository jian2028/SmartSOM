"""Three actual PPOs retain custom observation/reward state across complete resume."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_sampling import (
    assert_state_equal,
    checkpoint_rows,
    optimizer_and_rng,
    small_training,
)

from smartsom.config.experiment import ExperimentConfig, prepare
from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionRef,
    ExtensionSpec,
    NetworkBranch,
    NetworkSpec,
    RewardSpec,
)
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls, ValidationControls
from smartsom.learning.extensions import register_extension

pytestmark = pytest.mark.learning


class CountedPublicObservation:
    def __init__(self, parameters, layout):
        assert not parameters
        self.output_space = layout.grouped
        self.calls = self.episodes = 0

    def begin_episode(self):
        self.episodes += 1

    def encode(self, public):
        self.calls += 1

        def scale(value):
            if isinstance(value, (tuple, list)):
                return tuple(scale(child) for child in value)
            return value * (1 + 0.00001 * self.calls)

        return {key: scale(value) for key, value in public.groups}

    def state_dict(self):
        return {"calls": self.calls, "episodes": self.episodes}

    def load_state_dict(self, state):
        self.calls, self.episodes = state["calls"], state["episodes"]


class CountedReward:
    def __init__(self, parameters):
        self.scale = parameters["scale"]
        self.calls = self.episodes = 0

    def begin_episode(self):
        self.episodes += 1

    def transform(self, transition, reward):
        self.calls += 1
        return reward * self.scale + 0.0001 * self.calls

    def state_dict(self):
        return {"calls": self.calls, "episodes": self.episodes}

    def load_state_dict(self, state):
        self.calls, self.episodes = state["calls"], state["episodes"]


def custom_encoder(parameters, space):
    from smartsom.learning.torch_extensions import MLPEncoder

    return MLPEncoder(parameters, space)


def research_spec(resource):
    register_extension(
        "observation",
        "test.counted_public",
        "1",
        CountedPublicObservation,
        stateful=True,
    )
    register_extension(
        "reward", "test.counted_reward", "1", CountedReward, stateful=True
    )
    from smartsom.learning import torch_extensions

    register_extension(
        "torch_encoder",
        "test.feedforward",
        "1",
        custom_encoder,
        source_files=(Path(torch_extensions.__file__),),
    )
    actor = NetworkBranch(
        encoder=ExtensionRef(
            name="test.feedforward",
            version="1",
            parameters={"hidden_sizes": [16], "activation": "relu"},
        ),
        hidden_sizes=(16,),
    )
    critic = NetworkBranch(hidden_sizes=(32, 16))
    return ExtensionSpec(
        observation=ExtensionRef(name="test.counted_public", version="1"),
        network=NetworkSpec(
            actor=actor,
            critic=critic,
            roles={
                "agv_policy": ActorCriticSpec(
                    actor=NetworkBranch(hidden_sizes=(24,)),
                    critic=NetworkBranch(hidden_sizes=(20,)),
                )
            }
            if resource
            else {},
        ),
        reward=RewardSpec(
            team=ExtensionRef(
                name="test.counted_reward", version="1", parameters={"scale": 0.1}
            ),
            learner_scale=0.5,
            roles={
                "agv_policy": ExtensionRef(
                    name="test.counted_reward", version="1", parameters={"scale": 2.0}
                )
            }
            if resource
            else {},
        ),
    )


@pytest.fixture(
    scope="module",
    params=[(name, count) for name in ("sb3", "rllib", "marl") for count in (1, 2)],
)
def extended(request, tmp_path_factory):
    name, count = request.param
    packages = ["torch", "gymnasium", "sb3_contrib" if name == "sb3" else "ray"]
    if name == "marl":
        pass  # Grid resource learning uses RLlib directly.
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        required = (
            "SMARTSOM_REQUIRE_MARL" if name == "marl" else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required learning extras missing: {missing}")
        pytest.skip(f"optional learning extras missing: {missing}")
    resolved = small_training(
        name, tmp_path_factory.mktemp(f"extended-{name}-n{count}")
    )
    config = ExperimentConfig.model_validate_json(resolved.config_json)
    config.algorithm.extensions = research_spec(name == "marl")
    resolved = prepare(config)
    controls = TrainingControls(
        checkpoint_every_updates=1,
        keep_last=4,
        num_envs=count,
        sampling_processes=2 if count == 2 else 0,
        validation=ValidationControls(
            every_updates=2, replications=2, full_replay=True
        ),
    )
    continuous = train_one(resolved, controls=controls)
    partial = train_one(resolved, controls=replace(controls, stop_after_updates=2))
    resumed = train_one(
        resolved, controls=replace(controls, resume_from=partial.last_checkpoint)
    )
    return name, resolved, controls, continuous, partial, resumed


def test_extension_resume_matches_optimizer_rng_and_every_stream_state(extended):
    name, _, controls, continuous, partial, resumed = extended
    assert partial.environment_steps == 64 and resumed.environment_steps == 128
    assert partial.run_dir == resumed.run_dir
    assert checkpoint_rows(continuous) == checkpoint_rows(resumed)
    assert_state_equal(
        optimizer_and_rng(continuous, name), optimizer_and_rng(resumed, name)
    )
    manifests = [
        json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
        for result in (continuous, resumed)
    ]
    assert manifests[0]["final_weights_sha256"] == manifests[1]["final_weights_sha256"]
    assert manifests[1]["algorithm"]["extensions"]["observation"]["code_sha256"]
    rows = checkpoint_rows(resumed)
    selected = json.loads((resumed.checkpoint_dir / "extension_state.json").read_text())
    stream_zero = max(
        (row for row in rows if row["stream_id"] == 0),
        key=lambda row: row["local_episode"],
    )
    assert selected == stream_zero["extension_state"]["current"]
    for stream in range(controls.num_envs):
        current = max(
            (row for row in rows if row["stream_id"] == stream),
            key=lambda row: row["local_episode"],
        )
        state = current["extension_state"]["current"]
        assert state["learner_scale"] == 0.5
        assert all(
            component["state"]["episodes"] > 0 for component in state["components"]
        )
        assert any(component["state"]["calls"] > 0 for component in state["components"])
    audited = audit_training(resumed.run_dir)
    assert (
        audited["num_envs"] == controls.num_envs and audited["environment_steps"] == 128
    )
    if name == "marl":
        assert sum(len(step["indices"]) for row in rows for step in row["steps"]) == 128


def test_research_rewards_and_validation_are_separate_from_raw_physics(extended):
    from smartsom.config.training import episode_root
    from smartsom.learning.production import LearnedProductionDriver
    from smartsom.trace.production import ExecutionAudit

    _, resolved, _, _, _, resumed = extended
    rows = checkpoint_rows(resumed)
    changed_reward = False
    for row in rows:
        assert sum(
            step["reward_values"]["raw"] for step in row["steps"]
        ) == pytest.approx(row["return"])
        for step in row["steps"]:
            values = step["reward_values"]
            assert values["learner"] == pytest.approx(values["research"] * 0.5)
            changed_reward |= values["raw"] != values["research"]
            assert step["learner_rewards"] is not None
    assert changed_reward
    record = json.loads((resumed.run_dir / "run.json").read_text())
    attempt = resumed.run_dir / record["paths"]["training"]
    report = json.loads((attempt / "validation-000004.json").read_text())
    assert all(result["replay"] == "passed" for result in report["results"])
    case = resolved.resolved.episode(episode_root(101, 0))
    driver = LearnedProductionDriver(
        resumed.checkpoint_dir, case, deterministic=False, seed=811
    )
    try:
        checker = ExecutionAudit(case, driver.sim.snapshot(), driver.learning_contract)
        row = driver.next_tick()
        assert row is not None
        checker.append(row)
        for decision in row["decisions"]:
            assert decision["mask"][decision["selected_index"]]
    finally:
        driver.env.close()


def test_periodic_validation_cannot_change_training_extension_state_or_rng(extended):
    name, resolved, controls, continuous, _, _ = extended
    without = train_one(resolved, controls=replace(controls, validation=None))
    a, b = [
        json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
        for result in (continuous, without)
    ]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert checkpoint_rows(continuous) == checkpoint_rows(without)
    assert_state_equal(
        optimizer_and_rng(continuous, name), optimizer_and_rng(without, name)
    )


def test_extension_audit_rejects_a_changed_research_reward(extended, tmp_path):
    import shutil

    from smartsom.learning.checkpoint import file_hash
    from smartsom.trace.production import seal_checkpoint

    _, _, _, continuous, _, _ = extended
    root = tmp_path / "changed"
    shutil.copytree(continuous.run_dir, root)
    checkpoint = root / continuous.last_checkpoint.relative_to(continuous.run_dir)
    completed = checkpoint / "episodes.jsonl"
    rows = [json.loads(line) for line in completed.read_text().splitlines()]
    if rows:
        rows[0]["steps"][0]["reward_values"]["learner"] += 1.0
        completed.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        active = checkpoint / "active_episodes.json"
        rows = json.loads(active.read_text())
        rows[0]["steps"][0]["reward_values"]["learner"] += 1.0
        active.write_text(json.dumps(rows))
    # Re-sign the container to test semantic replay rather than only its checksum.
    seal_checkpoint(checkpoint)
    update = json.loads((checkpoint / "update.json").read_text())
    update["files"] = {
        str(p.relative_to(checkpoint)): file_hash(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file() and p.name != "update.json"
    }
    (checkpoint / "update.json").write_text(json.dumps(update))
    with pytest.raises(ValueError, match="decision/state/reward replay mismatch"):
        audit_training(checkpoint)
