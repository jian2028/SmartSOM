"""Three actual PPOs retain custom observation/reward state across complete resume."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_sampling import assert_state_equal, small_training

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
        packages.append("pettingzoo")
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
    spec = resolved.algorithm.algorithm.model_copy(
        update={"extensions": research_spec(name == "marl")}
    )
    resolved = replace(
        resolved, algorithm=resolved.algorithm.model_copy(update={"algorithm": spec})
    )
    controls = TrainingControls(
        checkpoint_every_updates=1,
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
    from smartsom.learning.training_state import load_state

    name, _, controls, continuous, partial, resumed = extended
    assert partial.environment_steps == 64 and resumed.environment_steps == 128
    assert (continuous.run_dir / "episodes.jsonl").read_bytes() == (
        resumed.run_dir / "episodes.jsonl"
    ).read_bytes()
    assert (continuous.last_checkpoint / "state_summary.json").read_bytes() == (
        resumed.last_checkpoint / "state_summary.json"
    ).read_bytes()
    assert_state_equal(
        *(
            load_state(result.last_checkpoint / "rng.pkl")
            for result in (continuous, resumed)
        )
    )
    manifests = [
        json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
        for result in (continuous, resumed)
    ]
    assert manifests[0]["final_weights_sha256"] == manifests[1]["final_weights_sha256"]
    assert manifests[1]["extensions"]["observation"]["code_sha256"]
    selected = json.loads(
        (resumed.checkpoint_dir / "extension_selection.json").read_text()
    )
    assert selected["scope"] == "model-only" and selected["stream_id"] == 0
    activity = load_state(resumed.last_checkpoint / "training/activity.pkl")
    states = (
        [stream["extension_state"] for stream in activity["streams"]]
        if controls.num_envs > 1
        else [activity["episode"]["extension_state"]]
    )
    assert len(states) == controls.num_envs
    scale = 0.5 * (0.0001 if name == "marl" else 1.0)
    assert all(
        state["learner_return"] == pytest.approx(state["research_return"] * scale)
        for state in states
    )
    audit = audit_training(resumed.run_dir)
    assert audit["num_envs"] == controls.num_envs and audit["environment_steps"] == 128
    if name == "marl":
        assert audit["agent_steps"] == 1536


def test_research_rewards_and_validation_are_separate_from_raw_physics(extended):
    from smartsom.experiments.training_validation import predictor
    from smartsom.learning.training_extensions import (
        environment_arguments,
        inference_view,
    )

    name, resolved, _, _, _, resumed = extended
    rows = [
        json.loads(line)
        for line in (resumed.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert all(len(row["extension_steps"]) == len(row["steps"]) for row in rows)
    for row in rows:
        assert sum(step["reward"] for step in row["steps"]) == row["return"]
        for raw, research in zip(row["steps"], row["extension_steps"], strict=True):
            assert "research_reward" not in raw and "reward_values" not in raw
            if name == "marl":
                values = research["reward_values"]["team"]
                assert values["raw"] == raw["reward"]
                assert (
                    values["learner"]
                    == values["research"]
                    * 0.5
                    * resolved.algorithm.algorithm.parameters.learner_reward_scale
                )
            else:
                assert research["learner_reward"] == research["research_reward"] * 0.5
    report = json.loads((resumed.run_dir / "validation-000004.json").read_text())
    assert all(result["replay"] == "passed" for result in report["results"])
    if name == "marl":
        from smartsom.learning.pettingzoo import SmartSOMParallelEnv

        kind = SmartSOMParallelEnv
    else:
        from smartsom.learning.gymnasium import SchedulingEnv

        kind = SchedulingEnv
    env = kind(
        resolved.episode(0).input,
        resolved.algorithm.algorithm.projection,
        **environment_arguments(resolved),
    )
    try:
        env.reset()
        predict = predictor(
            resolved.algorithm.algorithm.provider,
            resumed.checkpoint_dir,
            {},
            deterministic=False,
            seed=811,
        )
        actions = predict(inference_view(env))
        if name == "marl":
            assert all(
                env.projected.for_agent(agent).action_mask[index]
                for agent, index in actions.items()
            )
        else:
            assert env.projected.action_mask[actions]
    finally:
        env.close()


def test_periodic_validation_cannot_change_training_extension_state_or_rng(extended):
    _, resolved, controls, continuous, _, _ = extended
    without = train_one(resolved, controls=replace(controls, validation=None))
    a, b = [
        json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
        for result in (continuous, without)
    ]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert (continuous.run_dir / "episodes.jsonl").read_bytes() == (
        without.run_dir / "episodes.jsonl"
    ).read_bytes()


def test_extension_audit_rejects_a_changed_research_reward(extended):
    from smartsom.experiments.training_audit import load_training_snapshot
    from smartsom.learning.extension_audit import ExtensionTrainingAudit

    name, _, _, continuous, _, _ = extended
    resolved = load_training_snapshot(continuous.run_dir / "resolved_training.json")
    rows = [
        json.loads(line)
        for line in (continuous.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    audit = ExtensionTrainingAudit(resolved)
    for row in rows:
        if row["extension_steps"]:
            if name == "marl":
                row["extension_steps"][0]["reward_values"]["team"]["learner"] += 1.0
            else:
                row["extension_steps"][0]["learner_reward"] += 1.0
            with pytest.raises(ValueError, match="research observation/reward replay"):
                audit.episode(resolved.episode(row["episode"]).input, row)
            return
        audit.episode(resolved.episode(row["episode"]).input, row)
    pytest.fail("extended training produced no research transitions to audit")
