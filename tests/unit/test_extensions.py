"""Public extension contracts, independent numeric branches and state roundtrips."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest

from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionRef,
    ExtensionSpec,
    NetworkBranch,
    NetworkSpec,
    RewardSpec,
)
from smartsom.dispatch import DecisionContext
from smartsom.learning.extensions import (
    ExtensionsRuntime,
    ObservationSpace,
    PublicObservation,
    RewardTransition,
    bind_extensions,
    register_extension,
)


def require_optional(name):
    if os.environ.get("SMARTSOM_REQUIRE_EXTENSIONS") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


def public(role=None):
    return PublicObservation(
        DecisionContext(0, (), (), ()),
        (1.0, 2.0, 3.0),
        (("global", (1.0,)), ("local", (2.0, 3.0))),
        role,
    )


class CountingEncoder:
    def __init__(self, parameters, layout):
        self.count = 0
        self.output_space = layout.vector

    def encode(self, observation):
        self.count += 1
        return tuple(value + self.count for value in observation.vector)

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]


class CountingReward:
    def __init__(self, parameters):
        self.count = 0

    def transform(self, transition, reward):
        self.count += 1
        return reward + self.count

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]


class MissingStateEncoder:
    def __init__(self, parameters, layout):
        self.output_space = layout.vector


def test_config_and_registry_reject_import_paths_missing_names_and_changed_code():
    with pytest.raises(ValueError):
        ExtensionRef(name="/tmp/arbitrary.py:factory", version="1")
    with pytest.raises(ValueError, match="unregistered"):
        bind_extensions(
            ExtensionSpec(observation=ExtensionRef(name="missing", version="1")),
            "rllib.ppo",
        )
    spec = ExtensionSpec(observation=ExtensionRef(name="builtin.dict", version="1"))
    bound = bind_extensions(spec, "rllib.ppo")
    assert len(bound.observation.code_sha256) == 64
    assert spec.observation.code_sha256 is None
    wrong = bound.model_copy(
        update={
            "observation": bound.observation.model_copy(
                update={"code_sha256": "0" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="digest changed"):
        bind_extensions(wrong, "rllib.ppo")


def test_default_identity_and_public_input_are_explicit():
    assert bind_extensions(None, "rllib.ppo") is None
    value = public()
    runtime = ExtensionsRuntime(None, "rllib.ppo", {None: value.layout()})
    assert runtime.encode(value) == value.vector
    assert {field.name for field in fields(value)} == {
        "context",
        "vector",
        "groups",
        "role",
        "agent_id",
    }
    assert runtime.spaces[None].flat_size == len(value.vector)


def test_fixed_dict_shape_finite_values_and_terminal_zeros():
    value = public()
    runtime = ExtensionsRuntime(
        ExtensionSpec(observation=ExtensionRef(name="builtin.dict", version="1")),
        "rllib.ppo",
        {None: value.layout()},
    )
    assert runtime.encode(value) == {"global": (1.0,), "local": (2.0, 3.0)}
    assert runtime.spaces[None].zeros() == {"global": (0.0,), "local": (0.0, 0.0)}
    with pytest.raises(ValueError, match="shape"):
        runtime.encode(
            replace(value, groups=(("global", (1.0, 2.0)), ("local", (2.0, 3.0))))
        )
    with pytest.raises(ValueError, match="finite"):
        runtime.encode(
            replace(value, groups=(("global", (float("nan"),)), ("local", (2.0, 3.0))))
        )


def test_stateful_encoder_roundtrips_through_json_and_rejects_incompatible_state():
    register_extension(
        "observation", "test.counting_encoder", "1", CountingEncoder, stateful=True
    )
    value = public()
    spec = ExtensionSpec(
        observation=ExtensionRef(name="test.counting_encoder", version="1")
    )
    runtime = ExtensionsRuntime(spec, "rllib.ppo", {None: value.layout()})
    assert runtime.encode(value) == (2.0, 3.0, 4.0)
    saved = json.loads(json.dumps(runtime.state_dict()))
    restored = ExtensionsRuntime(spec, "rllib.ppo", {None: value.layout()})
    restored.load_state_dict(saved)
    assert restored.encode(value) == runtime.encode(value) == (3.0, 4.0, 5.0)
    saved["learner_scale"] = 2
    with pytest.raises(ValueError, match="contract mismatch"):
        restored.load_state_dict(saved)


def test_declared_stateful_extension_cannot_omit_state_protocol():
    register_extension(
        "observation", "test.missing_state", "1", MissingStateEncoder, stateful=True
    )
    with pytest.raises(ValueError, match="save/restore"):
        ExtensionsRuntime(
            ExtensionSpec(
                observation=ExtensionRef(name="test.missing_state", version="1")
            ),
            "rllib.ppo",
            {None: public().layout()},
        )


@pytest.mark.parametrize(
    "reason", [None, "completed", "deadlock", "policy_stalled", "budget_exhausted"]
)
def test_team_role_and_learner_rewards_cover_normal_and_terminal(reason):
    register_extension(
        "reward", "test.counting_reward", "1", CountingReward, stateful=True
    )
    scale = ExtensionRef(
        name="builtin.reward_scale",
        version="1",
        parameters={"scale": 2.0, "terminal_offset": 5.0},
    )
    spec = ExtensionSpec(
        reward=RewardSpec(
            team=ExtensionRef(name="test.counting_reward", version="1"),
            roles={"machine_policy": scale},
            learner_scale=0.1,
        )
    )
    runtime = ExtensionsRuntime(
        spec,
        "rllib.resource_ppo",
        {
            "machine_policy": public("machine_policy").layout(),
            "agv_policy": public("agv_policy").layout(),
        },
        learner_scale=0.5,
    )
    transition = RewardTransition(public().context, None, (), -10.0, 10, reason)
    batch = runtime.rewards(transition, ["machine_policy"] * 8 + ["agv_policy"] * 4)
    assert (batch.team.raw, batch.team.research, batch.team.learner) == (
        -10.0,
        -9.0,
        -0.45,
    )
    roles = dict(batch.roles)
    assert roles["agv_policy"] == batch.team
    assert roles["machine_policy"].raw == -10.0
    assert roles["machine_policy"].research == -18.0 + (5.0 if reason else 0.0)
    assert runtime.components[("reward", None)].instance.count == 1


def test_roles_and_supported_backends_are_enforced():
    register_extension(
        "observation",
        "test.only_rllib",
        "1",
        CountingEncoder,
        supported_backends=("rllib.ppo",),
        stateful=True,
    )
    with pytest.raises(ValueError, match="does not support"):
        bind_extensions(
            ExtensionSpec(
                observation=ExtensionRef(name="test.only_rllib", version="1")
            ),
            "sb3.maskable_ppo",
        )
    with pytest.raises(ValueError, match="role extensions"):
        bind_extensions(
            ExtensionSpec(
                role_observations={
                    "machine_policy": ExtensionRef(name="builtin.dict", version="1")
                }
            ),
            "rllib.ppo",
        )


@pytest.mark.parametrize(
    "provider", ["rllib.ppo", "sb3.maskable_ppo", "rllib.resource_ppo"]
)
def test_custom_feedforward_actor_critic_and_role_networks_roundtrip(provider):
    torch = require_optional("torch")
    from smartsom.learning.torch_extensions import build_actor_critic

    spec = NetworkSpec(
        actor=NetworkBranch(
            encoder=ExtensionRef(
                name="builtin.mlp",
                version="1",
                parameters={"hidden_sizes": [7], "activation": "relu"},
            ),
            hidden_sizes=(5,),
        ),
        critic=NetworkBranch(hidden_sizes=(11, 3)),
        roles={
            "agv_policy": ActorCriticSpec(
                actor=NetworkBranch(hidden_sizes=(13,)),
                critic=NetworkBranch(hidden_sizes=(17,)),
            )
        },
    )
    if provider != "rllib.resource_ppo":
        spec = spec.model_copy(update={"roles": {}})
    space = ObservationSpace(fields=(("global", (1,)), ("local", (2,))))
    observation = {"global": torch.ones((4, 1)), "local": torch.zeros((4, 2))}
    network = build_actor_critic(
        space,
        6,
        spec,
        provider,
        role="machine_policy" if provider == "rllib.resource_ppo" else None,
    )
    result = network(observation)
    assert result["logits"].shape == (4, 6) and result["values"].shape == (4,)
    assert not {id(p) for p in network.actor.parameters()} & {
        id(p) for p in network.critic.parameters()
    }
    loss = result["logits"].sum() + result["values"].sum()
    loss.backward()
    assert all(p.grad is not None for p in network.parameters())
    restored = build_actor_critic(
        space,
        6,
        spec,
        provider,
        role="machine_policy" if provider == "rllib.resource_ppo" else None,
    )
    restored.load_state_dict(network.state_dict())
    assert torch.equal(restored(observation)["logits"], result["logits"])
    mask = torch.tensor([[1, 0, 1, 0, 0, 0]] * 4)
    assert (
        network.masked_logits(observation, mask).argmax(-1).tolist()
        == [network.masked_logits(observation, mask).argmax(-1)[0].item()] * 4
    )
    if provider == "rllib.resource_ppo":
        agv = build_actor_critic(space, 6, spec, provider, role="agv_policy")
        assert sum(p.numel() for p in agv.parameters()) != sum(
            p.numel() for p in network.parameters()
        )


def test_extension_import_does_not_require_optional_frameworks():
    code = "import sys; import smartsom.learning.extensions; assert not set(('torch','ray','gymnasium','pettingzoo','numpy')) & set(sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("provider", ["sb3.maskable_ppo", "rllib.ppo"])
def test_extended_gym_preserves_physics_and_all_reward_streams(provider):
    require_optional("gymnasium")
    from test_learning_gymnasium import hand_index
    from test_production_runtime import small_scenario

    from smartsom.learning.production_env import ProductionEnv

    spec = ExtensionSpec(
        observation=ExtensionRef(name="builtin.dict", version="1"),
        reward=RewardSpec(
            team=ExtensionRef(
                name="builtin.reward_scale",
                version="1",
                parameters={"scale": 2.0, "terminal_offset": 7.0},
            ),
            learner_scale=0.1,
        ),
    )
    env = ProductionEnv(small_scenario(), 4, extensions=spec, provider=provider)
    raw = ProductionEnv(small_scenario(), 4)
    observation, _ = env.reset()
    raw.reset()
    assert env.observation_space.contains(observation)
    totals = dict(raw=0.0, research=0.0, learner=0.0)
    while not env.finished:
        index = hand_index(env)
        observation, reward, _, _, info = env.step(index)
        raw.step(index)
        assert env.observation_space.contains(observation)
        assert env.sim.snapshot() == raw.sim.snapshot()
        assert reward == info["reward_values"]["learner"]
        for key in totals:
            totals[key] += info["reward_values"][key]
    assert env.sim.completed == {"demand"}
    assert totals["raw"] == raw.episode_reward
    assert totals["research"] == pytest.approx(totals["raw"] * 2 + 7)
    assert totals["learner"] == pytest.approx(totals["research"] * 0.1)


def test_stateful_gym_cached_observation_and_active_episode_restore():
    require_optional("gymnasium")
    import numpy as np
    from test_production_runtime import small_scenario

    from smartsom.learning.production_env import ProductionEnv

    register_extension(
        "observation", "test.counting_encoder", "1", CountingEncoder, stateful=True
    )
    spec = ExtensionSpec(
        observation=ExtensionRef(name="test.counting_encoder", version="1")
    )
    env = ProductionEnv(small_scenario(), 4, extensions=spec)
    initial = env.hooks.runtime.state_dict()
    env.reset()
    encoder = env.hooks.runtime.components[("observation", None)].instance
    assert encoder.count == 1
    env.observation()
    env.observation()
    assert encoder.count == 1
    observed = env.observation()
    expected = observed.copy()
    observed[:] = -999
    np.testing.assert_array_equal(env.observation(), expected)
    assert encoder.count == 1
    env.step(4)
    saved = json.loads(json.dumps(env.hooks.runtime.state_dict()))
    restored = ProductionEnv(small_scenario(), 4, extensions=spec)
    restored.hooks.runtime.load_state_dict(initial)
    restored.reset()
    restored.step(4)
    assert restored.hooks.runtime.state_dict() == saved
    np.testing.assert_array_equal(restored.observation(), env.observation())
    assert (
        restored.hooks.runtime.components[("observation", None)].instance.count
        == encoder.count
    )


def test_resource_dict_and_role_rewards_keep_joint_physics():
    require_optional("gymnasium")
    from test_learning_gymnasium import hand_index
    from test_production_runtime import small_scenario

    from smartsom.learning.production_env import ProductionEnv

    spec = ExtensionSpec(
        observation=ExtensionRef(name="builtin.dict", version="1"),
        reward=RewardSpec(
            team=ExtensionRef(
                name="builtin.reward_scale", version="1", parameters={"scale": 2.0}
            ),
            roles={
                "agv_policy": ExtensionRef(
                    name="builtin.reward_scale",
                    version="1",
                    parameters={"scale": 3.0, "terminal_offset": 5.0},
                )
            },
            learner_scale=0.1,
        ),
    )
    env = ProductionEnv(
        small_scenario(), 4, extensions=spec, provider="rllib.resource_ppo"
    )
    raw = ProductionEnv(small_scenario(), 4)
    env.reset()
    raw.reset()
    totals = {role: 0.0 for role in env.hooks.roles}
    while not env.finished:
        index = hand_index(env)
        observation, _, _, _, info = env.step(index)
        raw.step(index)
        assert env.observation_space.contains(observation)
        assert env.sim.snapshot() == raw.sim.snapshot()
        values = info["reward_values"]["roles"]
        for role, value in values.items():
            totals[role] += value["research"]
            assert info["role_rewards"][role] == pytest.approx(value["research"] * 0.1)
    assert totals["machine_policy"] == pytest.approx(2 * raw.episode_reward)
    assert totals["agv_policy"] == pytest.approx(6 * raw.episode_reward + 5)
