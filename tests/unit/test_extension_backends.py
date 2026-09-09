"""Real optional policy/module adapters; drivers own full training acceptance."""

import importlib
import os

import pytest

from smartsom.config.codec import primitive
from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionRef,
    ExtensionSpec,
    NetworkBranch,
    NetworkSpec,
)
from smartsom.learning.extensions import bind_extensions


def require_optional(name):
    if os.environ.get("SMARTSOM_REQUIRE_EXTENSIONS") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


def dependencies(provider):
    gym = require_optional("gymnasium")
    torch = require_optional("torch")
    np = require_optional("numpy")
    require_optional("ray.rllib" if provider.startswith("rllib") else "sb3_contrib")
    return gym, torch, np


def configuration(provider, custom):
    network = None
    if custom:
        network = NetworkSpec(
            actor=NetworkBranch(
                encoder=ExtensionRef(
                    name="builtin.mlp", version="1", parameters={"hidden_sizes": [7]}
                ),
                hidden_sizes=(5,),
            ),
            critic=NetworkBranch(hidden_sizes=(11,)),
            roles={
                "agv_policy": ActorCriticSpec(
                    actor=NetworkBranch(hidden_sizes=(3,)),
                    critic=NetworkBranch(hidden_sizes=(13,)),
                )
            }
            if provider == "rllib.resource_ppo"
            else {},
        )
    return primitive(
        bind_extensions(
            ExtensionSpec(
                observation=ExtensionRef(name="builtin.dict", version="1"),
                network=network,
            ),
            provider,
        )
    )


@pytest.mark.parametrize("custom", [False, True])
def test_sb3_real_ppo_dict_update_and_serialization(tmp_path, custom):
    gym, torch, np = dependencies("sb3.maskable_ppo")
    from sb3_contrib import MaskablePPO

    from smartsom.learning.sb3_extensions import ExtensionMaskableActorCriticPolicy
    from smartsom.learning.weights import weights_digest

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-1, 1, (3,), np.float32),
                "local": gym.spaces.Box(-1, 1, (2,), np.float32),
            }
        )
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.count = 0
            return self.observe(), {}

        def observe(self):
            return {
                "state": np.full(3, self.count / 4, np.float32),
                "local": np.ones(2, np.float32),
            }

        def action_masks(self):
            return np.array([True, False, True])

        def step(self, action):
            assert action in (0, 2)
            self.count += 1
            return self.observe(), float(action == 2), self.count == 4, False, {}

    model = MaskablePPO(
        ExtensionMaskableActorCriticPolicy,
        TinyEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        seed=10,
        policy_kwargs={
            "extensions": configuration("sb3.maskable_ppo", custom),
            "provider": "sb3.maskable_ppo",
            "role": None,
            "fallback_hidden_sizes": [8],
        },
    )
    initial_actor = weights_digest(model.policy.network.actor.state_dict())
    initial_critic = weights_digest(model.policy.network.critic.state_dict())
    model.learn(16)
    assert initial_actor != weights_digest(model.policy.network.actor.state_dict())
    assert initial_critic != weights_digest(model.policy.network.critic.state_dict())
    observation, _ = TinyEnv().reset()
    action = model.predict(
        observation, deterministic=True, action_masks=[True, False, True]
    )[0]
    model.save(tmp_path / "model")
    loaded = MaskablePPO.load(tmp_path / "model")
    assert weights_digest(model.policy.state_dict()) == weights_digest(
        loaded.policy.state_dict()
    )
    assert (
        loaded.predict(
            observation, deterministic=True, action_masks=[True, False, True]
        )[0]
        == action
    )
    model.policy.save(tmp_path / "policy.pt")
    policy = ExtensionMaskableActorCriticPolicy.load(tmp_path / "policy.pt")
    assert weights_digest(policy.state_dict()) == weights_digest(
        model.policy.state_dict()
    )
    assert all(p.grad is not None for p in model.policy.network.parameters())


@pytest.mark.parametrize(
    "provider,role",
    [
        ("rllib.ppo", None),
        ("rllib.resource_ppo", "machine_policy"),
        ("rllib.resource_ppo", "agv_policy"),
    ],
)
@pytest.mark.parametrize("custom", [False, True])
def test_rllib_mask_value_gradient_and_actual_module_checkpoint(
    tmp_path, provider, role, custom
):
    gym, torch, np = dependencies(provider)
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModule

    from smartsom.learning.extension_tensors import observation_tensor
    from smartsom.learning.rllib_extensions import ExtensionPPOTorchRLModule
    from smartsom.learning.weights import weights_digest

    obs_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-1, 1, (3,), np.float32),
            "local": gym.spaces.Box(-1, 1, (2,), np.float32),
        }
    )
    module = ExtensionPPOTorchRLModule(
        observation_space=gym.spaces.Dict(
            {
                "observations": obs_space,
                "action_mask": gym.spaces.Box(0, 1, (3,), np.int8),
            }
        ),
        action_space=gym.spaces.Discrete(3),
        model_config={
            "extensions": configuration(provider, custom),
            "provider": provider,
            "role": role,
            "fallback_hidden_sizes": [8],
        },
    )
    observation = observation_tensor(
        {"state": [1, 0, 1], "local": [0, 1]}, add_batch=True
    )
    batch = {
        Columns.OBS: {
            "observations": observation,
            "action_mask": torch.tensor([[1, 0, 1]]),
        }
    }
    logits = module.forward_train(batch)[Columns.ACTION_DIST_INPUTS]
    values = module.compute_values(batch)
    assert logits.argmax(-1).item() in (0, 2) and values.shape == (1,)
    loss = logits[:, [0, 2]].sum() + values.sum()
    loss.backward()
    assert all(p.grad is not None for p in module.network.parameters())
    module.save_to_path(tmp_path / "module")
    restored = RLModule.from_checkpoint(tmp_path / "module")
    assert weights_digest(module.get_state()) == weights_digest(restored.get_state())
    assert torch.equal(
        module.forward_inference(batch)[Columns.ACTION_DIST_INPUTS],
        restored.forward_inference(batch)[Columns.ACTION_DIST_INPUTS],
    )
    batch[Columns.OBS]["action_mask"] = torch.zeros(1, 3)
    assert torch.isfinite(module.compute_values(batch)).all()
    assert torch.isfinite(module.forward_train(batch)[Columns.ACTION_DIST_INPUTS]).all()


def test_tensor_boundary_rejects_unpinned_or_unsupported_spaces():
    gym, torch, np = dependencies("sb3.maskable_ppo")
    from smartsom.learning.extension_tensors import (
        network_configuration,
        observation_tensor,
        public_space,
    )

    with pytest.raises(ValueError, match="pinned"):
        network_configuration(
            {"observation": {"name": "builtin.dict", "version": "1"}},
            "sb3.maskable_ppo",
            [8],
        )
    with pytest.raises(ValueError, match="fixed Box"):
        public_space(gym.spaces.Discrete(3))
    with pytest.raises(ValueError, match="nonfinite"):
        observation_tensor([float("nan")])
