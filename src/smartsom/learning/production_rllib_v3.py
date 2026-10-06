"""RLlib Learners adapted to semantic packets and physical-time targets."""

import copy

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from smartsom.learning.production_models import CandidateNetwork as CandidateNetwork
from smartsom.learning.production_models import pad_inputs


class SemanticModule(TorchRLModule):
    def setup(self):
        from smartsom.learning.policy_factory import network_class

        self.network = network_class(self.model_config.get("network_implementation"))(
            self.model_config["context_size"],
            self.model_config["network"],
            self.model_config["provider"],
            self.model_config["algorithm"],
            central=self.model_config.get("central", False),
            central_private_end=self.model_config.get("central_private_end", 16),
            candidate_width=self.model_config.get("candidate_width", 16),
        )
        if self.model_config["algorithm"] == "dqn":
            self.target = copy.deepcopy(self.network)
            for parameter in self.target.parameters():
                parameter.requires_grad_(False)

    def _forward_train(self, batch, **kwargs):
        if self.model_config["algorithm"] == "dqn":
            scores, _ = self.network(batch["obs"])
            return {"q": scores}
        obs = batch["obs"]
        batch_size, steps = obs["context"].shape[:2]
        flat = {k: v.reshape(batch_size * steps, *v.shape[2:]) for k, v in obs.items()}
        logits, _ = self.network(flat)
        _, values = self.network(batch["value_obs"])
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(batch["actions"].long().reshape(-1))
        active = batch["step_mask"].float().reshape(-1)
        return {
            "joint_log_probability": (log_probs * active)
            .reshape(batch_size, steps)
            .sum(-1),
            "values": values,
            "entropy": (distribution.entropy() * active)
            .reshape(batch_size, steps)
            .sum(-1),
        }

    def _forward_inference(self, batch, **kwargs):
        scores, values = self.network(batch["obs"])
        return {"scores": scores, "values": values}

    _forward_exploration = _forward_inference


class PhysicalPPOLearner(TorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        parameters = self.get_parameters(self.module[module_id])
        self.register_optimizer(
            module_id=module_id,
            optimizer=torch.optim.Adam(parameters),
            params=parameters,
            lr_or_lr_schedule=config.lr,
        )

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        parameters = self.module[module_id].model_config["parameters"]
        active = batch["actor_mask"].float()
        advantage = batch["advantages"].float()
        ratio = torch.exp(
            fwd_out["joint_log_probability"] - batch["old_log_probability"]
        )
        surrogate = torch.minimum(
            ratio * advantage,
            ratio.clamp(1 - parameters["clip_range"], 1 + parameters["clip_range"])
            * advantage,
        )
        actor = -(surrogate * active).sum() / active.sum().clamp(min=1)
        entropy = (fwd_out["entropy"] * active).sum() / active.sum().clamp(min=1)
        value_mask = batch["value_mask"].float()
        critic = (
            (fwd_out["values"] - batch["returns"]).square() * value_mask
        ).sum() / value_mask.sum().clamp(min=1)
        loss = (
            actor
            + parameters["value_coefficient"] * critic
            - parameters["entropy_coefficient"] * entropy
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite physical PPO loss")
        self.optimization_steps = getattr(self, "optimization_steps", 0) + 1
        self.metrics.log_dict(
            {
                "actor_loss": actor.detach(),
                "value_loss": critic.detach(),
                "entropy": entropy.detach(),
            },
            key=module_id,
            window=1,
        )
        return loss


class PhysicalDQNLearner(PhysicalPPOLearner):
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id]
        q = fwd_out["q"].gather(-1, batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            online_next, _ = module.network(batch["next_obs"])
            next_action = online_next.argmax(-1)
            target_next, _ = module.target(batch["next_obs"])
            next_q = target_next.gather(-1, next_action.unsqueeze(-1)).squeeze(-1)
            target = (
                batch["rewards"]
                + batch["discount"] * (1 - batch["terminated"].float()) * next_q
            )
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite physical Double-DQN loss")
        self.optimization_steps = getattr(self, "optimization_steps", 0) + 1
        self.metrics.log_dict({"td_loss": loss.detach()}, key=module_id, window=1)
        return loss


def build_learner(
    context_size,
    network,
    algorithm,
    parameters,
    *,
    central=False,
    central_private_end=16,
    device="cpu",
    candidate_width=16,
    network_implementation=None,
):
    config = PPOConfig() if algorithm == "ppo" else DQNConfig()
    config.framework("torch")
    config.lr = parameters["learning_rate"]
    config.grad_clip = parameters.get("max_grad_norm", 10.0)
    config.num_gpus_per_learner = int(device == "cuda")
    provider = (
        ("rllib.ppo" if central else "rllib.resource_ppo")
        if algorithm == "ppo"
        else "rllib.resource_dqn"
    )
    spec = RLModuleSpec(
        module_class=SemanticModule,
        observation_space=gym.spaces.Dict(
            {
                "context": gym.spaces.Box(-np.inf, np.inf, (context_size,), np.float32),
                "candidates": gym.spaces.Sequence(
                    gym.spaces.Box(-np.inf, np.inf, (candidate_width,), np.float32)
                ),
            }
        ),
        action_space=gym.spaces.Sequence(gym.spaces.Discrete(2**31 - 1)),
        model_config={
            "context_size": context_size,
            "network": network,
            "algorithm": algorithm,
            "parameters": parameters,
            "provider": provider,
            "central": central,
            "central_private_end": central_private_end,
            "candidate_width": candidate_width,
            **(
                {"network_implementation": network_implementation}
                if network_implementation is not None
                else {}
            ),
        },
    )
    cls = PhysicalPPOLearner if algorithm == "ppo" else PhysicalDQNLearner
    learner = cls(config=config, module_spec=spec)
    learner.build()
    return learner


def packet_batch(rows):
    from smartsom.learning.production_models import packet_arrays

    return MultiAgentBatch(
        {"default_policy": SampleBatch(packet_arrays(rows))}, len(rows)
    )


def replay_batch(rows, gamma):
    batch = {
        "obs": pad_inputs([r["input"] for r in rows]),
        "next_obs": pad_inputs([r["next_input"] for r in rows]),
        "actions": np.asarray([r["action"] for r in rows], np.int64),
        "rewards": np.asarray([r["reward"] for r in rows], np.float32),
        "discount": np.asarray([gamma ** r["dt"] for r in rows], np.float32),
        "terminated": np.asarray([r["terminated"] for r in rows], bool),
    }
    return MultiAgentBatch({"default_policy": SampleBatch(batch)}, len(rows))
