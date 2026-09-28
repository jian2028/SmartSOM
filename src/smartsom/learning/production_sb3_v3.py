"""MaskablePPO with a complete conditional physical-boundary rollout."""

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from smartsom.learning.production_models import CandidateNetwork, packet_arrays


class ContractSpaces(gym.Env):
    """Space declaration only. The coordinator owns the sole physical simulator."""

    def __init__(self, context_size):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (context_size,), np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("use the semantic physical-boundary collector")

    def step(self, action):
        raise RuntimeError("use the semantic physical-boundary collector")


class ConditionalPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        *args,
        context_size,
        network_spec,
        central_private_end,
        candidate_width,
        **kwargs,
    ):
        self.context_size, self.network_spec = context_size, network_spec
        self.central_private_end = central_private_end
        self.candidate_width = candidate_width
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.network = CandidateNetwork(
            self.context_size,
            self.network_spec,
            "sb3.maskable_ppo",
            central=True,
            central_private_end=self.central_private_end,
            candidate_width=self.candidate_width,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr_schedule(1))

    def evaluate_packet(self, batch):
        obs = batch["obs"]
        b, s = obs["context"].shape[:2]
        flat = {k: v.reshape(b * s, *v.shape[2:]) for k, v in obs.items()}
        logits, _ = self.network(flat)
        _, values = self.network(batch["value_obs"])
        dist = torch.distributions.Categorical(logits=logits)
        active = batch["step_mask"].float()
        probability = dist.log_prob(batch["actions"].long().reshape(-1)).reshape(b, s)
        entropy = dist.entropy().reshape(b, s)
        return (
            (probability * active).sum(-1),
            values,
            (entropy * active).sum(-1),
        )


class ConditionalMaskablePPO(MaskablePPO):
    """SB3 optimizer lifecycle; one joint ratio per complete physical packet."""

    def train_packets(self, rows, parameters, rng):
        self.policy.set_training_mode(True)
        for _ in range(parameters["n_epochs"]):
            order = list(range(len(rows)))
            rng.shuffle(order)
            for start in range(0, len(order), parameters["batch_size"]):
                selected = [
                    rows[i] for i in order[start : start + parameters["batch_size"]]
                ]
                raw = packet_arrays(selected)
                batch = {
                    k: (
                        {
                            n: torch.as_tensor(v, device=self.device)
                            for n, v in val.items()
                        }
                        if isinstance(val, dict)
                        else torch.as_tensor(val, device=self.device)
                    )
                    for k, val in raw.items()
                }
                logp, values, entropy = self.policy.evaluate_packet(batch)
                active = batch["actor_mask"].float()
                ratio = (logp - batch["old_log_probability"]).exp()
                advantage = batch["advantages"]
                clipped = ratio.clamp(
                    1 - parameters["clip_range"], 1 + parameters["clip_range"]
                )
                actor = -(
                    torch.minimum(ratio * advantage, clipped * advantage) * active
                ).sum() / active.sum().clamp(min=1)
                vm = batch["value_mask"].float()
                critic = (
                    (values - batch["returns"]).square() * vm
                ).sum() / vm.sum().clamp(min=1)
                ent = (entropy * active).sum() / active.sum().clamp(min=1)
                loss = (
                    actor
                    + parameters["value_coefficient"] * critic
                    - parameters["entropy_coefficient"] * ent
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite central MaskablePPO loss")
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.network.parameters(), parameters["max_grad_norm"]
                )
                self.policy.optimizer.step()
                self._n_updates += 1
        return self._n_updates


def build_sb3(
    context_size,
    network,
    parameters,
    seed,
    central_private_end=16,
    device="cpu",
    candidate_width=16,
):
    return ConditionalMaskablePPO(
        ConditionalPolicy,
        ContractSpaces(context_size),
        learning_rate=parameters["learning_rate"],
        n_steps=2,
        batch_size=2,
        seed=seed,
        device=device,
        policy_kwargs={
            "context_size": context_size,
            "network_spec": network,
            "central_private_end": central_private_end,
            "candidate_width": candidate_width,
        },
    )
