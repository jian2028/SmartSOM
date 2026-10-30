"""PPO updates with gamma=1 episode returns and independent critic bootstrap."""

import random

import torch
from torch import nn

from smartsom.domain.base_decisions import GROUPS
from smartsom.learning.models import make_models, packer
from smartsom.learning.ownership import NetworkOwnership
from smartsom.representations.base.batching import encode_requests


class BasePPO:
    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        torch.set_num_threads(numerical_threads)
        torch.manual_seed(seed)
        self.options, self.training, self.selection = options, training, selection
        # One deterministic forward contract for rollout, bootstrap and update:
        # eval disables Dropout and freezes BatchNorm statistics, not autograd.
        # Switching to train for updates would change PPO likelihood ratios even
        # before an optimizer step and make them depend on minibatch composition.
        self.models = make_models(options).eval()
        self.pack = packer(options)
        self.ownership = NetworkOwnership(options)
        with torch.no_grad():
            for group, model in self.models.items():
                scale = self.ownership.options[group].critic_initial_scale
                if scale != 1.0:
                    model.critic[-1].weight.mul_(scale)
                    model.critic[-1].bias.mul_(scale)
        self.models = self.ownership.expand(self.models)
        self.optimizers = {
            key: torch.optim.Adam(
                self.models[key].parameters(),
                lr=self.ownership.options[g].learning_rate,
            )
            for g in GROUPS
            for key in self.ownership.keys[g]
        }
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.records = []
        self.updates = 0
        self.last_losses = {}
        self.bound_policy_states = {}

    def choose(self, request, critic):
        deterministic = len(request.candidates) == 1
        if deterministic and not self.training:
            return request.candidates[0].action
        with torch.no_grad():
            model = self.models[self.ownership.key(request)]
            inputs = self.pack(
                [request],
                [critic] if self.training else None,
                self.options.count_scale,
                allow_empty_wait=self.options.allow_empty_wait,
            )
            if self.training:
                logits, value = model(inputs)
            else:
                logits = model.actor(inputs)
            probs = torch.softmax(logits[0], 0)
            index = (
                int(torch.multinomial(probs, 1, generator=self.generator))
                if self.selection == "sample"
                else int(logits[0].argmax())
            )
            log_prob = float(torch.log_softmax(logits[0], 0)[index])
        if self.training:
            self.records.append(
                {
                    "request": request,
                    "critic": critic,
                    "index": index,
                    "action": request.candidates[index].identity,
                    "log_prob": log_prob,
                    "value": float(value[0]),
                    "actor_mask": not deterministic,
                }
            )
        return request.candidates[index].action

    def update(self, rewards, *, terminal, final_critic):
        from smartsom.learning.diagnostics import UpdateDiagnostics
        from smartsom.learning.ppo.rollout import reward_suffix

        losses = {}
        diagnostics = UpdateDiagnostics()
        for group in GROUPS:
            rows = [r for r in self.records if r["request"].group == group]
            if not rows:
                continue
            settings = self.ownership.options[group]
            suffix = reward_suffix(rewards, settings.learning_reward_scale)
            with torch.no_grad():
                bootstraps = {
                    key: 0.0
                    if terminal
                    else float(
                        self.models[key].critic(torch.tensor([final_critic])).item()
                    )
                    for key in self.ownership.keys[group]
                }
            for row in rows:
                row["return"] = (
                    suffix[row["request"].tick]
                    + bootstraps[self.ownership.key(row["request"])]
                )
            advantages = torch.tensor([r["return"] - r["value"] for r in rows])
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp(min=1e-6)
            for row, adv in zip(rows, advantages, strict=True):
                row["advantage"] = float(adv)
            group_rows = rows
            for key in self.ownership.keys[group]:
                rows = [
                    r for r in group_rows if self.ownership.key(r["request"]) == key
                ]
                if not rows:
                    continue
                encoded = encode_requests([r["request"] for r in rows])
                for _ in range(settings.epochs):
                    self.rng.shuffle(rows)
                    for offset in range(0, len(rows), settings.batch_size):
                        batch = rows[offset : offset + settings.batch_size]
                        inputs = self.pack(
                            [r["request"] for r in batch],
                            [r["critic"] for r in batch],
                            self.options.count_scale,
                            encoded=encoded,
                            allow_empty_wait=self.options.allow_empty_wait,
                        )
                        logits, values = self.models[key](inputs)
                        distribution = torch.distributions.Categorical(logits=logits)
                        selected = torch.tensor([r["index"] for r in batch])
                        old = torch.tensor([r["log_prob"] for r in batch])
                        ratio = (distribution.log_prob(selected) - old).exp()
                        advantage = torch.tensor([r["advantage"] for r in batch])
                        actor_mask = torch.tensor([r["actor_mask"] for r in batch])
                        contributions = torch.minimum(
                            ratio * advantage,
                            ratio.clamp(1 - settings.clip, 1 + settings.clip)
                            * advantage,
                        )
                        actor = -(
                            contributions * actor_mask
                        ).sum() / actor_mask.sum().clamp(min=1)
                        target = torch.tensor([r["return"] for r in batch])
                        critic_loss = torch.nn.functional.smooth_l1_loss(values, target)
                        entropy = distribution.entropy()
                        loss = actor + critic_loss - settings.entropy * entropy.mean()
                        with torch.no_grad():
                            log_ratio = distribution.log_prob(selected) - old
                            diagnostics.add(
                                batch,
                                {
                                    "actor_loss": (-contributions).tolist(),
                                    "entropy": entropy.tolist(),
                                    "approx_kl": (ratio - 1 - log_ratio).tolist(),
                                    "clip_fraction": ((ratio - 1).abs() > settings.clip)
                                    .float()
                                    .tolist(),
                                },
                                mask=actor_mask.tolist(),
                            )
                            diagnostics.add(
                                batch,
                                {
                                    "critic_loss": torch.nn.functional.smooth_l1_loss(
                                        values, target, reduction="none"
                                    ).tolist()
                                },
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite Base PPO loss")
                        optimizer = self.optimizers[key]
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.models[key].parameters(), 1.0)
                        optimizer.step()
                        self.updates += 1
                        losses[group] = {
                            "actor": float(actor.detach()),
                            "critic": float(critic_loss.detach()),
                            "decisions": len(rows),
                        }
            losses[group]["decisions"] = len(group_rows)
            losses[group]["update_diagnostics"] = diagnostics.summary(group)
        self.records = []
        self.last_losses = losses
        return losses

    def get_state(self):
        return {
            "ownership": self.ownership.contract,
            "models": self.models.state_dict(),
            "optimizers": {g: o.state_dict() for g, o in self.optimizers.items()},
            "updates": self.updates,
            "rng": self.rng.getstate(),
            "generator": self.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

    def set_state(self, payload):
        self.ownership.validate(payload["ownership"])
        self.models.load_state_dict(payload["models"])
        self.models.eval()
        for g, o in self.optimizers.items():
            o.load_state_dict(payload["optimizers"][g])
        self.updates = payload["updates"]
        self.rng.setstate(payload["rng"])
        self.generator.set_state(payload["generator"])
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])

    def save(self, path):
        from smartsom.learning.state import save

        save(path, self.options, self.get_state())

    def load(self, path):
        from smartsom.learning.state import load

        self.set_state(load(path, self.options))

    def sampling_state(self):
        return {
            "ownership": self.ownership.contract,
            "models": self.models.state_dict(),
        }

    def load_sampling_state(self, state):
        self.ownership.validate(state["ownership"])
        self.models.load_state_dict(state["models"])

    def seed_sampling(self, seed):
        self.generator.manual_seed(seed)

    def discard_rollout(self):
        self.records = []

    def needs_critic(self, request):
        return self.training
