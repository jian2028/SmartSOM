"""Native masked Double DQN on complete per-resource decision transitions.

Gamma is one, as specified by the Base world objective. Rewards between decisions
include all intervening physical ticks, including rule-controlled decisions.
An unresolved tail at a safety cut is not a terminal training sample. It is
counted and omitted; no unobserved next action or terminal value is invented.
"""

import copy
import random
from pathlib import Path

import torch
from torch import nn

from smartsom.domain.base_decisions import GROUPS
from smartsom.learning.base_ppo import make_models, packer


class BaseDQN:
    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        torch.set_num_threads(numerical_threads)
        torch.manual_seed(seed)
        self.options, self.training, self.selection = options, training, selection
        self.models = make_models(options, value_head=False)
        self.pack = packer(options)
        self.targets = copy.deepcopy(self.models).requires_grad_(False)
        self.optimizers = {
            g: torch.optim.Adam(self.models[g].parameters(), lr=options.learning_rate)
            for g in GROUPS
        }
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.records, self.replay = [], {g: [] for g in GROUPS}
        self.updates = 0
        self.decisions = 0
        self.truncated_tails = 0
        self.last_losses = {}
        self.bound_policy_states = {}

    def needs_critic(self, request):
        return False

    def _pack(self, requests):
        return self.pack(
            requests,
            None,
            self.options.count_scale,
            allow_empty_wait=self.options.allow_empty_wait,
        )

    def choose(self, request, critic=None):
        with torch.no_grad():
            q = self.models[request.group].actor(self._pack([request]))[0]
        explore = self.selection == "sample" and (
            float(torch.rand((), generator=self.generator)) < self.options.epsilon
        )
        index = (
            int(torch.randint(len(request.candidates), (), generator=self.generator))
            if explore
            else int(q.argmax())
        )
        if self.training:
            self.records.append({"request": request, "index": index})
            self.decisions += 1
        return request.candidates[index].action

    def update(self, rewards, *, terminal, final_critic=None):
        suffix = [0.0] * (len(rewards) + 1)
        for i in reversed(range(len(rewards))):
            suffix[i] = suffix[i + 1] + rewards[i] * self.options.learning_reward_scale
        following = {}
        pending = {g: [] for g in GROUPS}
        for row in reversed(self.records):
            request = row["request"]
            key = (request.group, request.owner)
            next_request = following.get(key)
            if next_request is None and not terminal:
                self.truncated_tails += 1
            else:
                end = next_request.tick if next_request else len(rewards)
                pending[request.group].append(
                    {
                        "request": request,
                        "index": row["index"],
                        "reward": suffix[request.tick] - suffix[end],
                        "next": next_request,
                        "physical_dt": end - request.tick,
                    }
                )
            following[key] = request
        self.records = []
        losses = {}
        for group in GROUPS:
            replay = self.replay[group]
            replay.extend(reversed(pending[group]))
            del replay[: -self.options.replay_capacity]
            if not replay:
                continue
            for _ in range(self.options.epochs):
                rows = self.rng.sample(
                    replay, min(len(replay), self.options.batch_size)
                )
                inputs = self._pack([r["request"] for r in rows])
                q = self.models[group].actor(inputs)
                selected = torch.tensor([r["index"] for r in rows])
                predictions = q.gather(1, selected[:, None]).squeeze(1)
                target = torch.tensor([r["reward"] for r in rows], dtype=torch.float32)
                nonterminal = [i for i, r in enumerate(rows) if r["next"] is not None]
                if nonterminal:
                    inputs_next = self._pack([rows[i]["next"] for i in nonterminal])
                    with torch.no_grad():
                        action = self.models[group].actor(inputs_next).argmax(1)
                        value = (
                            self.targets[group]
                            .actor(inputs_next)
                            .gather(1, action[:, None])
                            .squeeze(1)
                        )
                    target[nonterminal] += value
                loss = nn.functional.smooth_l1_loss(predictions, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Base DQN loss")
                optimizer = self.optimizers[group]
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.models[group].parameters(), 1.0)
                optimizer.step()
                self.updates += 1
                if self.updates % self.options.target_update_interval == 0:
                    self.targets.load_state_dict(self.models.state_dict())
                losses[group] = {
                    "q": float(loss.detach()),
                    "replay_size": len(replay),
                    "transitions": len(pending[group]),
                }
        self.last_losses = losses
        return losses

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "contract": "smartsom.current-dqn/v1",
                "options": self.options.model_dump(mode="json", by_alias=True),
                "models": self.models.state_dict(),
                "targets": self.targets.state_dict(),
                "optimizers": {g: o.state_dict() for g, o in self.optimizers.items()},
                "replay": self.replay,
                "updates": self.updates,
                "decisions": self.decisions,
                "truncated_tails": self.truncated_tails,
                "rng": self.rng.getstate(),
                "generator": self.generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "bound_policy_states": self.bound_policy_states,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path):
        from smartsom.config.base_diagnostic import BaseAlgorithm

        payload = torch.load(path, map_location="cpu", weights_only=False)

        def identity(options):
            value = options.model_dump(mode="json", by_alias=True)
            for spec in value["providers"].values():
                spec.pop("path", None)
            for name in ("observation", "network"):
                if value.get(name) is not None:
                    value[name].pop("path", None)
            return value

        if payload["contract"] != "smartsom.current-dqn/v1" or identity(
            BaseAlgorithm.model_validate(payload["options"])
        ) != identity(self.options):
            raise ValueError("DQN checkpoint contract mismatch; use archived source")
        self.models.load_state_dict(payload["models"])
        self.targets.load_state_dict(payload["targets"])
        for group, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][group])
        self.replay = payload["replay"]
        self.updates = payload["updates"]
        self.decisions = payload["decisions"]
        self.truncated_tails = payload["truncated_tails"]
        self.rng.setstate(payload["rng"])
        self.generator.set_state(payload["generator"])
        torch.set_rng_state(payload["torch_rng"])
        self.bound_policy_states = payload["bound_policy_states"]
