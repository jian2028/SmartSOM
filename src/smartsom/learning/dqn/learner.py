"""Native masked Double DQN on complete per-resource decision transitions.

Gamma is one, as specified by the Base world objective. Rewards between decisions
include all intervening physical ticks, including rule-controlled decisions.
An unresolved tail at a safety cut is not a terminal training sample. It is
counted and omitted; no unobserved next action or terminal value is invented.
"""

import copy
import random

import torch
from torch import nn

from smartsom.domain.base_decisions import GROUPS
from smartsom.learning.models import make_models, packer
from smartsom.learning.ownership import NetworkOwnership


class BaseDQN:
    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        torch.set_num_threads(numerical_threads)
        torch.manual_seed(seed)
        self.options, self.training, self.selection = options, training, selection
        # Match the deterministic native forward contract (gradients still run
        # during update); target inference must never mutate running statistics.
        self.models = make_models(options, value_head=False).eval()
        self.ownership = NetworkOwnership(options)
        self.models = self.ownership.expand(self.models)
        self.pack = packer(options)
        self.targets = copy.deepcopy(self.models).requires_grad_(False).eval()
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
        self.records, self.replay = [], {key: [] for key in self.models}
        self.key_updates = {key: 0 for key in self.models}
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
            q = self.models[self.ownership.key(request)].actor(self._pack([request]))[0]
        explore = self.selection == "sample" and (
            float(torch.rand((), generator=self.generator))
            < self.ownership.options[request.group].epsilon
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
        from smartsom.learning.diagnostics import UpdateDiagnostics
        from smartsom.learning.dqn.replay import transitions

        pending, omitted = transitions(
            self.records,
            rewards,
            terminal=terminal,
            scale={
                g: s.learning_reward_scale for g, s in self.ownership.options.items()
            },
        )
        self.truncated_tails += omitted
        self.records = []
        losses = {}
        diagnostics = UpdateDiagnostics()
        for group in GROUPS:
            settings = self.ownership.options[group]
            for key in self.ownership.keys[group]:
                new = [
                    r for r in pending[group] if self.ownership.key(r["request"]) == key
                ]
                replay = self.replay[key]
                replay.extend(reversed(new))
                del replay[: -settings.replay_capacity]
                if not replay:
                    continue
                for _ in range(settings.epochs):
                    rows = self.rng.sample(
                        replay, min(len(replay), settings.batch_size)
                    )
                    inputs = self._pack([r["request"] for r in rows])
                    q = self.models[key].actor(inputs)
                    selected = torch.tensor([r["index"] for r in rows])
                    predictions = q.gather(1, selected[:, None]).squeeze(1)
                    target = torch.tensor(
                        [r["reward"] for r in rows], dtype=torch.float32
                    )
                    nonterminal = [
                        i for i, r in enumerate(rows) if r["next"] is not None
                    ]
                    if nonterminal:
                        inputs_next = self._pack([rows[i]["next"] for i in nonterminal])
                        with torch.no_grad():
                            action = self.models[key].actor(inputs_next).argmax(1)
                            value = (
                                self.targets[key]
                                .actor(inputs_next)
                                .gather(1, action[:, None])
                                .squeeze(1)
                            )
                        target[nonterminal] += value
                    loss = nn.functional.smooth_l1_loss(predictions, target)
                    with torch.no_grad():
                        diagnostics.add(
                            rows,
                            {
                                "q_loss": nn.functional.smooth_l1_loss(
                                    predictions, target, reduction="none"
                                ).tolist(),
                                "absolute_td_error": (target - predictions)
                                .abs()
                                .tolist(),
                                "q_value": predictions.tolist(),
                            },
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite Base DQN loss")
                    optimizer = self.optimizers[key]
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.models[key].parameters(), 1.0)
                    optimizer.step()
                    self.updates += 1
                    self.key_updates[key] += 1
                    # Shared roles retain the established global update clock.
                    # Independent owners use their own update count and interval.
                    for shared in GROUPS:
                        interval = self.ownership.options[shared].target_update_interval
                        if (
                            self.ownership.keys[shared] == (shared,)
                            and self.updates % interval == 0
                        ):
                            self.targets[shared].load_state_dict(
                                self.models[shared].state_dict()
                            )
                    if (
                        key != group
                        and self.key_updates[key] % settings.target_update_interval == 0
                    ):
                        self.targets[key].load_state_dict(self.models[key].state_dict())
                    losses[group] = {
                        "q": float(loss.detach()),
                        "replay_size": len(replay),
                        "transitions": len(new),
                    }
            if group in losses:
                losses[group]["update_diagnostics"] = diagnostics.summary(group)
        for group, values in losses.items():
            settings = self.ownership.options[group]
            values["epsilon"] = settings.epsilon
            values["owner_target_updates"] = {
                key: (self.updates if key == group else self.key_updates[key])
                // settings.target_update_interval
                for key in self.ownership.keys[group]
            }
            values["target_updates"] = sum(values["owner_target_updates"].values())
            values["replay_size"] = sum(
                len(self.replay[key]) for key in self.ownership.keys[group]
            )
        self.last_losses = losses
        return losses

    def get_state(self):
        return {
            "ownership": self.ownership.contract,
            "key_updates": self.key_updates,
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
        }

    def set_state(self, payload):
        self.ownership.validate(payload["ownership"])
        self.key_updates = payload["key_updates"]
        self.models.load_state_dict(payload["models"])
        self.targets.load_state_dict(payload["targets"])
        self.models.eval()
        self.targets.eval()
        for group, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][group])
        self.replay = payload["replay"]
        self.updates = payload["updates"]
        self.decisions = payload["decisions"]
        self.truncated_tails = payload["truncated_tails"]
        self.rng.setstate(payload["rng"])
        self.generator.set_state(payload["generator"])
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
