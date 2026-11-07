"""Episode-bound central PPO; physics and orchestration remain native."""

import random
import time
from dataclasses import replace

import torch

from smartsom.config.components import construct, resolve
from smartsom.domain.public_facts import freeze

from .rollout import objective


class CentralPPO:
    def __init__(
        self,
        options,
        seed,
        *,
        training=True,
        selection="sample",
        numerical_threads=1,
        learning_rate=0.0003,
        epochs=3,
        batch_size=256,
        clip=0.2,
        entropy=0.01,
        learning_reward_scale=0.001,
    ):
        torch.set_num_threads(numerical_threads)
        if selection not in {"sample", "argmax"}:
            raise ValueError("unknown central selection mode")
        self.options, self.seed = options, seed
        self.training, self.selection = training, selection
        self.learning_rate, self.epochs, self.batch_size = (
            learning_rate,
            epochs,
            batch_size,
        )
        self.clip, self.entropy, self.scale = clip, entropy, learning_reward_scale
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.model, self.optimizer, self.shape, self.layout = None, None, None, None
        self.models = torch.nn.ModuleDict()
        self.pack = resolve(options.network).pack
        self.records, self.updates, self.last_losses = [], 0, {}
        self.begin_episode()

    def begin_episode(self):
        self.prefix, self.prefix_actions, self.prefix_tick = [], [], None
        self.decision_count, self.inference_seconds = 0, 0.0

    def _initialize(self, shape, layout):
        if self.model is not None:
            if self.shape != shape or self.layout != layout:
                raise ValueError("central checkpoint/observation layout mismatch")
            return
        self.shape, self.layout = shape, layout
        # Model initialization must not consume another component's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = construct(
                self.options.network,
                group="central",
                options=self.options,
                value_head=True,
                context_size=shape[0],
                token_size=shape[1],
            ).eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.models["central"] = self.model

    def choose(self, request, critic=None):
        self._initialize(
            (len(request.inputs["context"]), len(request.inputs["token"])),
            request.inputs["layout"],
        )
        started = time.perf_counter()
        if request.tick != self.prefix_tick:
            self.prefix, self.prefix_actions = [], []
            self.prefix_tick = request.tick
        inputs = dict(request.inputs)
        inputs.update(
            prefix=tuple(self.prefix), prefix_actions=tuple(self.prefix_actions)
        )
        request = replace(request, inputs=freeze(inputs))
        forced = len(request.candidates) == 1
        if forced and not self.training:
            index = 0
        else:
            with torch.no_grad():
                batch = self.pack(
                    [request],
                    critics=[critic] if self.training else None,
                    options=self.options,
                )
                if self.training:
                    logits, values = self.model(batch)
                else:
                    logits = self.model.actor(batch)
                probabilities = logits[0].softmax(-1)
                index = (
                    int(torch.multinomial(probabilities, 1, generator=self.generator))
                    if self.selection == "sample"
                    else int(logits[0].argmax())
                )
                log_prob = float(logits[0].log_softmax(-1)[index])
            if self.training:
                self.records.append(
                    {
                        "request": request,
                        "critic": critic,
                        "index": index,
                        "action": request.candidates[index].identity,
                        "log_prob": log_prob,
                        "value": float(values[0]),
                        "actor_mask": not forced,
                    }
                )
        selected = request.candidates[index]
        self.prefix.append(
            (*request.inputs["token"], *request.inputs["candidates"][index])
        )
        self.prefix_actions.append(
            (
                request.group,
                request.owner,
                request.stage,
                selected.identity,
                selected.action,
            )
        )
        self.decision_count += 1
        self.inference_seconds += time.perf_counter() - started
        return selected.action

    def needs_critic(self, request):
        return self.training

    @property
    def decoder(self):
        return "policy-sampling" if self.selection == "sample" else "policy-argmax"

    def update(self, rewards, *, terminal, final_critic=None):
        if not self.records:
            self.last_losses = {}
            return {}
        from types import SimpleNamespace

        from smartsom.domain.base_decisions import GROUPS
        from smartsom.learning.returns import prepare_returns as estimate

        estimate(
            self.records,
            rewards,
            SimpleNamespace(
                gamma=1.0,
                learning_reward_scale=self.scale,
                return_estimator="episode_return",
                gae_lambda=0.95,
            ),
            terminal=terminal,
            final_critic=final_critic,
            model_for=lambda request: self.model,
        )
        for group in GROUPS:
            rows = [r for r in self.records if r["request"].group == group]
            if rows:
                advantages = torch.tensor([r["raw_advantage"] for r in rows])
                advantages = (advantages - advantages.mean()) / advantages.std(
                    unbiased=False
                ).clamp(min=1e-6)
                for row, adv in zip(rows, advantages, strict=True):
                    row["advantage"] = float(adv)
        losses = {}
        for _ in range(self.epochs):
            self.rng.shuffle(self.records)
            for offset in range(0, len(self.records), self.batch_size):
                rows = self.records[offset : offset + self.batch_size]
                batch = self.pack(
                    [r["request"] for r in rows],
                    critics=[r["critic"] for r in rows],
                    options=self.options,
                )
                logits, values = self.model(batch)
                loss, actor, critic = objective(
                    logits, values, rows, self.clip, self.entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite central PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.updates += 1
                losses = {
                    "actor": float(actor.detach()),
                    "critic": float(critic.detach()),
                    "decisions": len(self.records),
                    **self.episode_metrics(),
                }
        self.records = []
        self.last_losses = {"central": losses}
        return self.last_losses

    def episode_metrics(self):
        return {
            "decisions": self.decision_count,
            "inference_seconds": self.inference_seconds,
            "parameters": sum(p.numel() for p in self.model.parameters())
            if self.model
            else 0,
        }

    def sampling_state(self):
        return {
            "shape": self.shape,
            "layout": self.layout,
            "seed": self.seed,
            "model": self.model.state_dict() if self.model else None,
        }

    def load_sampling_state(self, state):
        self.seed = state["seed"]
        if state["shape"] is None:
            self.model, self.optimizer, self.shape, self.layout = None, None, None, None
            self.models = torch.nn.ModuleDict()
        else:
            self._initialize(tuple(state["shape"]), state["layout"])
            self.model.load_state_dict(state["model"])
            self.model.eval()
        self.discard_rollout()

    def get_state(self):
        return {
            **self.sampling_state(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "updates": self.updates,
            "rng": self.rng.getstate(),
            "generator": self.generator.get_state(),
        }

    def set_state(self, state):
        self.load_sampling_state(state)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.rng.setstate(state["rng"])
        self.generator.set_state(state["generator"])

    def seed_sampling(self, seed):
        self.generator.manual_seed(seed)

    def discard_rollout(self):
        self.records = []
        self.begin_episode()
