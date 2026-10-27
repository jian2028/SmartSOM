"""Optional CPU PPO for Base decisions; five groups and a public-state critic.

Full episodes provide gamma=1 return-to-go targets. The safety horizon uses a
critic bootstrap, never a falsely completed mean tardiness. Same-tick choices
each have their own likelihood but share the same physical reward suffix.
"""

import random
from dataclasses import asdict

import torch
from torch import nn

from smartsom.domain.base_decisions import (
    FEATURES,
    GROUPS,
    SET_NAMES,
    BaseCandidate,
    BaseDecision,
)

STAGES = {"process": 0, "pickup": 0, "delivery": 1, "accept": 0, "service": 1}


def mlp(input_size, width, output):
    return nn.Sequential(
        nn.Linear(input_size, width), nn.Tanh(), nn.Linear(width, output)
    )


class GroupNetwork(nn.Module):
    def __init__(self, width, *, empty_wait=False):
        super().__init__()
        self.encoders = nn.ModuleList(mlp(FEATURES, width, width) for _ in SET_NAMES)
        self.context = mlp(
            FEATURES + len(SET_NAMES) * (2 * width + 1), width * 2, width
        )
        self.candidate = mlp(FEATURES, width, width)
        self.heads = nn.ModuleList(mlp(2 * width, width, 1) for _ in range(2))
        self.stop = mlp(width, width, 1)
        self.critic = mlp(FEATURES, width * 2, 1)
        self.empty_wait = mlp(width, width, 1) if empty_wait else None

    def actor(self, batch):
        summaries = []
        for encoder, (values, mask) in zip(self.encoders, batch["sets"], strict=True):
            encoded = encoder(values)
            count = mask.sum(1, keepdim=True)
            mean = (encoded * mask.unsqueeze(-1)).sum(1) / count.clamp(min=1)
            maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            summaries.extend((mean, maximum, count / batch["count_scale"]))
        context = self.context(torch.cat((batch["context"], *summaries), dim=1))
        candidate = self.candidate(batch["candidates"])
        combined = torch.cat(
            (candidate, context.unsqueeze(1).expand(-1, candidate.shape[1], -1)), dim=2
        )
        heads = torch.stack([head(combined).squeeze(-1) for head in self.heads], dim=-1)
        logits = heads.gather(
            2, batch["stage"][:, None, None].expand(-1, candidate.shape[1], 1)
        ).squeeze(-1)
        logits = torch.where(batch["stop"], self.stop(context), logits)
        if self.empty_wait is not None:
            logits = torch.where(batch["empty_wait"], self.empty_wait(context), logits)
        logits = logits.masked_fill(~batch["mask"], -torch.inf)
        return logits

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def encode_requests(requests):
    """Cache immutable input tensors for one update, never learned embeddings."""
    return {
        id(r): (
            torch.tensor([c.features for c in r.candidates]),
            tuple(torch.tensor(values) for values in r.sets),
        )
        for r in requests
    }


def pack(requests, critics, count_scale, encoded=None, *, allow_empty_wait=False):
    n = len(requests)
    max_candidates = max(len(r.candidates) for r in requests)
    candidate = torch.zeros(n, max_candidates, FEATURES)
    mask = torch.zeros(n, max_candidates, dtype=torch.bool)
    stop = mask.clone()
    empty_wait = mask.clone() if allow_empty_wait else None
    sets = []
    for index in range(len(SET_NAMES)):
        size = max(1, max(len(r.sets[index]) for r in requests))
        values = torch.zeros(n, size, FEATURES)
        valid = torch.zeros(n, size, dtype=torch.bool)
        for i, r in enumerate(requests):
            if r.sets[index]:
                values[i, : len(r.sets[index])] = (
                    encoded[id(r)][1][index]
                    if encoded is not None
                    else torch.tensor(r.sets[index])
                )
                valid[i, : len(r.sets[index])] = True
        sets.append((values, valid))
    for i, r in enumerate(requests):
        count = len(r.candidates)
        candidate[i, :count] = (
            encoded[id(r)][0]
            if encoded is not None
            else torch.tensor([c.features for c in r.candidates])
        )
        mask[i, :count] = True
        stop[i, :count] = torch.tensor([c.action == ("STOP",) for c in r.candidates])
        if empty_wait is not None:
            empty_wait[i, :count] = torch.tensor(
                [c.action == ("EMPTY_WAIT",) for c in r.candidates]
            )
    return {
        "context": torch.tensor([r.context for r in requests]),
        "sets": sets,
        "candidates": candidate,
        "mask": mask,
        "stop": stop,
        "empty_wait": empty_wait,
        "stage": torch.tensor([STAGES[r.stage] for r in requests]),
        "critic": torch.tensor(critics, dtype=torch.float32)
        if critics is not None
        else None,
        "count_scale": count_scale,
    }


class BasePPO:
    def __init__(
        self, options, seed=0, *, training=True, selection="sample", numerical_threads=1
    ):
        torch.set_num_threads(numerical_threads)
        torch.manual_seed(seed)
        self.options, self.training, self.selection = options, training, selection
        self.models = nn.ModuleDict(
            {
                g: GroupNetwork(
                    options.width,
                    empty_wait=options.allow_empty_wait and g == "dispatcher",
                )
                for g in GROUPS
            }
        )
        if options.critic_initial_scale != 1.0:
            with torch.no_grad():
                for model in self.models.values():
                    model.critic[-1].weight.mul_(options.critic_initial_scale)
                    model.critic[-1].bias.mul_(options.critic_initial_scale)
        self.optimizers = {
            g: torch.optim.Adam(self.models[g].parameters(), lr=options.learning_rate)
            for g in GROUPS
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
            model = self.models[request.group]
            inputs = pack(
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
        if self.options.learning_reward_scale != 1.0:
            rewards = [r * self.options.learning_reward_scale for r in rewards]
        suffix = [0.0] * (len(rewards) + 1)
        for i in reversed(range(len(rewards))):
            suffix[i] = rewards[i] + suffix[i + 1]
        losses = {}
        for group in GROUPS:
            rows = [r for r in self.records if r["request"].group == group]
            if not rows:
                continue
            with torch.no_grad():
                bootstrap = (
                    0.0
                    if terminal
                    else float(
                        self.models[group].critic(torch.tensor([final_critic])).item()
                    )
                )
            for row in rows:
                row["return"] = suffix[row["request"].tick] + bootstrap
            advantages = torch.tensor([r["return"] - r["value"] for r in rows])
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp(min=1e-6)
            for row, adv in zip(rows, advantages, strict=True):
                row["advantage"] = float(adv)
            encoded = encode_requests([r["request"] for r in rows])
            for _ in range(self.options.epochs):
                self.rng.shuffle(rows)
                for offset in range(0, len(rows), self.options.batch_size):
                    batch = rows[offset : offset + self.options.batch_size]
                    inputs = pack(
                        [r["request"] for r in batch],
                        [r["critic"] for r in batch],
                        self.options.count_scale,
                        encoded=encoded,
                        allow_empty_wait=self.options.allow_empty_wait,
                    )
                    logits, values = self.models[group](inputs)
                    distribution = torch.distributions.Categorical(logits=logits)
                    selected = torch.tensor([r["index"] for r in batch])
                    old = torch.tensor([r["log_prob"] for r in batch])
                    ratio = (distribution.log_prob(selected) - old).exp()
                    advantage = torch.tensor([r["advantage"] for r in batch])
                    actor_mask = torch.tensor([r["actor_mask"] for r in batch])
                    contributions = torch.minimum(
                        ratio * advantage,
                        ratio.clamp(1 - self.options.clip, 1 + self.options.clip)
                        * advantage,
                    )
                    actor = -(
                        contributions * actor_mask
                    ).sum() / actor_mask.sum().clamp(min=1)
                    target = torch.tensor([r["return"] for r in batch])
                    critic_loss = torch.nn.functional.smooth_l1_loss(values, target)
                    loss = (
                        actor
                        + critic_loss
                        - self.options.entropy * distribution.entropy().mean()
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite Base PPO loss")
                    optimizer = self.optimizers[group]
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.models[group].parameters(), 1.0)
                    optimizer.step()
                    self.updates += 1
                    losses[group] = {
                        "actor": float(actor.detach()),
                        "critic": float(critic_loss.detach()),
                        "decisions": len(rows),
                    }
        self.records = []
        self.last_losses = losses
        return losses

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "contract": "smartsom.base-ppo/v1",
                "options": self.options.model_dump(mode="json", by_alias=True),
                "models": self.models.state_dict(),
                "optimizers": {g: o.state_dict() for g, o in self.optimizers.items()},
                "updates": self.updates,
                "rng": self.rng.getstate(),
                "generator": self.generator.get_state(),
                "bound_policy_states": self.bound_policy_states,
                "torch_rng": torch.get_rng_state(),
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        from smartsom.config.base_diagnostic import BaseAlgorithm

        saved_options = BaseAlgorithm.model_validate(payload["options"])

        def identity(options):
            result = options.model_dump(mode="json", by_alias=True)
            for provider in result["providers"].values():
                provider.pop("path")  # Relocation is allowed; content/factory is not.
            return result

        if payload["contract"] != "smartsom.base-ppo/v1" or identity(
            saved_options
        ) != identity(self.options):
            raise ValueError("Base checkpoint contract mismatch")
        self.models.load_state_dict(payload["models"])
        for g, o in self.optimizers.items():
            o.load_state_dict(payload["optimizers"][g])
        self.updates = payload["updates"]
        self.rng.setstate(payload["rng"])
        self.generator.set_state(payload["generator"])
        self.bound_policy_states = payload.get("bound_policy_states", {})
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])


def request_from_dict(data):
    data = dict(data)
    data["candidates"] = tuple(
        BaseCandidate(c["identity"], tuple(c["action"]), tuple(c["features"]))
        for c in data["candidates"]
    )
    return BaseDecision(**data)


def recorded_request(request):
    return asdict(request)
