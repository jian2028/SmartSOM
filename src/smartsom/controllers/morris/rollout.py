"""Native gamma-one returns and role-wise advantage normalization."""

import torch

from smartsom.domain.base_decisions import GROUPS
from smartsom.learning.ppo.rollout import reward_suffix


def prepare_returns(records, rewards, scale, bootstrap):
    suffix = reward_suffix(rewards, scale)
    for row in records:
        row["return"] = suffix[row["request"].tick] + bootstrap
    for group in GROUPS:
        rows = [r for r in records if r["request"].group == group]
        if not rows:
            continue
        advantages = torch.tensor([r["return"] - r["value"] for r in rows])
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp(min=1e-6)
        for row, advantage in zip(rows, advantages, strict=True):
            row["advantage"] = float(advantage)


def objective(logits, values, rows, clip, entropy_weight):
    distribution = torch.distributions.Categorical(logits=logits)
    selected = torch.tensor([r["index"] for r in rows])
    old = torch.tensor([r["log_prob"] for r in rows])
    ratio = (distribution.log_prob(selected) - old).exp()
    advantage = torch.tensor([r["advantage"] for r in rows])
    actor_mask = torch.tensor([r["actor_mask"] for r in rows])
    contributions = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    )
    actor = -(contributions * actor_mask).sum() / actor_mask.sum().clamp(min=1)
    targets = torch.tensor([r["return"] for r in rows])
    critic = torch.nn.functional.smooth_l1_loss(values, targets)
    entropy = distribution.entropy().mean()
    return actor + critic - entropy_weight * entropy, actor, critic
