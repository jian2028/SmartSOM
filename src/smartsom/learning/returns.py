"""Physical-time discounted returns and per-subject decision-step GAE."""

from collections import defaultdict

import torch

from smartsom.domain.training import bootstrap_input, subject_rewards
from smartsom.learning.subjects import subject_key
from smartsom.learning.value import values


def prepare_returns(
    rows, rewards, settings, *, terminal, final_critic, model_for, split_stages=()
):
    subjects = defaultdict(list)
    for row in rows:
        r = row["request"]
        subjects[subject_key(r, split_stages)].append(row)
    suffixes, bootstraps = {}, {}
    for sequence in subjects.values():
        request = sequence[-1]["request"]
        stream = subject_rewards(rewards, request)
        gamma, scale = settings.gamma, settings.learning_reward_scale
        cache_key = None if stream is rewards else (request.group, request.owner)
        if cache_key not in suffixes:
            suffix = [0.0] * (len(stream) + 1)
            for t in reversed(range(len(stream))):
                suffix[t] = stream[t] * scale + gamma * suffix[t + 1]
            suffixes[cache_key] = suffix
        suffix = suffixes[cache_key]
        bootstrap = 0.0
        if not terminal:
            model = model_for(request)
            value_input = bootstrap_input(final_critic, request)
            key = id(model), id(value_input)
            if key not in bootstraps:
                with torch.no_grad():
                    bootstraps[key] = float(values(model, [value_input])[0])
            bootstrap = bootstraps[key]
        next_value, next_tick, advantage = bootstrap, len(stream), 0.0
        for row in reversed(sequence):
            t = row["request"].tick
            discount = gamma ** (next_tick - t)
            reward = suffix[t] - discount * suffix[next_tick]
            if settings.return_estimator == "gae":
                delta = reward + discount * next_value - row["value"]
                advantage = delta + discount * settings.gae_lambda * advantage
                row["return"] = advantage + row["value"]
            else:
                row["return"] = suffix[t] + gamma ** (len(stream) - t) * bootstrap
                advantage = row["return"] - row["value"]
            row["raw_advantage"] = advantage
            next_value, next_tick = row["value"], t
