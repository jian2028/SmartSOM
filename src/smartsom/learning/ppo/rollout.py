"""Physical-time return suffixes; each same-tick decision sees the same suffix."""


def reward_suffix(rewards, scale=1.0):
    if scale != 1.0:
        rewards = [r * scale for r in rewards]
    suffix = [0.0] * (len(rewards) + 1)
    for i in reversed(range(len(rewards))):
        suffix[i] = rewards[i] + suffix[i + 1]
    return suffix
