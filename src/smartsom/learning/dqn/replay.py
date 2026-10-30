"""Complete per-resource transitions; unresolved safety-cut tails are omitted."""

from smartsom.domain.base_decisions import GROUPS


def transitions(records, rewards, *, terminal, scale):
    truncated_tails = 0
    scales = scale if isinstance(scale, dict) else dict.fromkeys(GROUPS, scale)
    suffixes = {}
    for factor in set(scales.values()):
        suffix = [0.0] * (len(rewards) + 1)
        for i in reversed(range(len(rewards))):
            suffix[i] = suffix[i + 1] + rewards[i] * factor
        suffixes[factor] = suffix
    following = {}
    pending = {g: [] for g in GROUPS}
    for row in reversed(records):
        request = row["request"]
        suffix = suffixes[scales[request.group]]
        key = (request.group, request.owner)
        next_request = following.get(key)
        if next_request is None and not terminal:
            truncated_tails += 1
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
    return pending, truncated_tails
