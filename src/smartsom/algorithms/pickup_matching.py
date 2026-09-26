"""Exact prefix matching; independent reproducible environment tie randomness."""

import hashlib
import json
import random
from functools import lru_cache


def matching_rng(seed, episode, source, tick, jobs, vehicles):
    identity = json.dumps(
        [seed, episode, source, tick, list(jobs), sorted(vehicles)],
        separators=(",", ":"),
    )
    return random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest()))


def global_optimal(jobs, vehicles, cost, rng):
    """Uniformly sample every optimum injection, including served vehicle subsets."""
    jobs, vehicles = tuple(jobs), tuple(sorted(vehicles))
    if len(jobs) > len(vehicles):
        raise ValueError("prefix exceeds available service vehicles")
    costs = tuple(tuple(cost(j, a) for a in vehicles) for j in jobs)

    @lru_cache(None)
    def solve(i, used):
        if i == len(jobs):
            return 0, 1
        best, count = float("inf"), 0
        for a in range(len(vehicles)):
            if used & (1 << a):
                continue
            tail, ways = solve(i + 1, used | (1 << a))
            total = costs[i][a] + tail
            if total < best:
                best, count = total, ways
            elif total == best:
                count += ways
        return best, count

    optimum, count = solve(0, 0)
    if optimum == float("inf"):
        raise ValueError("no finite feasible pickup matching")
    selected, used = [], 0
    for i, job in enumerate(jobs):
        best, ways = solve(i, used)
        draw = rng.randrange(ways)
        for a, vehicle in enumerate(vehicles):
            if used & (1 << a):
                continue
            tail, branch_count = solve(i + 1, used | (1 << a))
            if costs[i][a] + tail != best:
                continue
            if draw < branch_count:
                selected.append((vehicle, job))
                used |= 1 << a
                break
            draw -= branch_count
    return tuple(selected)


def priority_greedy(jobs, vehicles, cost, rng, inspection_tie=None):
    remaining, result = set(vehicles), []
    if len(jobs) > len(remaining):
        raise ValueError("prefix exceeds available service vehicles")
    for job in jobs:
        best = min(cost(job, a) for a in remaining)
        choices = sorted(a for a in remaining if cost(job, a) == best)
        if best == float("inf"):
            raise ValueError("no finite feasible pickup matching")
        if inspection_tie is not None:
            second = min(inspection_tie(job, a) for a in choices)
            choices = [a for a in choices if inspection_tie(job, a) == second]
        vehicle = rng.choice(choices)
        remaining.remove(vehicle)
        result.append((vehicle, job))
    return tuple(result)


MATCHING_RULES = {
    "global_optimal": global_optimal,
    "priority_greedy": priority_greedy,
}
