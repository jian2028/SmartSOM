"""Read-only semantic transitions usable by builtin and custom learners."""

from dataclasses import dataclass

from smartsom.domain.training import subject_rewards
from smartsom.learning.subjects import subject_key


@dataclass(frozen=True, slots=True)
class Transition:
    request: object
    action: object
    next_request: object
    reward: float
    physical_dt: int
    terminated: bool
    truncated: bool


def decision_transitions(records, rewards, *, terminal, gamma=1.0, split_stages=()):
    following = {
        subject_key(r, split_stages): r
        for r in getattr(rewards, "final_requests", {}).values()
    }
    result = []
    streams = {}
    seen = set()
    for row in reversed(records):
        request = row["request"]
        key = subject_key(request, split_stages)
        tail = key not in seen
        seen.add(key)
        nxt = following.get(key)
        end = nxt.tick if nxt is not None else len(rewards)
        if key not in streams:
            streams[key] = subject_rewards(rewards, request)
        stream = streams[key]
        reward = sum(
            gamma ** (i - request.tick) * stream[i] for i in range(request.tick, end)
        )
        result.append(
            Transition(
                request,
                request.candidates[row["index"]].identity,
                nxt,
                reward,
                end - request.tick,
                terminal and tail,
                not terminal and tail,
            )
        )
        following[key] = request
    return tuple(reversed(result))
