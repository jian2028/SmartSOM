"""Framework-free identities and committed learning data."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearningContext:
    tick: int
    group: str
    owner: str
    stage: str = "value"
    bootstrap: bool = False
    rejected_owners: tuple = ()


@dataclass(frozen=True, slots=True)
class CommittedAction:
    tick: int
    group: str
    owner: str
    stage: str
    identity: str
    action: object


@dataclass(frozen=True, slots=True)
class RewardTransition:
    start: int
    end: int
    actions: tuple
    before: object
    after: object
    events: tuple
    outcomes: object
    terminated: bool
    truncated: bool


class RewardTimeline(list):
    """Team timeline plus additive role and semantic-agent timelines."""

    def __init__(self):
        super().__init__()
        self.roles, self.agents = {}, {}
        self.final_requests = {}
        self.end_reason = None

    def append_result(self, result, elapsed):
        from math import isfinite

        team = float(result.get("team", 0.0))
        roles, agents = result.get("roles", {}), result.get("agents", {})
        if set(result) - {"team", "roles", "agents"}:
            raise ValueError("unknown reward output field")
        if elapsed <= 0 or not all(
            isfinite(float(x)) for x in (team, *roles.values(), *agents.values())
        ):
            raise ValueError("reward must be finite on a positive physical interval")
        if any(not isinstance(k, str) for k in roles) or any(
            not isinstance(k, tuple)
            or len(k) != 2
            or not all(isinstance(v, str) for v in k)
            for k in agents
        ):
            raise ValueError(
                "reward identities require role strings and (role, owner) tuples"
            )
        old = len(self)
        self.extend([0.0] * (elapsed - 1))
        self.append(team)
        for target, values in ((self.roles, roles), (self.agents, agents)):
            for key in target.keys() | values.keys():
                if key not in target:
                    target[key] = [0.0] * old
                row = target[key]
                row.extend([0.0] * (elapsed - 1))
                row.append(float(values.get(key, 0.0)))
        return {
            "team": team,
            "roles": {k: float(v) for k, v in roles.items()},
            "agents": {k: float(v) for k, v in agents.items()},
        }

    def for_subject(self, group, owner):
        role, agent = self.roles.get(group), self.agents.get((group, owner))
        if role is None and agent is None:
            return self
        return [
            v + (role[i] if role else 0.0) + (agent[i] if agent else 0.0)
            for i, v in enumerate(self)
        ]


def subject_rewards(rewards, request):
    return (
        rewards.for_subject(request.group, request.owner)
        if isinstance(rewards, RewardTimeline)
        else rewards
    )


class BootstrapInputs(dict):
    """Critic inputs by semantic learning subject at the episode boundary."""


def bootstrap_input(final, request):
    return (
        final[(request.group, request.owner)]
        if isinstance(final, BootstrapInputs)
        else final
    )


_PUBLIC_EVENT_FIELDS = frozenset(
    {
        "tick",
        "kind",
        "machine",
        "job",
        "mode",
        "nominal_ticks",
        "agv",
        "owner",
        "slot",
        "demand",
        "operation",
        "elapsed",
        "before",
        "after",
        "actor",
        "reason",
        "destination",
        "quality",
        "first",
    }
)


def public_event(event):
    """Only observable event facts; sampled completion times stay private."""
    from smartsom.domain.public_facts import freeze

    return freeze({k: v for k, v in event.items() if k in _PUBLIC_EVENT_FIELDS})
