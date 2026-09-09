"""Immutable episode inputs and limits, independent of any learner framework."""

from dataclasses import dataclass

from smartsom.domain import FactorySpec, WorkloadInstance
from smartsom.domain.arrivals import ArrivalPlan, DecisionTrigger
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.quality import ProbabilityVisibility, QualityPlan


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    factory: FactorySpec
    workload: WorkloadInstance
    arrivals: ArrivalPlan | None = None
    decision_trigger: DecisionTrigger = "dispatch_available"
    processing_times: ProcessingTimePlan | None = None
    machine_events: MachineOutagePlan | None = None
    transport_enabled: bool = False
    buffers_enabled: bool = False
    holding_buffer_enabled: bool = False
    quality: QualityPlan | None = None
    quality_probability_visibility: ProbabilityVisibility = "public"

    def options(self):
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("factory", "workload")
        }


@dataclass(frozen=True, slots=True)
class EpisodeLimits:
    max_decisions: int = 1024
    max_ticks: int = 10000

    def __post_init__(self):
        if any(
            type(x) is not int or x < 1 for x in (self.max_decisions, self.max_ticks)
        ):
            raise ValueError("episode limits must be positive integers")


class EpisodeStartFailure(RuntimeError):
    """A legitimate failed episode that has no initial learner decision."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def resource_outcome(*, tick, rounds, reason, limits, rewarded_tick, total_reward):
    """Shared joint-round accounting for ParallelEnv and ordinary run inference."""
    terminated, truncated = reason is not None, False
    if tick > limits.max_ticks or (
        reason != "completed"
        and (tick >= limits.max_ticks or rounds >= limits.max_decisions)
    ):
        reason, terminated, truncated = "budget_exhausted", False, True
    reward = -float(tick - rewarded_tick)
    if reason and reason != "completed":
        reward = -float(max(limits.max_ticks + 1, tick)) - total_reward
    return reason, reward, terminated, truncated


def central_outcome(*, tick, decisions, reason, limits, rewarded_tick, total_reward):
    """The original central Gym reward/termination accounting, shared with inference."""
    truncated = False
    if reason not in ("deadlock", "policy_stalled", "invalid_action") and (
        tick > limits.max_ticks
        or (
            reason != "completed"
            and (tick >= limits.max_ticks or decisions >= limits.max_decisions)
        )
    ):
        reason, truncated = "budget_exhausted", True
    reward = -(tick - rewarded_tick)
    if reason is not None and reason != "completed":
        reward = -max(limits.max_ticks + 1, tick) - total_reward
    return reason, reward, reason is not None and not truncated, truncated
