"""Replay a finite semantic action sequence through the public step API."""

from collections.abc import Iterable

from smartsom.dispatch import SemanticAction
from smartsom.domain import FactorySpec, WorkloadInstance
from smartsom.domain.arrivals import ArrivalPlan, DecisionTrigger
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.quality import ProbabilityVisibility, QualityPlan
from smartsom.engine.result import SimulationResult
from smartsom.engine.simulator import Simulator


class ReplayError(ValueError):
    """The action stream is shorter or longer than the episode."""


def replay(
    factory: FactorySpec,
    workload: WorkloadInstance,
    actions: Iterable[SemanticAction],
    *,
    arrivals: ArrivalPlan | None = None,
    decision_trigger: DecisionTrigger = "dispatch_available",
    processing_times: ProcessingTimePlan | None = None,
    machine_events: MachineOutagePlan | None = None,
    transport_enabled: bool = False,
    buffers_enabled: bool = False,
    holding_buffer_enabled: bool = False,
    quality: QualityPlan | None = None,
    quality_probability_visibility: ProbabilityVisibility = "public",
) -> SimulationResult:
    simulator = Simulator(
        factory,
        workload,
        arrivals=arrivals,
        decision_trigger=decision_trigger,
        processing_times=processing_times,
        machine_events=machine_events,
        transport_enabled=transport_enabled,
        buffers_enabled=buffers_enabled,
        holding_buffer_enabled=holding_buffer_enabled,
        quality=quality,
        quality_probability_visibility=quality_probability_visibility,
    )
    result = None
    for index, action in enumerate(actions):
        if simulator.current_decision is None:
            raise ReplayError(
                f"extra action at index {index} after completion: {action!r}"
            )
        result = simulator.step(action)
    if not isinstance(result, SimulationResult):
        context = simulator.current_decision
        raise ReplayError(
            f"action sequence ended before completion at tick {context.simulation_time}"
        )
    return result
