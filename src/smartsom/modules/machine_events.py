"""Immutable outage index and event inputs; the engine owns all transitions."""

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from smartsom.domain.machine_events import MachineOutage, MachineOutagePlan
from smartsom.domain.models import FactorySpec


@dataclass(frozen=True, slots=True)
class MachineEvent:
    simulation_time: int
    machine_id: str
    kind: Literal["breakdown", "repair"]


@dataclass(frozen=True, slots=True)
class OutageIndex:
    """Prefix downtime supports repeated elapsed-work checks without full scans."""

    intervals: tuple[MachineOutage, ...]
    starts: tuple[int, ...]
    cumulative: tuple[int, ...]

    @classmethod
    def create(cls, rows: tuple[MachineOutage, ...]):
        totals = [0]
        for row in rows:
            totals.append(totals[-1] + row.end_time - row.start_time)
        return cls(rows, tuple(row.start_time for row in rows), tuple(totals))

    def downtime_before(self, tick: int) -> int:
        count = bisect_right(self.starts, tick)
        if not count:
            return 0
        row = self.intervals[count - 1]
        return self.cumulative[count - 1] + min(tick, row.end_time) - row.start_time

    def is_down(self, tick: int) -> bool:
        count = bisect_right(self.starts, tick)
        return bool(count and tick < self.intervals[count - 1].end_time)

    def processing_between(self, start: int, end: int) -> int:
        return end - start - self.downtime_before(end) + self.downtime_before(start)


@dataclass(frozen=True, slots=True, init=False)
class MachineEventModule:
    by_machine: Mapping[str, OutageIndex]
    events: frozenset[MachineEvent]

    def __init__(self, factory: FactorySpec, plan: MachineOutagePlan | None):
        if plan is None:
            plan = MachineOutagePlan()
        if not isinstance(plan, MachineOutagePlan):
            raise ValueError("expected MachineOutagePlan")
        plan.validate(factory)
        grouped: dict[str, list[MachineOutage]] = {
            machine.machine_id: [] for machine in factory.machines
        }
        events = []
        for row in plan.outages:
            grouped[row.machine_id].append(row)
            events.extend(
                (
                    MachineEvent(row.start_time, row.machine_id, "breakdown"),
                    MachineEvent(row.end_time, row.machine_id, "repair"),
                )
            )
        object.__setattr__(
            self,
            "by_machine",
            MappingProxyType(
                {key: OutageIndex.create(tuple(rows)) for key, rows in grouped.items()}
            ),
        )
        object.__setattr__(self, "events", frozenset(events))
