"""Canonical finite machine outages, independent of jobs and algorithms."""

from dataclasses import dataclass

from smartsom.domain.models import FactorySpec, _identifier


@dataclass(frozen=True, slots=True)
class MachineOutage:
    machine_id: str
    start_time: int
    end_time: int

    def __post_init__(self):
        _identifier(self.machine_id, "machine_id")
        if (
            type(self.start_time) is not int
            or type(self.end_time) is not int
            or not 0 <= self.start_time < self.end_time
        ):
            raise ValueError("outage requires integer 0 <= start_time < end_time")


@dataclass(frozen=True, slots=True)
class MachineOutagePlan:
    outages: tuple[MachineOutage, ...] = ()

    def __post_init__(self):
        if not isinstance(self.outages, (tuple, list)) or any(
            not isinstance(row, MachineOutage) for row in self.outages
        ):
            raise ValueError("outages must contain MachineOutage values")
        merged: list[MachineOutage] = []
        for row in sorted(
            self.outages, key=lambda row: (row.machine_id, row.start_time, row.end_time)
        ):
            if (
                merged
                and merged[-1].machine_id == row.machine_id
                and row.start_time <= merged[-1].end_time
            ):
                previous = merged.pop()
                row = MachineOutage(
                    row.machine_id,
                    previous.start_time,
                    max(previous.end_time, row.end_time),
                )
            merged.append(row)
        object.__setattr__(self, "outages", tuple(merged))

    def validate(self, factory: FactorySpec) -> None:
        unknown = {row.machine_id for row in self.outages} - {
            machine.machine_id for machine in factory.machines
        }
        if unknown:
            raise ValueError(f"outages reference unknown machines: {sorted(unknown)}")
