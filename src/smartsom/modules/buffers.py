"""Immutable capacity lookup and pure occupancy/reservation queries."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from smartsom.domain import (
    BufferReservation,
    BufferState,
    FactorySpec,
    JobLocation,
    JobPosition,
    MachineBuffers,
)


@dataclass(frozen=True, slots=True, init=False)
class BufferModule:
    limits: Mapping[str, MachineBuffers]
    enabled: bool

    def __init__(self, factory: FactorySpec, enabled: bool) -> None:
        configured = {x.machine_id: x for x in factory.buffers} if enabled else {}
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(
            self,
            "limits",
            MappingProxyType(
                {
                    m.machine_id: configured.get(
                        m.machine_id, MachineBuffers(m.machine_id)
                    )
                    for m in sorted(factory.machines, key=lambda x: x.machine_id)
                }
            ),
        )

    @property
    def finite(self) -> bool:
        return any(
            x.pre_capacity is not None or x.post_capacity is not None
            for x in self.limits.values()
        )

    def jobs(
        self,
        machine: str,
        kind: Literal["prebuffer", "postbuffer"],
        positions: Mapping[str, JobPosition],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key, p in positions.items()
                if p.location == JobLocation(kind, machine)
            )
        )

    def space(
        self,
        machine: str,
        kind: Literal["prebuffer", "postbuffer"],
        positions: Mapping[str, JobPosition],
        reservations: Collection[BufferReservation] = (),
    ) -> bool:
        capacity = getattr(
            self.limits[machine],
            "pre_capacity" if kind == "prebuffer" else "post_capacity",
        )
        reserved = (
            sum(r.machine_id == machine for r in reservations)
            if kind == "prebuffer"
            else 0
        )
        return (
            capacity is None
            or len(self.jobs(machine, kind, positions)) + reserved < capacity
        )

    def snapshots(
        self,
        positions: Mapping[str, JobPosition],
        reservations: Collection[BufferReservation],
    ) -> tuple[BufferState, ...]:
        if not self.enabled:
            return ()
        return tuple(
            BufferState(
                key,
                limit.pre_capacity,
                limit.post_capacity,
                self.jobs(key, "prebuffer", positions),
                self.jobs(key, "postbuffer", positions),
                tuple(
                    sorted(
                        (r for r in reservations if r.machine_id == key),
                        key=lambda r: r.transport_sequence,
                    )
                ),
            )
            for key, limit in self.limits.items()
        )
