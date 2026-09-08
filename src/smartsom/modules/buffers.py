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
from smartsom.domain.buffers import (
    HoldingBuffer,
    HoldingBufferState,
    HoldingReservation,
)


@dataclass(frozen=True, slots=True, init=False)
class BufferModule:
    limits: Mapping[str, MachineBuffers]
    enabled: bool
    holding: HoldingBuffer | None

    def __init__(
        self, factory: FactorySpec, enabled: bool, holding_enabled: bool = False
    ) -> None:
        object.__setattr__(
            self, "holding", factory.holding_buffer if holding_enabled else None
        )
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
        kind: Literal["prebuffer", "postbuffer", "holding"],
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
        kind: Literal["prebuffer", "postbuffer", "holding"],
        positions: Mapping[str, JobPosition],
        reservations: Collection[BufferReservation] = (),
    ) -> bool:
        capacity = (
            self.holding.capacity
            if kind == "holding"
            else getattr(
                self.limits[machine],
                "pre_capacity" if kind == "prebuffer" else "post_capacity",
            )
        )
        reserved = (
            sum(
                isinstance(r, BufferReservation) and r.machine_id == machine
                for r in reservations
            )
            if kind == "prebuffer"
            else sum(
                isinstance(r, HoldingReservation) and r.buffer_id == machine
                for r in reservations
            )
            if kind == "holding"
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
                        (
                            r
                            for r in reservations
                            if isinstance(r, BufferReservation) and r.machine_id == key
                        ),
                        key=lambda r: r.transport_sequence,
                    )
                ),
            )
            for key, limit in self.limits.items()
        )

    def holding_snapshot(self, positions, reservations=()) -> HoldingBufferState | None:
        if self.holding is None:
            return None
        h = self.holding
        return HoldingBufferState(
            h.buffer_id,
            h.node_id,
            h.capacity,
            self.jobs(h.buffer_id, "holding", positions),
            tuple(
                sorted(
                    (r for r in reservations if isinstance(r, HoldingReservation)),
                    key=lambda r: r.transport_sequence,
                )
            ),
        )
