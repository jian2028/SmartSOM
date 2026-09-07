"""Completion, machine and arrival phases ordered by time and semantic IDs."""

from dataclasses import dataclass
from heapq import heappop, heappush

from smartsom.modules.arrivals import ArrivalEvent
from smartsom.modules.machine_events import MachineEvent
from smartsom.modules.transport import TransportEvent


@dataclass(frozen=True, order=True, slots=True)
class CompletionEvent:
    simulation_time: int
    operation_id: str
    processing_mode_id: str
    machine_id: str


type Event = CompletionEvent | ArrivalEvent | MachineEvent | TransportEvent


class EventCalendar:
    def __init__(self) -> None:
        self._events: list[tuple[tuple, Event]] = []
        self._completions: dict[str, CompletionEvent] = {}

    def _live(self, event: Event) -> bool:
        return (
            not isinstance(event, CompletionEvent)
            or self._completions.get(event.operation_id) is event
        )

    def _discard_cancelled(self) -> None:
        while self._events and not self._live(self._events[0][1]):
            heappop(self._events)

    def cancel_completion(self, operation_id: str) -> None:
        del self._completions[operation_id]

    @property
    def next_time(self) -> int | None:
        self._discard_cancelled()
        return self._events[0][0][0] if self._events else None

    @property
    def pending(self) -> tuple[Event, ...]:
        return tuple(
            event
            for _, event in sorted(self._events, key=lambda item: item[0])
            if self._live(event)
        )

    def schedule(self, event: Event) -> None:
        if isinstance(event, CompletionEvent):
            if event.operation_id in self._completions:
                raise ValueError("operation already has an active completion")
            self._completions[event.operation_id] = event
            key = (
                event.simulation_time,
                0,
                event.operation_id,
                event.processing_mode_id,
                event.machine_id,
            )
        elif isinstance(event, MachineEvent):
            key = (
                event.simulation_time,
                1 if event.kind == "breakdown" else 2,
                event.machine_id,
            )
        elif isinstance(event, TransportEvent):
            key = (
                event.simulation_time,
                3 if event.kind == "pickup" else 4,
                event.agv_id,
                event.job_id,
                event.transport_sequence,
            )
        else:
            key = (
                event.simulation_time,
                5 if event.kind == "reveal" else 6,
                event.job_id,
            )
        heappush(self._events, (key, event))

    def pop(self) -> Event:
        self._discard_cancelled()
        event = heappop(self._events)[1]
        if isinstance(event, CompletionEvent):
            del self._completions[event.operation_id]
        return event
