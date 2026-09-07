"""Events ordered by time, completion/reveal/release phase, and semantic IDs."""

from dataclasses import dataclass
from heapq import heappop, heappush

from smartsom.modules.arrivals import ArrivalEvent


@dataclass(frozen=True, order=True, slots=True)
class CompletionEvent:
    simulation_time: int
    operation_id: str
    processing_mode_id: str
    machine_id: str


class EventCalendar:
    def __init__(self) -> None:
        self._events: list[tuple[tuple, CompletionEvent | ArrivalEvent]] = []

    @property
    def next_time(self) -> int | None:
        return self._events[0][0][0] if self._events else None

    @property
    def pending(self) -> tuple[CompletionEvent | ArrivalEvent, ...]:
        return tuple(
            event for _, event in sorted(self._events, key=lambda item: item[0])
        )

    def schedule(self, event: CompletionEvent | ArrivalEvent) -> None:
        if isinstance(event, CompletionEvent):
            key = (
                event.simulation_time,
                0,
                event.operation_id,
                event.processing_mode_id,
                event.machine_id,
            )
        else:
            key = (
                event.simulation_time,
                1 if event.kind == "reveal" else 2,
                event.job_id,
            )
        heappush(self._events, (key, event))

    def pop(self) -> CompletionEvent | ArrivalEvent:
        return heappop(self._events)[1]
