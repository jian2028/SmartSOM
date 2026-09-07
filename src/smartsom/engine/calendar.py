"""Completion calendar ordered by time and semantic IDs, never insertion order."""

from dataclasses import dataclass
from heapq import heappop, heappush


@dataclass(frozen=True, order=True, slots=True)
class CompletionEvent:
    simulation_time: int
    operation_id: str
    processing_mode_id: str
    machine_id: str


class EventCalendar:
    def __init__(self) -> None:
        self._events: list[CompletionEvent] = []

    @property
    def next_time(self) -> int | None:
        return self._events[0].simulation_time if self._events else None

    @property
    def pending(self) -> tuple[CompletionEvent, ...]:
        return tuple(sorted(self._events))

    def schedule(self, event: CompletionEvent) -> None:
        heappush(self._events, event)

    def pop(self) -> CompletionEvent:
        return heappop(self._events)
