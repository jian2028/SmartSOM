"""Deterministic test providers; neither owns or changes simulator state."""

from smartsom.algorithms.transport import select_transport
from smartsom.dispatch import DecisionContext, SemanticAction


class ScriptError(ValueError):
    pass


class FirstFeasiblePolicy:
    def select_action(self, context: DecisionContext) -> SemanticAction:
        if not context.candidates:
            return select_transport(context)
        return min(
            (candidate.action for candidate in context.candidates),
            key=lambda action: (action.operation_id, action.processing_mode_id),
        )


class ScriptedPolicy:
    def __init__(self, actions: tuple[SemanticAction, ...]) -> None:
        self._actions = tuple(actions)
        self._position = 0

    def select_action(self, context: DecisionContext) -> SemanticAction:
        if self._position == len(self._actions):
            raise ScriptError(
                f"script exhausted before completion at tick {context.simulation_time}"
            )
        action = self._actions[self._position]
        self._position += 1
        return action

    def ensure_exhausted(self) -> None:
        if self._position != len(self._actions):
            raise ScriptError("script has extra actions after completion")
