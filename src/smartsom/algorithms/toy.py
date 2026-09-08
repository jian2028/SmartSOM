"""Deterministic test providers; neither owns or changes simulator state."""

from smartsom.algorithms.quality import quality_candidates
from smartsom.algorithms.transport import select_transport
from smartsom.dispatch import DecisionContext, SemanticAction
from smartsom.domain.validation import _identifier


class ScriptError(ValueError):
    pass


class FirstFeasiblePolicy:
    def __init__(self, quality_mode: str | None = None):
        if quality_mode is not None:
            _identifier(quality_mode, "quality_mode")
        self.quality_mode = quality_mode

    def select_action(self, context: DecisionContext) -> SemanticAction:
        candidates = quality_candidates(context, self.quality_mode)
        if not candidates:
            return select_transport(context)
        return min(
            (candidate.action for candidate in candidates),
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
