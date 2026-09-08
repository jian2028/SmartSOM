"""Shortest processing time among the current legal dispatch candidates."""

from smartsom.algorithms.quality import quality_candidates
from smartsom.algorithms.transport import select_transport
from smartsom.dispatch import DecisionContext, SemanticAction
from smartsom.domain.validation import _identifier


class SPTPolicy:
    def __init__(self, quality_mode: str | None = None):
        if quality_mode is not None:
            _identifier(quality_mode, "quality_mode")
        self.quality_mode = quality_mode

    def select_action(self, context: DecisionContext) -> SemanticAction:
        candidates = quality_candidates(context, self.quality_mode)
        if not candidates:
            return select_transport(context)
        return min(
            candidates,
            key=lambda candidate: (
                candidate.nominal_ticks,
                candidate.action.operation_id,
                candidate.action.processing_mode_id,
            ),
        ).action
