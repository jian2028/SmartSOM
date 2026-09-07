"""Shortest processing time among the current legal dispatch candidates."""

from smartsom.algorithms.transport import select_transport
from smartsom.dispatch import DecisionContext, SemanticAction


class SPTPolicy:
    def select_action(self, context: DecisionContext) -> SemanticAction:
        if not context.candidates:
            return select_transport(context)
        return min(
            context.candidates,
            key=lambda candidate: (
                candidate.nominal_ticks,
                candidate.action.operation_id,
                candidate.action.processing_mode_id,
            ),
        ).action
