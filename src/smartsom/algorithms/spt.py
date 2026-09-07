"""Shortest processing time among the current legal dispatch candidates."""

from smartsom.dispatch import DecisionContext, Dispatch


class SPTPolicy:
    def select_action(self, context: DecisionContext) -> Dispatch:
        return min(
            context.candidates,
            key=lambda candidate: (
                candidate.nominal_ticks,
                candidate.action.operation_id,
                candidate.action.processing_mode_id,
            ),
        ).action
