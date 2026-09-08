"""Fixed quality-label selection, independent of probabilities and hidden outcomes."""

from smartsom.dispatch import DecisionContext


def quality_candidates(context: DecisionContext, mode: str | None):
    if mode is None:
        return context.candidates
    groups = {}
    for row in context.quality_modes:
        groups.setdefault((row.operation_id, row.base_processing_mode_id), set()).add(
            row.quality_mode_id
        )
    if not groups or any(mode not in labels for labels in groups.values()):
        # Empty arrival notices may precede the first revealed job.
        if not context.jobs and not context.candidates:
            return ()
        raise ValueError(f"fixed quality mode {mode!r} is unavailable")
    allowed = {
        (row.operation_id, row.processing_mode_id)
        for row in context.quality_modes
        if row.quality_mode_id == mode
    }
    return tuple(
        c
        for c in context.candidates
        if (c.action.operation_id, c.action.processing_mode_id) in allowed
    )
