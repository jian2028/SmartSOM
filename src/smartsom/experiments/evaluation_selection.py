"""Explicit unselected outcomes never disguise a missing selected artifact."""

import json
from pathlib import Path

from smartsom._filesystem import native_path


def unselected_outcome(prepared, source, selection):
    options = getattr(prepared.config.evaluation, "no_eligible_best", None)
    if source is None or selection != "best" or options is None:
        return None
    record = json.loads(
        native_path(Path(source) / "run.json").read_text(encoding="utf-8")
    )
    outcome = record.get("selection_outcome")
    if not outcome or outcome.get("status") != "no_eligible_best":
        return None
    from smartsom.experiments.checkpoint_retention import manifest

    retained = manifest(source)
    if (
        record.get("status") not in {"completed", "early_stopped"}
        or outcome.get("best_update") is not None
        or outcome.get("validation_rounds", 0) < 1
        or native_path(Path(source) / "checkpoints/best.json").exists()
        or (
            retained is not None
            and (
                retained["selected_best"] is not None
                or retained["best_update"] is not None
            )
        )
    ):
        raise ValueError("inconsistent no-eligible-best training outcome")
    return {
        "status": "no_eligible_best",
        "selected_checkpoint": None,
        "final": None,
        "diagnostic_label": (
            options.last_diagnostic.label
            if options.last_diagnostic and options.last_diagnostic.enabled
            else None
        ),
        "ranking_eligible": False,
    }
