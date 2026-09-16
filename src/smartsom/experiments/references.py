"""Retention references for new update checkpoints, outside immutable payloads."""

import json
from pathlib import Path


def protect_model_reference(inference, reference):
    """Protect only complete update-format checkpoints; leave legacy bytes intact."""
    inference = Path(inference).resolve()
    update = inference.parent if inference.name == "inference" else inference
    if (update / "update.json").is_file():
        from smartsom.config.codec import digest
        from smartsom.experiments.evidence import write_json
        from smartsom.experiments.production_training import verify_checkpoint

        verify_checkpoint(update)
        (update.parent / "references" / update.name).mkdir(parents=True, exist_ok=True)
        # References are mutable retention metadata beside the immutable payload.
        write_json(
            update.parent
            / "references"
            / update.name
            / f"{digest(str(reference))}.json",
            {"reference": str(reference)},
        )
        return True
    manifest = update / "manifest.json"
    if not manifest.is_file():
        return False
    from smartsom.experiments.training_lifecycle import SCHEMA, protect_checkpoint

    if json.loads(manifest.read_text()).get("schema") != SCHEMA:
        return False
    protect_checkpoint(update, reference)
    return True
