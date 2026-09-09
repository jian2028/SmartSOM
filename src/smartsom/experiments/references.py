"""Retention references for new update checkpoints, outside immutable payloads."""

import json
from pathlib import Path


def protect_model_reference(inference, reference):
    """Protect only complete update-format checkpoints; leave legacy bytes intact."""
    inference = Path(inference).resolve()
    update = inference.parent if inference.name == "inference" else inference
    manifest = update / "manifest.json"
    if not manifest.is_file():
        return False
    from smartsom.experiments.training_lifecycle import SCHEMA, protect_checkpoint

    if json.loads(manifest.read_text()).get("schema") != SCHEMA:
        return False
    protect_checkpoint(update, reference)
    return True
