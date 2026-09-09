"""Generated reports protect new saves without modifying historical payloads."""

import json

from smartsom.experiments.references import protect_model_reference
from smartsom.experiments.training_lifecycle import SCHEMA, inspect_resume_checkpoint


def test_retention_reference_is_outside_signed_checkpoint_members(tmp_path):
    inference = tmp_path / "update-1/inference"
    inference.mkdir(parents=True)
    manifest = inference.parent / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": SCHEMA, "status": "complete", "files": {}})
    )
    original = manifest.read_bytes()
    assert protect_model_reference(inference, "reports/evaluation.json")
    assert manifest.read_bytes() == original
    assert inspect_resume_checkpoint(inference.parent)["files"] == {}
    assert len(list((inference.parent / "references").glob("*.json"))) == 1
    assert protect_model_reference(inference, "reports/evaluation.json")
    assert len(list((inference.parent / "references").glob("*.json"))) == 1


def test_legacy_checkpoint_bytes_remain_unchanged(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_bytes(b"legacy model")
    assert not protect_model_reference(checkpoint, "report.html")
    assert list(checkpoint.iterdir()) == [checkpoint / "checkpoint.json"]
    assert (checkpoint / "checkpoint.json").read_bytes() == b"legacy model"
