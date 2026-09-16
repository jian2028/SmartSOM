"""Public workflow roots and terminal states remain authoritative."""

import json

import pytest

from smartsom import api
from smartsom.experiments.catalog import read_run
from smartsom.experiments.evidence import write_json


def test_simulation_returns_the_authoritative_experiment_directory(tmp_path):
    config = api.load_preset("test")
    config.output.root = str(tmp_path)
    result = api.run(config)
    assert result.evidence_dir == result.run_dir
    metadata = json.loads((result.run_dir / "run.json").read_text())
    assert metadata["experiment"]["config"]["seed"] == config.seed
    assert (
        metadata["experiment"]["scientific_sha256"]
        == api.prepare(config, training=False).scientific_sha256
    )
    assert {path.name for path in result.run_dir.iterdir()} == {
        "run.json",
        "trace.jsonl",
    }
    assert (result.evidence_dir / "trace.jsonl").is_file()
    assert read_run(result.run_dir).status == "completed"


@pytest.mark.parametrize(
    "failure,status",
    [
        (RuntimeError("evaluation failed"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_combined_evaluation_failure_cannot_leave_training_success_as_overall_status(
    tmp_path, monkeypatch, failure, status
):
    root = tmp_path / "experiment"
    root.mkdir()
    write_json(
        root / "run.json",
        {
            "status": "completed",
            "kind": "training",
            "paths": {"training": "evidence/training"},
        },
    )
    trained = api.TrainingResult(
        root, root / "evidence/training", None, None, 4096, 16, 16, "completed"
    )
    monkeypatch.setattr(api, "train", lambda *a, **k: trained)

    def evaluate(*a, **k):
        current = json.loads((root / "run.json").read_text())
        assert current["status"] == "running"
        assert current["kind"] == "train_evaluate"
        failure.run_dir = root / "evaluation/failed-attempt"
        raise failure

    monkeypatch.setattr(api, "evaluate", evaluate)
    with pytest.raises(type(failure)) as raised:
        api.train_evaluate(api.load_preset("marl_micro"))
    assert raised.value is failure
    record = json.loads((root / "run.json").read_text())
    assert record["training_status"] == "completed"
    assert record["evaluation_status"] == record["status"] == status
    assert record["paths"]["evaluation"] == "evaluation/failed-attempt"
