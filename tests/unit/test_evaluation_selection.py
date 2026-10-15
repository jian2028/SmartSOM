"""Unselected science is explicit; missing or corrupt selected models still fail."""

import json
from types import SimpleNamespace

import pytest

from smartsom.config.experiment import LastDiagnosticOptions, NoEligibleBestOptions
from smartsom.experiments.evaluation_selection import unselected_outcome


def setup(tmp_path, diagnostic=False):
    options = NoEligibleBestOptions(
        last_diagnostic=LastDiagnosticOptions() if diagnostic else None
    )
    prepared = SimpleNamespace(
        config=SimpleNamespace(evaluation=SimpleNamespace(no_eligible_best=options))
    )
    record = {
        "status": "completed",
        "selection_outcome": {
            "status": "no_eligible_best",
            "best_update": None,
            "validation_rounds": 3,
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return prepared, record


def test_skip_primary_and_explicit_optional_last_diagnostic(tmp_path):
    prepared, _ = setup(tmp_path)
    outcome = unselected_outcome(prepared, tmp_path, "best")
    assert outcome["final"] is None and outcome["selected_checkpoint"] is None
    assert outcome["diagnostic_label"] is None and not outcome["ranking_eligible"]
    prepared, _ = setup(tmp_path, diagnostic=True)
    assert (
        unselected_outcome(prepared, tmp_path, "best")["diagnostic_label"]
        == "FINAL-LAST-DIAGNOSTIC"
    )
    assert unselected_outcome(prepared, tmp_path, "last") is None


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {
            "selection_outcome": {
                "status": "no_eligible_best",
                "best_update": 8,
                "validation_rounds": 3,
            }
        },
        {
            "selection_outcome": {
                "status": "no_eligible_best",
                "best_update": None,
                "validation_rounds": 0,
            }
        },
    ],
)
def test_invalid_authoritative_outcome_rejected(tmp_path, change):
    prepared, record = setup(tmp_path)
    record.update(change)
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        unselected_outcome(prepared, tmp_path, "best")


def test_selected_missing_alias_is_not_no_eligible_best(tmp_path):
    prepared, record = setup(tmp_path)
    record["selection_outcome"] = {
        "status": "selected",
        "best_update": 8,
        "validation_rounds": 3,
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    assert unselected_outcome(prepared, tmp_path, "best") is None


def test_contradictory_best_alias_rejected(tmp_path):
    prepared, _ = setup(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/best.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        unselected_outcome(prepared, tmp_path, "best")
