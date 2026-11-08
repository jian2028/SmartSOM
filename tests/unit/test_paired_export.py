"""Explicit paired evidence: hand-counted windows and unobservable cases."""

import copy
import hashlib
import json

import pytest
import yaml

from smartsom.experiments.cli import main
from smartsom.telemetry.paired_export import export


def case(times):
    return {
        "completed": len(times),
        "total_orders": 10,
        "evaluation_identity": {
            "checkpoint_sha256": "model",
            "method": "dqn",
            "learning": True,
            "selection": "argmax",
            "policy_seed": 42,
            "source_sha256": "source",
            "runtime_sha256": "runtime",
            "algorithm_sha256": "algorithm",
        },
        "research_metrics": {
            "schema": "smartsom.research-metrics/v1",
            "level": "detailed",
            "start_tick": 0,
            "end_tick": 100,
            "fingerprints": {"exogenous": "same"},
            "orders": [{"completion_tick": t, "final_pass": True} for t in times],
            "outage_realization": [{"machine_id": "m", "start": 0, "end": 10}],
            "events": [
                {"kind": "breakdown", "tick": 0, "machine": "m"},
                {"kind": "repair", "tick": 10, "machine": "m"},
            ],
        },
    }


def write_case(tmp_path, name, row):
    path = tmp_path / name
    path.write_text(json.dumps(row) + "\n")
    return {
        "path": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "line": 1,
    }


def config(tmp_path):
    nominal = case([10, 20, 30, 40, 90])
    nominal["research_metrics"]["outage_realization"] = []
    nominal["research_metrics"]["events"] = []
    document = {
        "schema": "smartsom.paired-export/v1",
        "nominal": write_case(tmp_path, "normal.jsonl", nominal),
        "perturbed": write_case(tmp_path, "fault.jsonl", case([10, 25, 40, 90])),
        "windows": [{"start": 10, "end": 40}],
        "overall_window": {"start": 10, "end": 40},
        "fault": {"machine_id": "m", "fault_tick": 0, "repair_tick": 10},
        "recovery": {
            "semantics": "paired_repair_anchored",
            "duration_from": "repair",
            "version": "fixture-v1",
            "reference": "matched-model",
            "window_ticks": 10,
            "consecutive_windows": 2,
            "fraction": 0.95,
        },
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, document


def same_run_config(tmp_path, times):
    path, document = config(tmp_path)
    row = case(times)
    row["research_metrics"]["outage_realization"] = [
        {"machine_id": "m", "start": 30, "end": 40}
    ]
    row["research_metrics"]["events"] = [
        {"kind": "breakdown", "tick": 30, "machine": "m"},
        {"kind": "repair", "tick": 40, "machine": "m"},
    ]
    document.pop("nominal")
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    document["fault"] = {"machine_id": "m", "fault_tick": 30, "repair_tick": 40}
    document["recovery"].update(
        semantics="same_run_pre_fault",
        baseline={
            "window_ticks": 10,
            "baseline_windows": 1,
            "minimum_window_deliveries": 1,
            "warmup_end_tick": 0,
        },
    )
    path.write_text(yaml.safe_dump(document))
    return path, document, row


def configured_same_run(tmp_path, times, mode):
    path, document, row = same_run_config(tmp_path, times)
    row["research_metrics"]["sample_interval_ticks"] = 10
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    document["experiment"] = "analysis-experiment.yaml"
    document["recovery"] = {
        "semantics": "configured_same_run",
        "version": "fixture",
        "reference": "same-run",
    }
    chosen = {
        "mode": mode,
        "warmup_end_tick": 0,
        "baseline_windows": 1,
        "fraction": 0.95,
        "consecutive_windows": 2,
        "duration_from": "repair",
        "maximum_fault_events": 10,
    }
    if mode == "auto":
        chosen.pop("baseline_windows")
        chosen["auto"] = {
            "plateau_windows": 2,
            "preset_windows": 1,
            "maximum_cv": 0,
            "maximum_relative_trend": 0,
            "maximum_upward_trend": 0,
            "search_order": "latest",
        }
    experiment = {
        "run": {"metric_interval": 10},
        "evaluation": {
            "window_analysis": {
                "throughput_width_ticks": 10,
                "throughput_step_ticks": 10,
                "recovery_window_ticks": 10,
                "same_run": chosen,
            }
        },
    }
    (tmp_path / "analysis-experiment.yaml").write_text(yaml.safe_dump(experiment))
    path.write_text(yaml.safe_dump(document))
    return path, document, row, experiment


def test_configured_manual_uses_n_sampling_windows(tmp_path):
    path, _, _, experiment = configured_same_run(
        tmp_path, [5, 15, 25, 45, 55, 95], "manual"
    )
    chosen = experiment["evaluation"]["window_analysis"]["same_run"]
    chosen["baseline_windows"] = 2
    (tmp_path / "analysis-experiment.yaml").write_text(yaml.safe_dump(experiment))
    report = export(path)
    result = report["recovery"]
    assert result["period_selection"]["pre_fault"]["selected"][0]["start_tick"] == 10
    assert result["period_selection"]["pre_fault"]["mean_throughput"] == pytest.approx(
        0.1
    )
    assert result["status"] == "threshold_confirmed"
    assert result["confirmation_tick"] == 60


def test_configured_auto_reports_each_missing_plateau_and_falls_back(tmp_path):
    path, _, _, _ = configured_same_run(
        tmp_path, [5, 5, 5, 15, 15, 25, 45, 55, 95], "auto"
    )
    result = export(path)["recovery"]
    selection = result["period_selection"]
    assert selection["plateau_not_found"] == "both"
    assert selection["pre_fault"]["status"] == "preset"
    assert selection["fault_period"]["status"] == "preset"
    assert selection["pre_fault"]["selected"][0]["start_tick"] == 20
    assert result["robustness"]["meaning"] == "preset_window_ratio_not_steady_state"
    assert result["status"] == "threshold_confirmed"


def test_auto_preset_count_is_required_and_selects_exact_number(tmp_path):
    path, _, _, experiment = configured_same_run(
        tmp_path, [5, 5, 5, 15, 15, 25, 45, 55, 95], "auto"
    )
    auto = experiment["evaluation"]["window_analysis"]["same_run"]["auto"]
    del auto["preset_windows"]
    target = tmp_path / "analysis-experiment.yaml"
    target.write_text(yaml.safe_dump(experiment))
    with pytest.raises(ValueError, match="preset_windows"):
        export(path)
    auto["preset_windows"] = 2
    target.write_text(yaml.safe_dump(experiment))
    selected = export(path)["recovery"]["period_selection"]["pre_fault"]
    assert selected["status"] == "preset"
    assert [(w["start_tick"], w["end_tick"]) for w in selected["selected"]] == [
        (10, 20),
        (20, 30),
    ]


def test_configured_auto_reports_fault_only_missing_and_zero_baseline(tmp_path):
    path, _, _, _ = configured_same_run(tmp_path, [5, 15, 25, 45, 55, 95], "auto")
    result = export(path)["recovery"]
    assert result["period_selection"]["plateau_not_found"] == "fault_period"
    assert result["period_selection"]["pre_fault"]["status"] == "plateau"
    path, _, _, _ = configured_same_run(tmp_path, [45, 55, 95], "auto")
    result = export(path)["recovery"]
    assert result["status"] == "not_applicable"
    assert result["reason"] == "zero_baseline"


def test_forced_aggregate_limit_uses_fallback_and_marks_detection_unknown(tmp_path):
    from smartsom.config.evaluation_protocol import SameRunAnalysis
    from smartsom.telemetry.paired_export import Fault, _configured_recovery

    _, _, row, experiment = configured_same_run(
        tmp_path, [5, 15, 25, 45, 55, 95], "auto"
    )
    settings = SameRunAnalysis.model_validate(
        experiment["evaluation"]["window_analysis"]["same_run"]
    )
    result = _configured_recovery(
        row,
        Fault(machine_id="m", fault_tick=30, repair_tick=40),
        settings,
        10,
        10,
        force_fallback=True,
    )
    assert result["period_selection"]["plateau_not_found"] == "unknown"
    assert (
        result["period_selection"]["pre_fault"]["reason"]
        == "aggregate_plateau_work_limit"
    )
    assert result["robustness"]["status"] == "preset"
    assert result["status"] == "threshold_confirmed"


def test_episode_limits_filter_future_events_and_preserve_running_work(
    tmp_path, monkeypatch
):
    import smartsom.telemetry.paired_export as module
    from smartsom.config.evaluation_protocol import WindowAnalysis

    _, _, row, experiment = configured_same_run(
        tmp_path, [5, 15, 25, 45, 55, 95], "auto"
    )
    cfg = experiment["evaluation"]["window_analysis"]
    cfg["same_run"]["maximum_fault_events"] = 2
    cfg["same_run"]["auto"]["plateau_windows"] = 1000
    analysis = WindowAnalysis.model_validate(cfg)
    calls = []

    def fake(report, fault, settings, sample, width, *, force_fallback=False):
        calls.append((fault.machine_id, force_fallback))
        return {"machine_id": fault.machine_id}

    monkeypatch.setattr(module, "_configured_recovery", fake)
    row["research_metrics"]["outage_realization"].append(
        {"machine_id": "future", "start": 101, "end": 110}
    )
    result = module.same_run_episode(row, analysis, 10)
    assert result["status"] == "observed" and calls == [("m", False)]
    row["research_metrics"]["outage_realization"].extend(
        [
            {"machine_id": "x", "start": 20, "end": 21},
            {"machine_id": "y", "start": 25, "end": 26},
        ]
    )
    result = module.same_run_episode(row, analysis, 10)
    assert (
        result["reason"] == "fault_event_limit" and result["observed_fault_count"] == 3
    )
    row["research_metrics"]["outage_realization"] = [
        {"machine_id": "m", "start": 30, "end": 40}
    ]
    row["research_metrics"]["end_tick"] = 999990
    result = module.same_run_episode(row, analysis, 10)
    assert result["status"] == "observed" and calls[-1] == ("m", True)
    row["research_metrics"]["end_tick"] = 1000000
    result = module.same_run_episode(row, analysis, 10)
    assert (
        result["reason"] == "aggregate_window_limit"
        and result["observed_fault_count"] == 1
    )


def test_configured_auto_before_only_fallback_has_combined_fallback_status(tmp_path):
    path, document, row, _ = configured_same_run(
        tmp_path, [5, 5, 5, 15, 15, 25, 35, 45, 55, 65, 95], "auto"
    )
    row["total_orders"] = 20
    row["research_metrics"]["outage_realization"] = [
        {"machine_id": "m", "start": 30, "end": 50}
    ]
    row["research_metrics"]["events"] = [
        {"kind": "breakdown", "tick": 30, "machine": "m"},
        {"kind": "repair", "tick": 50, "machine": "m"},
    ]
    document["fault"]["repair_tick"] = 50
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    assert result["period_selection"]["plateau_not_found"] == "before_fault"
    assert result["robustness"]["status"] == "preset"
    assert result["robustness"]["baseline_source"] == "preset"
    assert result["robustness"]["fault_source"] == "plateau"
    assert (result["machine_id"], result["fault_tick"], result["repair_tick"]) == (
        "m",
        30,
        50,
    )


def test_auto_skips_zero_baseline_plateau_and_finds_later_positive(tmp_path):
    path, document, row, experiment = configured_same_run(
        tmp_path, [25, 35, 45, 55, 65, 75, 95], "auto"
    )
    experiment["evaluation"]["window_analysis"]["same_run"]["auto"]["search_order"] = (
        "earliest"
    )
    (tmp_path / "analysis-experiment.yaml").write_text(yaml.safe_dump(experiment))
    row["research_metrics"]["outage_realization"] = [
        {"machine_id": "m", "start": 40, "end": 60}
    ]
    row["research_metrics"]["events"] = [
        {"kind": "breakdown", "tick": 40, "machine": "m"},
        {"kind": "repair", "tick": 60, "machine": "m"},
    ]
    document["fault"] = {"machine_id": "m", "fault_tick": 40, "repair_tick": 60}
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    assert result["period_selection"]["pre_fault"]["status"] == "plateau"
    assert result["period_selection"]["pre_fault"]["selected"][0]["start_tick"] == 20
    assert result["period_selection"]["pre_fault"]["mean_throughput"] == pytest.approx(
        0.1
    )


def test_auto_zero_fault_platform_has_undefined_relative_statistics(tmp_path):
    path, document, row, _ = configured_same_run(
        tmp_path, [5, 15, 25, 55, 65, 95], "auto"
    )
    row["research_metrics"]["outage_realization"] = [
        {"machine_id": "m", "start": 30, "end": 50}
    ]
    row["research_metrics"]["events"] = [
        {"kind": "breakdown", "tick": 30, "machine": "m"},
        {"kind": "repair", "tick": 50, "machine": "m"},
    ]
    document["fault"]["repair_tick"] = 50
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    plateau = result["period_selection"]["fault_period"]
    assert plateau["zero_output_constant"] is True
    assert plateau["cv"] is None and plateau["relative_trend_per_window"] is None
    assert result["robustness"]["ratio"] == 0


def test_plateau_limit_falls_back_and_insufficient_windows_are_unavailable():
    from types import SimpleNamespace

    from smartsom.telemetry.plateau_analysis import _select

    spec = SimpleNamespace(
        plateau_windows=1000, preset_windows=1, search_order="latest"
    )
    point = {"throughput": 1}
    result = _select([point] * 3000, spec)
    assert result["status"] == "preset" and result["reason"] == "plateau_work_limit"
    spec.preset_windows = 2
    spec.plateau_windows = 3
    result = _select([point], spec)
    assert result["status"] == "unavailable"
    assert result["reason"] == "plateau_not_found_and_insufficient_preset_windows"


def test_same_run_one_window_recovery_and_fault_average(tmp_path):
    path, _, _ = same_run_config(tmp_path, [5, 15, 25, 45, 55, 95])
    exported = export(path)
    result = exported["recovery"]
    assert result["status"] == "threshold_confirmed"
    assert result["confirmation_tick"] == 60
    assert result["adaptability"] == pytest.approx(1 / 20)
    assert result["baseline"]["stability"] == "unverified_single_window"
    assert result["baseline"]["reason"] is None
    assert result["fault_period_mean_ratio"]["ratio"] == 0
    assert result["fault_period_mean_ratio"]["meaning"].endswith("not_steady_state")
    assert exported["overall"]["perturbed"]["completed"] == 2


def test_same_run_multiple_windows_require_and_apply_stability_limits(tmp_path):
    path, document, _ = same_run_config(tmp_path, [5, 15, 25, 45, 55, 95])
    baseline = document["recovery"]["baseline"]
    baseline["baseline_windows"] = 2
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="require explicit CV"):
        export(path)
    baseline.update(maximum_cv=0.1, maximum_relative_trend=0.1)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    assert result["status"] == "threshold_confirmed"
    assert result["baseline"]["stability"] == "passes_configured_checks"
    assert result["baseline"]["mean_throughput"] == pytest.approx(0.1)


def test_same_run_unstable_baseline_and_new_fault_at_repair(tmp_path):
    path, document, row = same_run_config(tmp_path, [5, 15, 15, 25, 45, 55, 95])
    baseline = document["recovery"]["baseline"]
    baseline.update(baseline_windows=2, maximum_cv=0.1, maximum_relative_trend=0.1)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    assert result["status"] == "insufficient_baseline"
    assert result["baseline"]["stability"] == "not_established"

    baseline["baseline_windows"] = 1
    baseline.pop("maximum_cv")
    baseline.pop("maximum_relative_trend")
    row["research_metrics"]["outage_realization"].append(
        {"machine_id": "other", "start": 40, "end": 50}
    )
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)["recovery"]
    assert result["status"] == "censored"
    assert result["reason"] == "new_fault"


def test_same_run_consecutive_windows_reset_after_drop(tmp_path):
    path, _, _ = same_run_config(tmp_path, [5, 15, 25, 45, 65, 75, 95])
    result = export(path)["recovery"]
    assert result["status"] == "threshold_confirmed"
    assert result["confirmation_tick"] == 80
    assert result["adaptability"] == pytest.approx(1 / 40)


def test_same_run_missing_repair_after_censor_and_invalid_overall(tmp_path):
    path, document, row = same_run_config(tmp_path, [5, 15, 25])
    row["research_metrics"]["end_tick"] = 35
    row["research_metrics"]["events"].pop()
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    assert export(path)["recovery"]["status"] == "censored"
    assert export(path)["recovery"]["reason"] == "observation_too_short"
    document["overall_window"] = {"start": 40, "end": 20}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="overall window"):
        export(path)


@pytest.mark.parametrize(
    "change, expected",
    [
        ("warmup", "baseline_overlaps_configured_warmup"),
        ("zero", "zero_baseline"),
        ("overlap", "overlapping_fault"),
        ("no_repair", "event_boundary_missing"),
        ("not_recovered", "observation_ended"),
    ],
)
def test_same_run_rejects_unreliable_or_censors_unfinished(tmp_path, change, expected):
    path, document, row = same_run_config(tmp_path, [5, 15, 25, 45, 55, 95])
    if change == "warmup":
        document["recovery"]["baseline"]["warmup_end_tick"] = 21
    elif change == "zero":
        row["research_metrics"]["orders"] = [
            {"completion_tick": t, "final_pass": True} for t in [5, 15, 45, 55, 95]
        ]
    elif change == "overlap":
        row["research_metrics"]["outage_realization"].append(
            {"machine_id": "other", "start": 24, "end": 35}
        )
    elif change == "no_repair":
        row["research_metrics"]["events"].pop()
    else:
        row["research_metrics"]["orders"] = [
            {"completion_tick": t, "final_pass": True} for t in [5, 15, 25, None]
        ]
    document["perturbed"] = write_case(tmp_path, "same.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    assert export(path)["recovery"]["reason"] == expected


def test_cli_exports_hand_calculated_windows_and_refuses_overwrite(tmp_path):
    path, _ = config(tmp_path)
    output = tmp_path / "evidence.json"
    assert main(["paired-export", str(path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    window = result["windows"][0]
    assert window["nominal"]["completed"] == 3
    assert window["perturbed"]["completed"] == 2
    assert window["comparison"]["throughput_ratio"] == pytest.approx(2 / 3)
    assert result["overall"]["comparison"]["throughput_ratio"] == pytest.approx(2 / 3)
    assert result["settings"]["recovery"]["window_ticks"] == 10
    assert result["cases"]["perturbed"]["research_metrics"]["events"][0]["tick"] == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["paired-export", str(path), "--output", str(output)])
    assert output.read_bytes() == before


def test_export_reuses_counts_and_matches_public_robustness(tmp_path, monkeypatch):
    import smartsom.telemetry.paired_export as module
    import smartsom.telemetry.research_comparison as comparison

    path, _ = config(tmp_path)
    nominal = json.loads((tmp_path / "normal.jsonl").read_text())
    perturbed = json.loads((tmp_path / "fault.jsonl").read_text())
    expected = comparison.robustness(nominal, perturbed, 10, 40)
    calls = []
    original = comparison.fixed_window

    def count(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "fixed_window", count)
    monkeypatch.setattr(comparison, "fixed_window", count)
    # Isolate export's comparison windows from recovery's separate scan.
    monkeypatch.setattr(
        module, "recovery", lambda *args, **kwargs: {"status": "censored"}
    )
    result = export(path)
    assert result["windows"][0]["comparison"] == expected
    assert result["overall"]["comparison"] == expected
    assert len(calls) == 4  # Two cases for each of the two requested windows.


def test_changed_files_or_unpaired_identities_are_rejected(tmp_path):
    path, document = config(tmp_path)
    changed = case([20])
    (tmp_path / "fault.jsonl").write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        export(path)
    changed["research_metrics"]["fingerprints"]["exogenous"] = "different"
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", changed)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="differ"):
        export(path)


def test_zero_reference_and_identity_error_precede_missing_fault(tmp_path):
    path, document = config(tmp_path)
    nominal = case([5, 6])
    nominal["research_metrics"]["outage_realization"] = []
    nominal["research_metrics"]["events"] = []
    document["nominal"] = write_case(tmp_path, "normal.jsonl", nominal)
    path.write_text(yaml.safe_dump(document))
    comparison = export(path)["windows"][0]["comparison"]
    assert comparison["status"] == "not_applicable"
    assert comparison["reason"] == "zero_reference"
    assert comparison["throughput_ratio"] is None

    perturbed = case([10, 25])
    perturbed["evaluation_identity"]["source_sha256"] = "other"
    perturbed["research_metrics"]["events"] = []
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", perturbed)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="same policy"):
        export(path)


def test_truncated_and_ended_runs_do_not_invent_zero_tail(tmp_path):
    path, document = config(tmp_path)
    document["windows"] = [{"start": 90, "end": 110}]
    finished = case([10, 20])
    finished["total_orders"] = 2
    nominal = copy.deepcopy(finished)
    nominal["research_metrics"]["outage_realization"] = []
    nominal["research_metrics"]["events"] = []
    document["nominal"] = write_case(tmp_path, "normal.jsonl", nominal)
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", copy.deepcopy(finished))
    path.write_text(yaml.safe_dump(document))
    result = export(path)
    assert result["windows"][0]["comparison"]["status"] == "censored"


def test_missing_protocol_and_incomplete_fault_are_not_implicit(tmp_path):
    path, document = config(tmp_path)
    del document["recovery"]["fraction"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        export(path)
    document["recovery"]["fraction"] = 0.95
    altered = case([10, 20])
    altered["research_metrics"]["events"] = []
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", altered)
    path.write_text(yaml.safe_dump(document))
    result = export(path)
    assert result["recovery"]["status"] == "not_recorded"
    assert result["overall"]["comparison"]["reason"] == "selected_fault_event_missing"


def test_identical_or_extra_faults_cannot_masquerade_as_pair(tmp_path):
    path, document = config(tmp_path)
    same = case([10, 25, 40, 90])
    same["research_metrics"]["events"] = []
    document["nominal"] = write_case(tmp_path, "normal.jsonl", same)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="exactly the selected fault"):
        export(path)
    same["research_metrics"]["outage_realization"] = []
    document["nominal"] = write_case(tmp_path, "normal.jsonl", same)
    extra = case([10, 25, 40, 90])
    extra["research_metrics"]["outage_realization"].append(
        {"machine_id": "other", "start": 20, "end": 30}
    )
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", extra)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="exactly the selected fault"):
        export(path)


def test_missing_identity_is_not_hidden_by_censored_window(tmp_path):
    path, document = config(tmp_path)
    row = case([10])
    del row["evaluation_identity"]
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", row)
    document["windows"] = [{"start": 90, "end": 110}]
    path.write_text(yaml.safe_dump(document))
    comparison = export(path)["windows"][0]["comparison"]
    assert comparison["status"] == "not_recorded"
    assert comparison["reason"] == "evaluation_identity_missing"


def test_empty_identity_and_nominal_fault_are_not_valid_pair(tmp_path):
    path, document = config(tmp_path)
    for side in ("nominal", "perturbed"):
        row = json.loads((tmp_path / document[side]["path"]).read_text())
        row["evaluation_identity"]["source_sha256"] = None
        document[side] = write_case(tmp_path, document[side]["path"], row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)
    assert result["overall"]["comparison"]["status"] == "not_recorded"
    assert result["recovery"]["status"] == "not_recorded"
    normal = json.loads((tmp_path / "normal.jsonl").read_text())
    normal["research_metrics"]["events"] = [
        {"kind": "breakdown", "tick": 0, "machine": "m"}
    ]
    document["nominal"] = write_case(tmp_path, "normal.jsonl", normal)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="nominal case contains"):
        export(path)
    del normal["research_metrics"]["outage_realization"]
    document["nominal"] = write_case(tmp_path, "normal.jsonl", normal)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="nominal case contains"):
        export(path)


def test_nonlearning_policy_can_have_null_checkpoint(tmp_path):
    path, document = config(tmp_path)
    for side in ("nominal", "perturbed"):
        row = json.loads((tmp_path / document[side]["path"]).read_text())
        row["evaluation_identity"].update(learning=False, checkpoint_sha256=None)
        document[side] = write_case(tmp_path, document[side]["path"], row)
    path.write_text(yaml.safe_dump(document))
    assert export(path)["overall"]["comparison"]["status"] == "observed"


def test_unserializable_output_does_not_leave_partial_file(tmp_path):
    from smartsom.telemetry.paired_export import write_export

    target = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        write_export(target, {"value": float("nan")})
    assert not target.exists()


def test_selected_origin_and_machine_utilization_are_explicit(tmp_path):
    path, document = config(tmp_path)
    row = case([10, 20, 30, 40, 90])
    row.update(total_orders=10, completed=5, failed=1, tick=100, observed_overdue=120)
    row["research_metrics"]["summary"] = {
        "on_time_coverage": 0.3,
        "machine_state_ticks": {"processing": 60, "idle": 30, "blocked": 0, "down": 10},
        "machine_state_fractions": {
            "processing": 0.6,
            "idle": 0.3,
            "blocked": 0,
            "down": 0.1,
        },
        "available_processing_fraction": 2 / 3,
    }
    normal = copy.deepcopy(row)
    normal["research_metrics"]["outage_realization"] = []
    normal["research_metrics"]["events"] = []
    document["nominal"] = write_case(tmp_path, "normal.jsonl", normal)
    document["perturbed"] = write_case(tmp_path, "fault.jsonl", row)
    path.write_text(yaml.safe_dump(document))
    result = export(path)
    out = result["outcomes"]["perturbed"]
    assert out["performance"]["delivery_fraction"] == 0.5
    assert out["performance"]["qualified_coverage"] == 0.4
    assert out["performance"]["mean_tardiness"] is None
    assert out["observed_tardiness_per_order"] == 12
    assert out["resources"]["machine_state_fractions"]["processing"] == 0.6
    assert out["resources"]["available_processing_fraction"] == pytest.approx(2 / 3)
    assert result["selected_adaptability"]["value_per_tick"] == 0.05
    assert result["recovery"]["adaptability_basis"] == "repair_to_confirmation_ticks"
    assert (
        result["experiment_source_role"]
        == "analysis_window_configuration_not_case_provenance"
    )
    document["recovery"]["duration_from"] = "fault"
    path.write_text(yaml.safe_dump(document))
    fault_result = export(path)
    assert fault_result["selected_adaptability"]["value_per_tick"] == pytest.approx(
        1 / 30
    )
    assert (
        fault_result["recovery"]["adaptability_basis"] == "repair_to_confirmation_ticks"
    )


def test_experiment_controls_record_and_calculation_intervals(tmp_path):
    path, document = config(tmp_path)
    for side in ("nominal", "perturbed"):
        row = json.loads((tmp_path / document[side]["path"]).read_text())
        row["research_metrics"]["sample_interval_ticks"] = 10
        document[side] = write_case(tmp_path, document[side]["path"], row)
    experiment = {
        "run": {"metric_interval": 10},
        "evaluation": {
            "window_analysis": {
                "throughput_width_ticks": 100,
                "throughput_step_ticks": 100,
                "recovery_window_ticks": 10,
            }
        },
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    document["experiment"] = "experiment.yaml"
    del document["windows"]
    document["window_grid"] = {"start": 0, "end": 100}
    del document["recovery"]["window_ticks"]
    path.write_text(yaml.safe_dump(document))
    result = export(path)
    assert result["outcomes"]["nominal"]["recorded_sample_interval_ticks"] == 10
    assert result["windows"][0]["start"] == 0 and result["windows"][0]["end"] == 100
    assert result["windows"][0]["nominal"]["completed"] == 5
    assert result["resolved_recovery_window_ticks"] == 10
    document["window_grid"]["width"] = 50
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="differs from experiment"):
        export(path)


def test_window_series_exact_endpoints_and_missing_detail():
    from smartsom.config.evaluation_protocol import WindowAnalysis
    from smartsom.telemetry.window_series import throughput_series

    options = WindowAnalysis(
        throughput_width_ticks=10, throughput_step_ticks=10, recovery_window_ticks=100
    )
    row = case([0, 10, 20, 25])
    row["research_metrics"]["end_tick"] = 25
    result = throughput_series(row, options)
    assert [(w["start"], w["end"], w["completed"]) for w in result["windows"]] == [
        (0, 10, 1),
        (10, 20, 1),
    ]
    assert result["partial_final_window"] == "omitted"
    assert throughput_series({}, options)["status"] == "not_recorded"
    large = case([1])
    large["research_metrics"]["end_tick"] = 10001
    dense = WindowAnalysis(
        throughput_width_ticks=1, throughput_step_ticks=1, recovery_window_ticks=100
    )
    assert throughput_series(large, dense)["reason"] == "window_count_exceeds_10000"
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            WindowAnalysis(
                throughput_width_ticks=invalid,
                throughput_step_ticks=10,
                recovery_window_ticks=100,
            )


def test_controlled_recovery_window_conflicts_are_rejected():
    from smartsom.config.evaluation_protocol import EvaluationProtocol

    candidate = {
        "name": "x",
        "load": "normal",
        "orders": 8,
        "initial_orders": 4,
        "mean_interarrival": 10,
        "horizon_ticks": 1000,
        "window_ticks": 50,
        "baseline_windows": 3,
        "minimum_window_deliveries": 1,
        "maximum_cv": 1.0,
        "maximum_relative_trend": 1.0,
        "faults": [{"machine_id": "m", "start_tick": 200, "duration_ticks": 20}],
    }
    doc = {
        "window_analysis": {
            "throughput_width_ticks": 100,
            "throughput_step_ticks": 100,
            "recovery_window_ticks": 100,
        },
        "recovery": {"action": "calibrate", "candidates": [candidate]},
    }
    with pytest.raises(ValueError, match="differs from window_analysis"):
        EvaluationProtocol.model_validate(doc)
    candidate["window_ticks"] = 100
    candidate["faults"][0]["start_tick"] = 300
    assert (
        EvaluationProtocol.model_validate(doc).recovery.candidates[0].window_ticks
        == 100
    )


def test_official_experiment_evaluation_uses_yaml_window_settings(tmp_path):
    from pathlib import Path

    from smartsom import api

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    root = tmp_path / "evaluation"
    result = api.run_experiment(
        source,
        task="evaluate",
        sets=[
            f'experiment.run.output="{root}"',
            "experiment.run.methods=[rule]",
            "experiment.tracking.mode=disabled",
            "scenario.jobs.orders.total=8",
            "scenario.jobs.orders.initial=4",
            "scenario.jobs.arrivals.mean_interval_ticks=4",
            "experiment.run.metrics=detailed",
            "experiment.run.metric_interval=10",
            "experiment.validation_cases=4",
            "experiment.drain_ticks=100",
            "experiment.drain_frozen=true",
            "experiment.run.seconds=60",
            "experiment.evaluation.window_analysis={throughput_width_ticks: 100, throughput_step_ticks: 100, recovery_window_ticks: 100, same_run: {mode: manual, warmup_end_tick: 0, baseline_windows: 2, fraction: 0.95, consecutive_windows: 2, duration_from: repair, maximum_fault_events: 10}}",
        ],
    )
    assert result["status"] == "completed"
    row = json.loads(next(root.rglob("cases.jsonl")).read_text().splitlines()[0])
    assert row["research_metrics"]["sample_interval_ticks"] == 10
    assert row["window_analysis"]["width_ticks"] == 100
    assert row["window_analysis"]["recovery_window_ticks"] == 100
    assert row["window_analysis"]["status"] == "observed"
    same_run = row["window_analysis"]["same_run_recovery"]
    assert same_run["settings"]["mode"] == "manual"
    assert same_run["sample_window_ticks"] == 10
    assert row["window_analysis"]["windows"]
    for window in row["window_analysis"]["windows"]:
        expected = sum(
            o["completion_tick"] is not None
            and window["start"] < o["completion_tick"] <= window["end"]
            for o in row["research_metrics"]["orders"]
        )
        assert window["completed"] == expected


def test_real_simulator_cases_flow_into_paired_export(tmp_path):
    """Actual simulated decisions/events, with explicitly synthetic test identity."""
    from dataclasses import replace

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import BaseWorkload, generate_scenario
    from smartsom.config.base_layout import base_factory
    from smartsom.domain.production import Outage
    from smartsom.experiments.reservation_runtime import episode
    from tests.helpers.algorithm import algorithm_options

    factory = base_factory("small")
    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=20,
        initial_orders=10,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    options = algorithm_options(method="rule")
    nominal = generate_scenario(factory, workload, 17, "normal", 500)
    machine = factory.machines[0].machine_id
    disturbed = replace(nominal, outages=(Outage(machine, 40, 60),))
    reports = []
    for scenario in (nominal, disturbed):
        policy = build_policy(options, 101, training=False, selection="argmax")
        report, _, _ = episode(
            factory,
            workload,
            options,
            policy,
            17,
            "normal",
            500,
            scenario=scenario,
            metrics_level="detailed",
            metric_interval=10,
            max_ticks=300,
        )
        # This is a direct engine test, not an official evaluation manifest.
        report["evaluation_identity"] = {
            "checkpoint_sha256": None,
            "method": "rule",
            "learning": False,
            "selection": "argmax",
            "policy_seed": 101,
            "source_sha256": "test-engine",
            "runtime_sha256": "test-runtime",
            "algorithm_sha256": "test-rule",
        }
        reports.append(report)
    assert (
        reports[0]["research_metrics"]["fingerprints"]["exogenous"]
        == reports[1]["research_metrics"]["fingerprints"]["exogenous"]
    )
    assert any(
        event["kind"] == "breakdown" and event["tick"] == 40
        for event in reports[1]["research_metrics"]["events"]
    )
    assert any(
        event["kind"] == "repair" and event["tick"] == 60
        for event in reports[1]["research_metrics"]["events"]
    )
    path, document = config(tmp_path)
    document["nominal"] = write_case(tmp_path, "real-normal.jsonl", reports[0])
    document["perturbed"] = write_case(tmp_path, "real-fault.jsonl", reports[1])
    document["fault"] = {"machine_id": machine, "fault_tick": 40, "repair_tick": 60}
    document["windows"] = [{"start": 0, "end": 200}]
    document["overall_window"] = {"start": 0, "end": 200}
    path.write_text(yaml.safe_dump(document))
    output = tmp_path / "real-report.json"
    assert main(["paired-export", str(path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    row = result["overall"]
    for report, label in zip(reports, ("nominal", "perturbed"), strict=True):
        actual = sum(
            0 < (order.get("completion_tick") or 0) <= 200
            for order in report["research_metrics"]["orders"]
        )
        assert row[label]["completed"] == actual
    assert row["comparison"]["status"] == "observed"
    assert row["comparison"]["throughput_ratio"] == pytest.approx(
        row["perturbed"]["completed"] / row["nominal"]["completed"]
    )
    assert result["recovery"]["status"] == "not_applicable"
    assert result["recovery"]["reason"] == "zero_reference"
