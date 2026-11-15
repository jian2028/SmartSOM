"""Scientific Rule calibration/evaluation through the official Tune entrypoint."""

import json

import pytest


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
        operations={"resources": {"cpus": 3}},
        sets=[
            f'experiment.run.output="{root}"',
            "experiment.run.methods=[rule]",
            "experiment.tracking.mode=disabled",
            "scenario.jobs.orders.total=8",
            "scenario.jobs.orders.initial=4",
            "scenario.jobs.arrivals.mean_interval_ticks=4",
            "experiment.run.metrics=detailed",
            "experiment.run.metric_interval=10",
            "experiment.periodic_evaluation_cases=4",
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


def test_recovery_candidate_scan_real_entry_and_receipts(tmp_path):
    from pathlib import Path

    from smartsom import api

    candidate = dict(
        name="engineering",
        load="normal",
        orders=4,
        initial_orders=2,
        mean_interarrival=50,
        horizon_ticks=100,
        window_ticks=10,
        baseline_windows=3,
        minimum_window_deliveries=1,
        maximum_cv=0.1,
        maximum_relative_trend=0.1,
        faults=[dict(machine_id="machine_00", start_tick=40, duration_ticks=10)],
    )
    config = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    config = compile_reservation(config).materialize(inputs)
    root = tmp_path / "scan"
    result = api.run_experiment(
        config,
        task="calibrate",
        operations={"resources": {"cpus": 3}},
        sets=[
            f'experiment.run.output="{root}"',
            "experiment.run.methods=[rule]",
            "experiment.run.seconds=30",
            "experiment.tracking.mode=disabled",
            "experiment.evaluation.recovery.action=calibrate",
            f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
        ],
    )
    assert result["status"] == "completed"
    report = json.loads((root / "recovery-calibration.json").read_text())
    assert report["accepted"] == {"engineering": False}
    assert len(report["results"]) == 1
    receipts = list((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 1
    row = json.loads((root / "cases.jsonl").read_text())
    assert row["tick"] == 100
    assert row["research_metrics"]["entity_series"][-1]["tick"] == 100


def test_frozen_recovery_scan_is_consumed_by_real_controlled_evaluation(tmp_path):
    from pathlib import Path

    from smartsom import api

    candidate = dict(
        name="engineering",
        load="normal",
        orders=64,
        initial_orders=32,
        mean_interarrival=100,
        horizon_ticks=1400,
        window_ticks=200,
        baseline_windows=3,
        minimum_window_deliveries=1,
        maximum_cv=100,
        maximum_relative_trend=100,
        faults=[dict(machine_id="machine_00", start_tick=600, duration_ticks=10)],
    )
    config = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    config = compile_reservation(config).materialize(inputs)
    shared = [
        "experiment.run.methods=[rule]",
        "experiment.run.seconds=30",
        "experiment.tracking.mode=disabled",
        "scenario.jobs.processing.base_ticks={A: 1, B: 1, C: 1, D: 1}",
        "experiment.periodic_evaluation_cases=4",
        "experiment.run.metric_interval=10",
        "experiment.evaluation.window_analysis={throughput_width_ticks: 100, throughput_step_ticks: 100, recovery_window_ticks: 200, same_run: {mode: auto, warmup_end_tick: 0, fraction: 0.95, consecutive_windows: 2, duration_from: repair, maximum_fault_events: 10, auto: {plateau_windows: 2, preset_windows: 1, maximum_cv: 0, maximum_relative_trend: 0, maximum_upward_trend: 0, search_order: latest}}}",
        f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
    ]
    scan, evaluated = tmp_path / "scan", tmp_path / "evaluate"
    api.run_experiment(
        config,
        task="calibrate",
        operations={"resources": {"cpus": 3}},
        sets=shared
        + [
            f'experiment.run.output="{scan}"',
            "experiment.evaluation.recovery.action=calibrate",
        ],
    )
    report = json.loads((scan / "recovery-calibration.json").read_text())
    assert report["accepted"]["engineering"], report["results"]
    api.run_experiment(
        config,
        task="evaluate",
        operations={"resources": {"cpus": 3}},
        sets=shared
        + [
            f'experiment.run.output="{evaluated}"',
            "experiment.evaluation.recovery.action=evaluate",
            "experiment.evaluation.recovery.selected=engineering",
            f'experiment.evaluation.recovery.calibration="{scan / "recovery-calibration.json"}"',
        ],
    )
    rows = [
        json.loads(line)
        for line in (evaluated / "cases.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["research_metrics"]["outage_realization"] == []
    assert rows[1]["research_metrics"]["outage_realization"] == [
        dict(machine_id="machine_00", start=600, end=610)
    ]
    event = rows[1]["window_analysis"]["same_run_recovery"]["events"][0]
    assert (event["machine_id"], event["fault_tick"], event["repair_tick"]) == (
        "machine_00",
        600,
        610,
    )
    assert event["period_selection"]["fault_period"]["status"] == "preset"
    assert event["period_selection"]["fault_period"]["reason"] == "plateau_not_found"
    assert (
        rows[0]["research_metrics"]["fingerprints"]["orders"]
        == rows[1]["research_metrics"]["fingerprints"]["orders"]
    )
    result = json.loads((evaluated / "recovery-summary.json").read_text())
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] in {
        "recovered",
        "not_recovered",
        "no_observed_degradation",
    }
    # Exercise the real controlled-evaluation records through the independent
    # paired exporter, with expectations calculated from the saved order ticks.
    import hashlib

    import yaml

    from smartsom.telemetry.paired_export import export

    cases = evaluated / "cases.jsonl"
    sha = hashlib.sha256(cases.read_bytes()).hexdigest()
    spec = {
        "schema": "smartsom.paired-export/v1",
        "nominal": {"path": str(cases), "sha256": sha, "line": 1},
        "perturbed": {"path": str(cases), "sha256": sha, "line": 2},
        "windows": [{"start": 600, "end": 800}],
        "overall_window": {"start": 600, "end": 800},
        "fault": {"machine_id": "machine_00", "fault_tick": 600, "repair_tick": 610},
        "recovery": {
            "semantics": "paired_repair_anchored",
            "duration_from": "repair",
            "version": "engineering-test",
            "reference": "paired-nominal-case",
            "window_ticks": 200,
            "consecutive_windows": 2,
            "fraction": 0.95,
        },
    }
    protocol = tmp_path / "paired.yaml"
    protocol.write_text(yaml.safe_dump(spec))
    paired = export(protocol)
    for label, row in zip(("nominal", "perturbed"), rows, strict=True):
        expected = sum(
            600 < order["completion_tick"] <= 800
            for order in row["research_metrics"]["orders"]
            if order["completion_tick"] is not None
        )
        assert paired["windows"][0][label]["completed"] == expected
    assert paired["overall"]["comparison"]["status"] == "observed"
    spec.pop("nominal")
    spec["recovery"].update(
        semantics="same_run_pre_fault",
        baseline={
            "window_ticks": 200,
            "baseline_windows": 1,
            "minimum_window_deliveries": 1,
            "warmup_end_tick": 0,
        },
    )
    protocol.write_text(yaml.safe_dump(spec))
    same_run = export(protocol)
    assert same_run["schema"] == "smartsom.same-run-export-result/v1"
    assert same_run["recovery"]["status"] in {
        "threshold_confirmed",
        "censored",
        "not_applicable",
        "insufficient_baseline",
    }
    assert same_run["windows"][0]["perturbed"]["completed"] == sum(
        600 < order["completion_tick"] <= 800
        for order in rows[1]["research_metrics"]["orders"]
        if order["completion_tick"] is not None
    )
    # Hand-audited deterministic engineering case: 11 jobs in (400,600],
    # 1 in (600,610], then 0, 4, 0 in the three complete 200-tick windows.
    times = [
        order["completion_tick"]
        for order in rows[1]["research_metrics"]["orders"]
        if order["completion_tick"] is not None
    ]
    assert [
        sum(start < t <= end for t in times)
        for start, end in [
            (400, 600),
            (600, 610),
            (610, 810),
            (810, 1010),
            (1010, 1210),
        ]
    ] == [11, 1, 0, 4, 0]
    recovery = same_run["recovery"]
    assert recovery["baseline"]["mean_throughput"] == pytest.approx(11 / 200)
    assert recovery["fault_period_mean_ratio"]["ratio"] == pytest.approx(20 / 11)
    assert recovery["status"] == "censored"
    assert recovery["reason"] == "observation_ended"
    assert recovery["observed_repair_ticks"] == 790
    assert recovery["adaptability"] is None
