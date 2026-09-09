import json
import os
import shutil
import subprocess
from pathlib import Path

from smartsom.experiments.evidence import _file_digest, write_json
from smartsom.experiments.report import build_report, load_report_data


def model(tmp_path, name="model", seed=101, *, update=False):
    directory = tmp_path / name / "inference"
    directory.mkdir(parents=True)
    write_json(directory / "checkpoint.json", {"model": name})
    snapshot = directory.parent / "resolved_training.json"
    write_json(
        snapshot,
        {
            "schema": "smartsom.resolved-training/v1",
            "resolved": {"run": {"seed": seed}},
        },
    )
    if update:
        write_json(
            directory.parent / "manifest.json",
            {
                "schema": "smartsom.update-checkpoint/v1",
                "status": "complete",
                "files": {
                    str(p.relative_to(directory.parent)): _file_digest(p)
                    for p in directory.parent.rglob("*")
                    if p.is_file()
                },
            },
        )
    return {
        "path": str(directory),
        "manifest_sha256": _file_digest(directory / "checkpoint.json"),
        "training_snapshot": str(snapshot),
        "training_snapshot_sha256": _file_digest(snapshot),
    }


def planned(replication, algorithm, checkpoint=None):
    return {
        "case_id": "micro",
        "replication": replication,
        "algorithm_id": algorithm,
        "checkpoint": checkpoint,
        "world_sha256": f"world-{replication}",
    }


def evaluation(root, plan, results, *, coverage=None):
    root.mkdir(parents=True)
    write_json(root / "plan.json", {"entries": plan})
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.evaluation/v1",
            "status": "finished_with_failures",
            "requested": len(plan),
            "results": results,
            "input_coverage": coverage,
        },
    )
    return root


def test_report_retains_failure_before_any_evaluation_run(tmp_path):
    root = tmp_path / "failed-evaluation"
    root.mkdir()
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.evaluation/v1",
            "status": "failed",
            "stage": "model_import",
            "error": "ValueError: bundle checksum mismatch",
            "results": [],
            "requested": 0,
        },
    )
    data = load_report_data(root)
    assert len(data["evaluations"]) == 1
    record = data["evaluations"][0]
    assert record["status"] == "failed"
    assert record["stage"] == "model_import"
    assert "checksum mismatch" in record["error"]
    assert record["coverage"] == record["pairs"] == []
    report = build_report(root, root / "reports/report.html")
    assert "bundle checksum mismatch" in report.read_text()


def test_pair_delta_requires_unique_complete_matching_planned_world(tmp_path):
    checkpoint = model(tmp_path)
    plan, results = [], []
    for rep in range(6):
        pair = [planned(rep, "model", checkpoint), planned(rep, "builtin.spt")]
        plan.extend(pair)
        results.extend(
            [
                {**pair[0], "status": "completed", "makespan": 12},
                {**pair[1], "status": "completed", "makespan": 10},
            ]
        )
    results[2].update(status="not_completed", reason="policy_stalled", makespan=None)
    results.pop(4)  # missing model in replication 2
    next(r for r in results if r["replication"] == 3 and r["algorithm_id"] == "model")[
        "world_sha256"
    ] = "different"
    next(r for r in results if r["replication"] == 4 and r["algorithm_id"] == "model")[
        "checkpoint"
    ] = model(tmp_path, "other", 102)
    results.append(dict(results[-1]))  # duplicate baseline in replication 5
    root = evaluation(tmp_path / "evaluation", plan, results)
    data = load_report_data(root)["evaluations"][0]
    assert [r["status"] for r in data["pairs"]] == [
        "paired",
        "not_completed",
        "missing_model",
        "world_mismatch",
        "model_identity_mismatch",
        "duplicate_result",
    ]
    assert [r["makespan_delta"] for r in data["pairs"]] == [
        2,
        None,
        None,
        None,
        None,
        None,
    ]
    selected = next(
        r
        for r in data["coverage"]
        if r["algorithm_id"] == "model" and r["training_seed"] == 101
    )
    assert selected["requested"] == 6
    assert selected["completed"] == 3
    assert selected["failure_reasons"] == {"policy_stalled": 1, "missing_result": 2}


def test_same_model_replicates_and_different_training_seeds_are_not_pooled(tmp_path):
    first = model(tmp_path, "first", 101)
    second = model(tmp_path, "second", 102)
    # Even byte-identical saved model manifests must not merge different training seeds.
    Path(second["path"], "checkpoint.json").write_bytes(
        Path(first["path"], "checkpoint.json").read_bytes()
    )
    second["manifest_sha256"] = first["manifest_sha256"]
    plan = [
        planned(0, "model", first),
        planned(1, "model", first),
        planned(2, "model", second),
    ]
    root = evaluation(
        tmp_path / "evaluation",
        plan,
        [
            {**r, "status": "completed", "makespan": v}
            for r, v in zip(plan, [10, 14, 30], strict=True)
        ],
    )
    groups = load_report_data(root)["evaluations"][0]["coverage"]
    assert [
        (r["training_seed"], r["requested"], r["complete_case_mean_makespan"])
        for r in groups
    ] == [(101, 2, 12), (102, 1, 30)]


def test_missing_plan_has_unknown_coverage_and_no_delta(tmp_path):
    rows = [planned(0, "model"), planned(0, "builtin.spt")]
    root = evaluation(
        tmp_path / "evaluation",
        [],
        [{**r, "status": "completed", "makespan": 10} for r in rows],
    )
    (root / "plan.json").unlink()
    result = load_report_data(root)["evaluations"][0]
    assert all(r["requested"] is None for r in result["coverage"])
    assert result["pairs"][0]["status"] == "pair_plan_unavailable"
    assert result["pairs"][0]["makespan_delta"] is None


def test_checkpoint_copies_are_excluded_and_resume_attempts_remain_separate(tmp_path):
    root = tmp_path / "training"
    for relative, episodes in [
        ("evidence/training/first", 1),
        ("evidence/training/resumed", 2),
        ("evidence/training/resumed/checkpoints/update/training", 2),
        ("evidence/initialized-model", 9),
        ("imports/old", 9),
    ]:
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "episodes.jsonl").write_text(
            "\n".join(json.dumps({"episode": i, "return": -i}) for i in range(episodes))
        )
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "kind": "training",
            "seed": 101,
            "status": "completed",
            "paths": {"training": "evidence/training/resumed"},
            "attempts": [
                {
                    "status": "interrupted",
                    "paths": {"training": "evidence/training/first"},
                }
            ],
        },
    )
    runs = load_report_data(root)["runs"]
    assert len(runs) == 2
    assert [r["training"]["attempt"] for r in runs] == [0, 1]
    assert [r["training"]["current_attempt"] for r in runs] == [False, True]
    assert [len(r["series"][0]["points"]) for r in runs] == [1, 2]
    assert all("training seed 101" in r["label"] for r in runs)


def test_report_protects_all_new_model_references_and_preserves_legacy_bytes(tmp_path):
    new = model(tmp_path, "new", update=True)
    old = model(tmp_path, "old")
    plan = [planned(0, "model", new), planned(0, "checkpoint-other", old)]
    overlap = {
        "interpretation": "Different seeds alone do not establish cross-scenario generalization.",
        "training_history": {
            "status": "recorded_episodes_only",
            "all_training_samples_covered": False,
        },
    }
    root = evaluation(
        tmp_path / "evaluation",
        plan,
        [{**r, "status": "completed", "makespan": 12} for r in plan],
        coverage=overlap,
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    report = build_report(root, root / "reports/index.html")
    assert "all_training_samples_covered" in report.read_text()
    assert "recorded_episodes_only" in report.read_text()
    assert all(p.read_bytes() == content for p, content in before.items())
    references = list((Path(new["path"]).parent / "references").glob("*.json"))
    assert len(references) == 1
    assert json.loads(references[0].read_text())["reference"] == str(report)
    assert not (Path(old["path"]).parent / "references").exists()


def test_offline_dom_event_controls_filter_zoom_and_playback(tmp_path):
    """Independent DOM logic check, explicitly not a real browser acceptance test."""
    node = shutil.which("node")
    if node is None:
        import pytest

        if os.environ.get("SMARTSOM_REQUIRE_REPORTS") == "1":
            pytest.fail("Node is required to exercise the offline report controls")
        pytest.skip("Node is needed for the optional offline DOM harness")
    root = tmp_path / "events"
    root.mkdir()
    events = [
        {
            "kind": "dispatch",
            "simulation_time": 0,
            "sequence": 0,
            "machine_id": "M1",
            "action": {"operation_id": "A1"},
        },
        {
            "kind": "dispatch",
            "simulation_time": 0,
            "sequence": 1,
            "machine_id": "M2",
            "action": {"operation_id": "B1"},
        },
        {
            "kind": "complete",
            "simulation_time": 2,
            "sequence": 2,
            "machine_id": "M1",
            "action": {"operation_id": "A1"},
        },
        {
            "kind": "complete",
            "simulation_time": 3,
            "sequence": 3,
            "machine_id": "M2",
            "action": {"operation_id": "B1"},
        },
    ]
    for event in events:
        event["job_id"] = "A" if event["machine_id"] == "M1" else "B"
    (root / "trace.jsonl").write_text("\n".join(map(json.dumps, events)))
    report = build_report(root, tmp_path / "report.html")
    data = load_report_data(root)
    script = report.read_text().split("</script><script>")[1].split("</script>")[0]
    harness = Path(__file__).with_name("report_dom_harness.js")
    result = subprocess.run(
        [node, str(harness)],
        input=json.dumps({"data": data, "script": script}),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline DOM controls passed"


def test_train_evaluate_root_symlink_selections_protect_last_and_best(tmp_path):
    root = tmp_path / "training"
    training = root / "evidence/training/current"
    last = model(training / "checkpoints", "update-last", update=True)
    best = model(training / "checkpoints", "update-best", update=True)
    (training / "episodes.jsonl").write_text(json.dumps({"episode": 0, "return": -1}))
    (root / "checkpoints").mkdir()
    for name, checkpoint in (("last", last), ("best", best)):
        update = Path(checkpoint["path"]).parent
        write_json(
            training / "checkpoints" / f"{name}.json", {"checkpoint": str(update)}
        )
        (root / "checkpoints" / name).symlink_to(
            Path("../") / update.relative_to(root), target_is_directory=True
        )
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "kind": "train_evaluate",
            "seed": 101,
            "status": "completed",
            "paths": {"training": "evidence/training/current"},
        },
    )
    data = load_report_data(root)
    assert len(data["runs"]) == 1
    assert data["runs"][0]["training"]["training_seed"] == 101
    assert set(data["model_references"]) == {last["path"], best["path"]}
    assert not list(root.rglob("references"))
    build_report(root, root / "reports/training.html")
    assert len(list(root.rglob("references/*.json"))) == 2
