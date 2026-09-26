"""Study orchestration failure, restart and frozen-input behavior."""

import json
from types import SimpleNamespace

import pytest

from smartsom import api
from smartsom.experiments import composable_study as study


@pytest.fixture
def child(tmp_path, monkeypatch):
    (tmp_path / "controls").mkdir()
    (tmp_path / "experiments").mkdir()
    plan = {
        "recipe": {"controls": ["initial"]},
        "datasets": {"test_000": {"V": {"low": 0.3}}},
        "implementation_sha256": "code",
        "entries": [
            {
                "id": "h0_low_ppo_auto",
                "H_case": "h0",
                "V_case": "low",
                "H": 0,
                "V": 0.2,
                "algorithm": "ppo",
                "transport": "auto",
            }
        ],
    }
    state = {"status": "prepared", "entries": {}, "controls": {}}
    study._json(tmp_path / "study.json", state)
    frozen = SimpleNamespace(scientific_sha256="science", evaluation_json="[{}]")
    calls = []
    row = dict(
        status="completed",
        truncated=False,
        engineering_failure=False,
        return_=1,
        delivered=1,
        flow_time=2,
        waiting=0,
        makespan=2,
    )
    row["return"] = row.pop("return_")

    def train(prepared):
        calls.append("train")
        root = tmp_path / "experiments/child"
        root.mkdir()
        study._json(
            root / "run.json",
            dict(
                kind="training",
                status="completed",
                scientific_sha256="science",
                implementation_sha256="code",
            ),
        )
        return SimpleNamespace(run_dir=root)

    def evaluate(source):
        calls.append("evaluate")
        root = tmp_path / "evaluated"
        root.mkdir(exist_ok=True)
        study._json(root / "run.json", {"results": [row]})
        return SimpleNamespace(run_dir=root)

    monkeypatch.setattr(study, "_load", lambda directory: plan)
    monkeypatch.setattr(study, "_prepared", lambda directory, spec: frozen)
    monkeypatch.setattr(study, "prepared_from_run", lambda source: frozen)
    monkeypatch.setattr(study, "evaluation_recipe", lambda prepared, path: prepared)
    monkeypatch.setattr(
        study, "evaluate_cases", lambda prepared, cases, **kwargs: [row]
    )
    monkeypatch.setattr(
        study,
        "show_study",
        lambda directory: json.loads((tmp_path / "study.json").read_text()),
    )
    monkeypatch.setattr(api, "train_prepared", train)
    monkeypatch.setattr(api, "evaluate", evaluate)
    return tmp_path, row, calls


def test_serial_execution_skips_completed_and_reports_test_v(child):
    directory, row, calls = child
    assert study.run_study(directory)["status"] == "completed"
    assert calls == ["train", "evaluate"]
    assert study.run_study(directory)["status"] == "completed"
    assert calls == ["train", "evaluate"]
    report = json.loads((directory / "summary.json").read_text())["rows"]
    assert len(report) == 2
    assert all(r["V_train"] == 0.2 and r["V_test"] == 0.3 for r in report)
    assert all(r["V_test_by_case"] == [0.3] for r in report)


def test_cached_failed_initial_control_is_retested_and_preserved_on_retry(child):
    directory, row, calls = child
    failed = dict(row, engineering_failure=True)
    study._json(directory / "controls/h0_low_ppo_auto_initial.json", [failed])
    assert study.run_study(directory)["status"] == "failed"
    assert calls == ["train", "evaluate"]
    assert study.run_study(directory)["status"] == "failed"
    assert calls == ["train", "evaluate"]
    assert study.run_study(directory, retry_failed=True)["status"] == "completed"
    assert calls == ["train", "evaluate", "evaluate"]
    backups = list((directory / "controls").glob("*.failed-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())[0]["engineering_failure"]
    cached = directory / "controls/h0_low_ppo_auto_initial.json"
    assert not json.loads(cached.read_text())[0]["engineering_failure"]


def test_source_input_hash_includes_referenced_policy_and_matching(tmp_path):
    import yaml

    config = tmp_path / "study.yaml"
    config.write_text("schema: smartsom.composable-study/v1\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump({"implementation": {"kind": "rule", "name": "edd"}})
    )
    composition = tmp_path / "composition.yaml"
    composition.write_text(
        yaml.safe_dump(
            {
                "pickup_matching": "policy.yaml",
                "groups": {"buffer": {"policy": "policy.yaml"}},
            }
        )
    )
    recipe = SimpleNamespace(
        factories={"h0": "policy.yaml"},
        transport={},
        algorithms={},
        workload="policy.yaml",
        compositions={"ppo": "composition.yaml"},
    )
    before = study.source_inputs(config, recipe)
    policy.write_text(
        yaml.safe_dump({"implementation": {"kind": "rule", "name": "spt"}})
    )
    after = study.source_inputs(config, recipe)
    assert before != after
    assert before["files"][str(config)] == after["files"][str(config)]
    assert before["files"][str(policy)] != after["files"][str(policy)]


def test_parallel_worker_preserves_failure_and_explicit_retry(child):
    directory, row, calls = child
    plan = study._load(directory)
    spec = plan["entries"][0]
    cached = directory / "controls" / (spec["id"] + "_initial.json")
    study._json(cached, [dict(row, engineering_failure=True)])
    worker = directory / "workers" / spec["id"]
    study._study_worker(directory, spec, plan, {"status": "running"}, False, worker)
    result = json.loads((worker / "entry.json").read_text())
    assert result["entries"][spec["id"]]["status"] == "failed"
    entry = result["entries"][spec["id"]]
    study._study_worker(directory, spec, plan, entry, True, worker)
    result = json.loads((worker / "entry.json").read_text())
    assert result["entries"][spec["id"]]["status"] == "completed"
    assert len(list((directory / "controls").glob("*.failed-*.json"))) == 1
