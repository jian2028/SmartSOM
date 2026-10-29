"""Real Base continuation and detached-process contracts."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/base-diagnostic/small.yaml"


@pytest.mark.parametrize("cap", [2000, 3000, 4000])
@pytest.mark.parametrize("finishes", [False, True])
def test_calibration_extension_respects_cap_and_incomplete_outcome(
    tmp_path, monkeypatch, cap, finishes
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime

    boundaries = []

    def bounded_case(factory, workload, options, policy, seed, condition, drain, **kw):
        # Exercise the actual calibration callback with controlled progress.
        seen = [drain]
        while drain < cap:
            drain = kw["extend"](
                SimpleNamespace(tick=drain), [{"last_productive": drain}]
            )
            seen.append(drain)
            assert drain <= cap
        if not finishes:
            assert (
                kw["extend"](SimpleNamespace(tick=drain), [{"last_productive": drain}])
                is None
            )
        boundaries.append(seen)
        return (
            {
                "condition": condition,
                "status": "completed" if finishes else "truncated",
                "tick": drain,
                "completed": workload.orders if finishes else 1,
                "wall_seconds": 0,
                "horizon_checks": [{"last_productive": drain}],
            },
            [],
            None,
        )

    monkeypatch.setattr(runtime, "episode", bounded_case)
    root = tmp_path / "calibration"
    with (
        nullcontext()
        if finishes
        else pytest.raises(RuntimeError, match="calibration incomplete")
    ):
        api.run_experiment(
            CONFIG,
            task="calibrate",
            sets=[f'experiment.run.output="{root}"', f"experiment.run.max_drain={cap}"],
        )
    expected = [2000] if cap == 2000 else [2000, cap]
    assert boundaries == [expected] * 8
    report = json.loads((root / "calibration.json").read_text())
    assert report["status"] == ("frozen" if finishes else "calibration_incomplete")
    assert report["drain_ticks"] <= cap
    assert all(row["candidate_drain"] == cap for row in report["results"])
    assert all(row["unresolved"] is not finishes for row in report["results"])


def test_legacy_calibration_rejects_limit_below_initial_before_freeze(tmp_path):
    from types import SimpleNamespace

    from smartsom.experiments.reservation_runtime import calibrate

    output = tmp_path / "calibration"
    with pytest.raises(ValueError, match="max_drain must be at least"):
        calibrate(
            SimpleNamespace(
                experiment=CONFIG, output=output, max_drain=1999, seconds=0.001, cases=1
            )
        )
    assert not output.exists()


@pytest.fixture
def calibrated(tmp_path):
    from smartsom import api

    shared = [
        "workload.orders=8",
        "workload.initial_orders=4",
        "workload.mean_interarrival=4",
        "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        "algorithm.production.policies.ppo.learner.params.epochs=1",
        "experiment.run.seconds=120",
    ]
    calibration = tmp_path / "calibration"
    api.run_experiment(
        CONFIG,
        task="calibrate",
        sets=shared + [f'experiment.run.output="{calibration}"'],
    )
    return shared + [f'experiment.run.calibration="{calibration / "calibration.json"}"']


def test_batch_wall_budget_preserves_outcome_and_runs_later_stage(tmp_path, calibrated):
    import yaml

    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "author-inputs"
    inputs.mkdir()
    batch_inputs = tmp_path / "batch-inputs"
    batch_inputs.mkdir()
    plan = compile_reservation(
        CONFIG,
        task="train",
        sets=calibrated + ["experiment.run.seconds=0.001"],
    )
    plan.materialize(inputs)
    document = yaml.safe_load((inputs / "experiment.yaml").read_text())
    for name in ("factory", "workload", "algorithm"):
        document[name] = str(inputs / document[name])
    document["run"]["output"] = str(tmp_path / "batch-output/train")
    (batch_inputs / "train.yaml").write_text(yaml.safe_dump(document))
    document["run"].update(
        task="smoke",
        seconds=120,
        methods=["rule"],
        smoke_ticks=20,
        output=str(tmp_path / "batch-output/later"),
    )
    document["batch"]["stage"] = 1
    (batch_inputs / "later.yaml").write_text(yaml.safe_dump(document))

    def cli(*args):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "smartsom.experiments.cli",
                *map(str, args),
                "--progress",
                "off",
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    cli("batch-run", batch_inputs, "--max-concurrent", "1", "--foreground")
    (root,) = (tmp_path / "batch-output").iterdir()
    parent = json.loads((root / "batch.json").read_text())
    assert parent["status"] == "completed"
    assert parent["files"]["train"]["native_status"] == "wall_budget_reached"
    assert parent["files"]["later"]["status"] == "completed"
    child = root / "experiments/train"
    progress = json.loads((child / "progress.json").read_text())
    assert progress["status"] == "wall_budget_reached"
    assert progress["physical_ticks"] < progress["tick_budget"]
    assert progress["episodes"] == progress["updates"] == 0
    batch = json.loads((child / "batch.json").read_text())
    assert batch["status"] == "completed"
    assert batch["native_status"] == progress["status"]
    assert batch["deadline_unix"] == progress["deadline_unix"]
    later = json.loads((root / "experiments/later/summary.json").read_text())
    assert later["results"][0]["tick"] == 20
    # Resume must not reclassify the terminal outcome or renew its wall budget.
    evidence = {
        p: p.read_bytes()
        for subdir in (child, root / "experiments/later")
        for p in subdir.rglob("*")
        if p.is_file()
    }
    cli("resume", root, "--foreground")
    cli("resume", root, "--retry-failed", "--foreground")
    assert all(p.read_bytes() == content for p, content in evidence.items())
    assert (
        json.loads((root / "batch.json").read_text())["files"]["train"]["native_status"]
        == "wall_budget_reached"
    )


def test_complete_update_recovery_is_exact_and_bounded(
    tmp_path, calibrated, monkeypatch
):
    import torch

    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime

    original = runtime.episode
    calls = 0
    interrupted = False

    def episode(*args, **kwargs):
        nonlocal calls
        kwargs["max_ticks"] = min(40, kwargs["max_ticks"])
        result = original(*args, **kwargs)
        calls += 1
        if interrupted and calls == 2:
            raise RuntimeError("injected after rollout before commit")
        return result

    monkeypatch.setattr(runtime, "episode", episode)
    for name in ("uninterrupted", "recovered"):
        calls = 0
        interrupted = name == "recovered"
        result = api.run_experiment(
            CONFIG,
            task="train",
            sets=calibrated
            + [
                f'experiment.run.output="{tmp_path / name}"',
                "experiment.train_ticks=120",
            ],
        )
        assert result["physical_ticks"] == 120 and result["episodes"] == 3
    a = torch.load(tmp_path / "uninterrupted/last.pt", weights_only=False)
    b = torch.load(tmp_path / "recovered/last.pt", weights_only=False)

    def equal(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(a, b)
    assert len(json.loads((tmp_path / "recovered/attempts.json").read_text())) == 1
    from smartsom.experiments.reservation import execute_saved

    assert execute_saved(tmp_path / "recovered")["episodes"] == 3
    path = tmp_path / "recovered/inputs/algorithm.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="identity"):
        execute_saved(tmp_path / "recovered")


def test_background_launch_completes_with_frozen_inputs(tmp_path):
    root = tmp_path / "background"
    command = [
        sys.executable,
        "-m",
        "smartsom.experiments.cli",
        "run",
        str(CONFIG),
        "--background",
        "--set",
        f'experiment.run.output="{root}"',
        "--set",
        "experiment.run.methods=[rule]",
        "--set",
        "experiment.run.smoke_ticks=20",
        "--progress",
        "off",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (root / "progress.json").exists():
            value = json.loads((root / "progress.json").read_text())
            if value["status"] in ("completed", "failed"):
                break
        time.sleep(0.1)
    assert value["status"] == "completed"
    assert (root / "control/background-request.json").exists()
    assert (root / "logs/stdout.log").exists()


def test_stop_is_not_automatically_retried(tmp_path, calibrated, monkeypatch):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    original = runtime.episode

    def stopped(*args, **kwargs):
        raise StopRequested("test stop")

    monkeypatch.setattr(runtime, "episode", stopped)
    root = tmp_path / "stopped"
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task="train",
            sets=calibrated
            + [f'experiment.run.output="{root}"', "experiment.train_ticks=40"],
        )
    assert len(json.loads((root / "attempts.json").read_text())) == 1
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(runtime, "episode", original)
    assert execute_saved(root)["physical_ticks"] == 40
    assert json.loads((root / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("task,total", [("calibrate", 8), ("evaluate", 8)])
def test_stage_cases_resume_without_repeating_commits(
    tmp_path, calibrated, monkeypatch, task, total
):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    root = tmp_path / (task + "-resume")
    original = runtime.episode
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise StopRequested("interrupt fourth case")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "episode", interrupted)
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task=task,
            sets=calibrated
            + [
                f'experiment.run.output="{root}"',
                "experiment.validation_cases=4",
                "experiment.run.methods=[rule,random]",
            ],
        )
    receipts = sorted((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 3
    before = {p.name: p.read_bytes() for p in receipts}
    manifest = (root / "manifest.json").read_bytes()
    result = execute_saved(root)
    assert result["status"] in {"completed", "frozen"}
    assert calls == total + 1
    assert (root / "manifest.json").read_bytes() == manifest
    assert all(
        (root / "case-commits" / name).read_bytes() == data
        for name, data in before.items()
    )
    assert len(list((root / "case-commits").glob("case-*.json"))) == total


def test_evaluation_expired_deadline_cannot_reset_on_resume(
    tmp_path, calibrated, monkeypatch
):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    root = tmp_path / "expired"
    original = runtime.episode

    def stopped(*args, **kwargs):
        raise StopRequested("before first case")

    monkeypatch.setattr(runtime, "episode", stopped)
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task="evaluate",
            sets=calibrated
            + [
                f'experiment.run.output="{root}"',
                "experiment.validation_cases=4",
                "experiment.run.methods=[rule,random]",
            ],
        )
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(runtime, "episode", original)
    monkeypatch.setattr(runtime.time, "time", lambda: manifest["started_unix"] + 121)
    with pytest.raises(TimeoutError):
        execute_saved(root)
    assert not list((root / "case-commits").glob("case-*.json"))
    assert json.loads((root / "manifest.json").read_text()) == manifest


def test_episode_budget_validation_selection_and_periodic_checkpoint(
    tmp_path, calibrated, monkeypatch
):
    from smartsom import api
    from smartsom.experiments import reservation_validation

    original_validate = reservation_validation.validate
    validated = []

    def inspect_committed_progress(session):
        active = len(session.rows) in session.ex.validation.milestones
        if active:
            progress = json.loads((session.root / "progress.json").read_text())
            assert progress["status"] == "validating"
            assert progress["episodes"] == len(session.rows)
            assert progress["physical_ticks"] == session.ticks
            assert progress["committed_physical_ticks"] == session.ticks
            validated.append(len(session.rows))
        result = original_validate(session)
        if active:
            progress = json.loads((session.root / "progress.json").read_text())
            assert progress["episodes"] == len(session.rows)
            assert progress["physical_ticks"] == session.ticks
        return result

    monkeypatch.setattr(reservation_validation, "validate", inspect_committed_progress)

    root = tmp_path / "episodes"
    result = api.run_experiment(
        CONFIG,
        task="train",
        sets=calibrated
        + [
            f'experiment.run.output="{root}"',
            "experiment.train_episodes=2",
            "experiment.condition_sampling=balanced_cycle",
            "experiment.validation_cases=4",
            "experiment.validation.milestones=[1,2]",
            "experiment.validation.sample_action_seeds=1",
            "experiment.validation.selection=sample_complete_cost",
            "experiment.checkpoint_every_episodes=2",
        ],
    )
    assert result["status"] == "completed" and result["episodes"] == 2
    assert validated == [1, 2]
    assert result["episode_budget"] == 2 and result["tick_budget"] is None
    rows = [
        json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["condition"] for row in rows] == ["normal", "low"]
    assert (root / "checkpoints/episode-00000002.pt").is_file()
    assert not (root / "checkpoints/episode-00000001.pt").exists()
    selection = json.loads((root / "selection.json").read_text())
    assert selection["status"] == "selected"
    assert selection["selected_episode"] in (1, 2)
    receipts = {p: p.read_bytes() for p in (root / "validation").rglob("cases.jsonl")}
    for path in receipts:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        progress = rows[0]["evaluation_identity"]["training_progress"]
        assert progress["episodes"] in (1, 2)
        assert progress["physical_ticks"] > 0
        assert progress["rollout_and_update_seconds"] > 0
        assert all(
            row["evaluation_identity"]["training_progress"] == progress for row in rows
        )

    assert len(receipts) == 4
    assert api.resume(root)["episodes"] == 2
    assert all(p.read_bytes() == content for p, content in receipts.items())


def test_evaluation_resume_freezes_actual_platform_and_threads(
    tmp_path, calibrated, monkeypatch
):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    root = tmp_path / "runtime-panel"
    original_episode = runtime.episode
    original_source = runtime.source_identity
    original_evaluate = runtime._evaluate
    setting = {"threads": 1, "platform": None}
    calls = 0

    def source():
        identity = original_source()
        if setting["platform"] is not None:
            identity["environment"]["platform"] = setting["platform"]
        return identity

    def evaluate(args):
        args.numerical_threads = setting["threads"]
        return original_evaluate(args)

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StopRequested("after first durable case")
        return original_episode(*args, **kwargs)

    monkeypatch.setattr(runtime, "source_identity", source)
    monkeypatch.setattr(runtime, "_evaluate", evaluate)
    monkeypatch.setattr(runtime, "episode", interrupted)
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task="evaluate",
            sets=calibrated
            + [
                f'experiment.run.output="{root}"',
                "experiment.validation_cases=4",
                "experiment.run.methods=[rule]",
            ],
        )
    receipts = list((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 1
    before = receipts[0].read_bytes()
    monkeypatch.setattr(runtime, "episode", original_episode)
    for threads, platform in ((2, None), (1, "other-platform")):
        setting.update(threads=threads, platform=platform)
        with pytest.raises(ValueError, match="frozen training settings"):
            execute_saved(root)
        assert len(list((root / "case-commits").glob("case-*.json"))) == 1
        assert receipts[0].read_bytes() == before
    setting.update(threads=1, platform=None)
    execute_saved(root)
    rows = [
        json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert len({r["evaluation_identity"]["runtime_sha256"] for r in rows}) == 1


def test_actual_native_worker_pause_resume_and_stop(tmp_path, calibrated):
    import yaml

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import runtime_control

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan = compile_reservation(
        CONFIG,
        task="train",
        sets=calibrated + ["experiment.train_ticks=10000", "experiment.run.seconds=60"],
    )
    plan.materialize(inputs)
    document = yaml.safe_load((inputs / "experiment.yaml").read_text())
    for name in ("factory", "workload", "algorithm"):
        document[name] = str(inputs / document[name])
    document["run"]["output"] = str(tmp_path / "output" / "training")
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "training.yaml").write_text(yaml.safe_dump(document))
    log = tmp_path / "driver.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "smartsom.experiments.cli",
                "run",
                str(folder),
                "--foreground",
                "--max-concurrent",
                "1",
                "--progress",
                "off",
            ],
            stdout=stream,
            stderr=stream,
        )
        root = None

        def poll(predicate):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = predicate()
                if result:
                    return result
                if process.poll() is not None:
                    pytest.fail(log.read_text())
                time.sleep(0.05)
            pytest.fail(log.read_text())

        try:
            root = poll(lambda: next((tmp_path / "output").glob("*"), None))
            child = root / "experiments/training"
            marker = child / "continuation/current.json"
            poll(lambda: marker.exists())
            runtime_control.submit(root, "pause")
            poll(
                lambda: (
                    runtime_control.snapshot(root)
                    .get("tasks", {})
                    .get("training", {})
                    .get("status")
                    == "paused"
                )
            )
            saved = marker.read_bytes()
            assert json.loads((child / "batch.json").read_text())["status"] == "paused"
            runtime_control.submit(root, "resume")
            poll(lambda: marker.read_bytes() != saved)
            runtime_control.submit(root, "stop")
            assert process.wait(timeout=30) == 1, log.read_text()
            assert json.loads((child / "batch.json").read_text())["status"] == "stopped"
        finally:
            if process.poll() is None and root is not None:
                runtime_control.submit(root, "stop")
                process.wait(timeout=30)


def test_completion_protocol_freezes_guard_worlds_and_common_window(
    tmp_path, calibrated
):
    from smartsom import api

    root = tmp_path / "completion"
    api.run_experiment(
        CONFIG,
        task="evaluate",
        sets=calibrated
        + [
            f'experiment.run.output="{root}"',
            "experiment.run.methods=[rule]",
            "experiment.validation_cases=4",
            "experiment.evaluation.mode=completion",
            "experiment.evaluation.completion_drain_multiplier=2",
        ],
    )
    launch = json.loads((root / "plan.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert launch["run_id"] == manifest["run_id"]
    rows = [
        json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()
    ]
    frozen = json.loads((root / "case-commits/plan.json").read_text())
    assert len(rows) == len(frozen["cases"]) == 4
    for row, entry in zip(rows, frozen["cases"], strict=True):
        assert (
            row["common_window"]["end_tick"]
            == row["last_arrival"] + manifest["drain_ticks"]
        )
        assert (
            entry["case"]["scenario"]["tick_limit"]
            == row["last_arrival"] + 2 * manifest["drain_ticks"]
        )
        assert row["tick"] <= entry["case"]["scenario"]["tick_limit"]
        if row["status"] != "completed":
            assert row["termination_reason"] == "completion_protection_limit"
    report = json.loads((root / "research-summary.json").read_text())
    assert report["methods"]["rule"]["all"]["actual_cases"] == 4


def test_car_uses_real_validation_panels_and_frozen_rule_reference(
    tmp_path, calibrated
):
    from smartsom import api

    reference = tmp_path / "reference"
    api.run_experiment(
        CONFIG,
        task="evaluate",
        sets=calibrated
        + [
            f'experiment.run.output="{reference}"',
            "experiment.run.methods=[rule]",
            "experiment.validation_cases=4",
        ],
    )
    target = reference / "research-summary.json"
    report = json.loads(target.read_text())
    assert report["methods"]["rule"]["all"]["metrics"]["joint_cost"] is not None
    root = tmp_path / "learning"
    api.run_experiment(
        CONFIG,
        task="train",
        sets=calibrated
        + [
            f'experiment.run.output="{root}"',
            "experiment.train_episodes=3",
            "experiment.validation_cases=4",
            "experiment.validation.milestones=[1,2,3]",
            "experiment.validation.modes=[sample]",
            "experiment.validation.sample_action_seeds=1",
            f'experiment.evaluation.learning_target.reference="{target}"',
        ],
    )
    result = json.loads((root / "learning-target.json").read_text())
    assert result["modes"]["sample"]["status"] in {"reached", "censored"}
    assert result["target"]["reference_sha256"]
    assert (
        result["target"]["rule_cost"]
        == report["methods"]["rule"]["all"]["metrics"]["joint_cost"]
    )
    frozen = list((root / "inputs/evaluation-references").glob("*.json"))
    assert len(frozen) == 1 and frozen[0].read_bytes() == target.read_bytes()
    from smartsom.config.codec import ConfigurationError
    from smartsom.config.reservation import compile_reservation

    with pytest.raises(ConfigurationError, match="panel definition"):
        compile_reservation(
            CONFIG,
            task="train",
            sets=calibrated
            + [
                "experiment.validation_cases=8",
                f'experiment.evaluation.learning_target.reference="{target}"',
            ],
        )
    assert report["reference_panel"]["sha256"]
    assert len(report["reference_panel"]["worlds"]) == 4
    # A retained learning target must not force final test worlds to equal validation.
    api.run_experiment(
        CONFIG,
        task="evaluate",
        sets=calibrated
        + [
            f'experiment.run.output="{tmp_path / "final-test"}"',
            "experiment.run.methods=[rule]",
            "experiment.run.split=test",
            "experiment.validation_cases=4",
            "experiment.test_cases=4",
            f'experiment.evaluation.learning_target.reference="{target}"',
        ],
    )


@pytest.mark.parametrize("corrupt", ["checkpoint_sha256", "source_sha256", None])
def test_recovery_learning_checkpoint_evidence_checked_before_cases(tmp_path, corrupt):
    import hashlib

    import yaml

    from smartsom import api
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import BaseAlgorithm
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation_runtime import source_identity

    candidate = dict(
        name="probe",
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
    checkpoint = tmp_path / "model.pt"
    sets = [
        f'experiment.run.output="{tmp_path / "run"}"',
        "experiment.run.methods=[configured]",
        "experiment.run.seconds=30",
        "experiment.tracking.mode=disabled",
        f'experiment.run.checkpoint="{checkpoint}"',
        "experiment.evaluation.recovery.action=calibrate",
        f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
    ]
    plan = compile_reservation(CONFIG, task="calibrate", sets=sets)
    options = BaseAlgorithm.model_validate(
        yaml.safe_load(dict(plan.documents)["algorithm"])
    )
    build_policy(options, 77).save(checkpoint)
    evidence = {
        "schema": "smartsom.checkpoint-evidence/v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_sha256": source_identity()["implementation_sha256"],
        "episodes": 0,
        "physical_ticks": 0,
        "train_seed": 77,
        "run_id": "engineering-check",
    }
    if corrupt:
        evidence[corrupt] = "invalid"
    checkpoint.with_suffix(".pt.json").write_text(json.dumps(evidence))
    if corrupt:
        with pytest.raises(
            ValueError,
            match="checkpoint training (evidence digest mismatch|source differs)",
        ):
            api.run_experiment(CONFIG, task="calibrate", sets=sets)
        assert not (tmp_path / "run/cases.jsonl").exists()
    else:
        result = api.run_experiment(CONFIG, task="calibrate", sets=sets)
        assert result["status"] == "completed"
        case = json.loads((tmp_path / "run/cases.jsonl").read_text().splitlines()[0])
        assert case["evaluation_identity"]["training_progress"]["train_seed"] == 77
        assert case["evaluation_identity"]["training_progress"]["episodes"] == 0
        manifest = json.loads((tmp_path / "run/manifest.json").read_text())
        assert manifest["training_progress"]["run_id"] == "engineering-check"
