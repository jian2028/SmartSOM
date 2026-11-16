"""Scientific calibration and public RLlib/Tune lifecycle contracts.

Retired native lifecycle snapshots and coverage mapping are in
archive/rllib-transition-tests.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/examples/small/default/experiment.yaml"


@pytest.fixture(autouse=True)
def standalone_recipe(tmp_path, monkeypatch):
    """Exercise single-trial internals; research-directory expansion has its own gate."""
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    compile_reservation(CONFIG, sets=["experiment.tracking.mode=disabled"]).materialize(
        inputs
    )
    monkeypatch.setattr(sys.modules[__name__], "CONFIG", inputs / "experiment.yaml")


def _run_phase(config, *, task=None, sets=()):
    """Invoke scientific phases explicitly, without scheduler or implicit retry.

    Public Tune behavior has separate real-controller tests. These local phases
    allow deterministic fault injection inside rollout/commit boundaries.
    """
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation import allocate

    return _resume_phase(allocate(compile_reservation(config, task=task, sets=sets)))


def _resume_phase(root):
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.reservation import arguments, load

    root = Path(root)
    load(root)
    args, settings = arguments(root)
    assert settings.task == "calibrate"
    runtime.calibrate(args)
    return json.loads((root / "progress.json").read_text())


@pytest.mark.parametrize("cap", [2000, 3000, 4000])
@pytest.mark.parametrize("finishes", [False, True])
def test_calibration_extension_respects_cap_and_incomplete_outcome(
    tmp_path, monkeypatch, cap, finishes
):
    from contextlib import nullcontext
    from types import SimpleNamespace

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
        _run_phase(
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


def test_batch_wall_budget_preserves_outcome_and_runs_later_stage(tmp_path):
    import yaml

    from tests.helpers.rllib_runtime import author_runtime, command, last_checkpoint

    folder = tmp_path / "batch-inputs"
    path, profile = author_runtime(folder / "train", evaluation=False)
    doc = yaml.safe_load(path.read_text())
    doc["run"]["seconds"] = 0.001
    path.write_text(yaml.safe_dump(doc))
    later, _ = author_runtime(folder / "later", evaluation=False)
    doc = yaml.safe_load(later.read_text())
    doc["run"].update(task="smoke", methods=["rule"], smoke_ticks=20)
    doc["batch"] = {"stage": 1}
    later.write_text(yaml.safe_dump(doc))
    command(
        "run",
        folder,
        "--profile",
        profile,
        "--max-concurrent",
        "1",
        "--output-root",
        tmp_path / "outputs",
        "--progress",
        "off",
    )
    (root,) = (tmp_path / "outputs").iterdir()
    files = json.loads((root / "plan.json").read_text())["files"]
    train = next(row for row in files if row["task"] == "train")
    smoke = next(row for row in files if row["task"] == "smoke")
    child = root / train["child"]
    progress = json.loads((child / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["physical_ticks"] == 0 and progress["algorithm_iteration"] == 0
    state = last_checkpoint(child) / "smartsom-runtime.json"
    assert json.loads(state.read_text())["deadline_unix"] <= time.time()
    assert (
        json.loads((root / smoke["child"] / "summary.json").read_text())["results"][0][
            "tick"
        ]
        == 20
    )
    before = {
        p: p.read_bytes()
        for p in (
            state,
            child / "progress.json",
            root / smoke["child"] / "summary.json",
        )
    }
    command("resume", root, "--foreground", "--progress", "off")
    command("retry", root, "--all", "--foreground", "--progress", "off")
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads((root / "batch.json").read_text())["status"] == "completed"


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


def test_actual_tune_cli_pause_resume_and_stop(tmp_path):
    from tests.helpers.rllib_runtime import (
        assert_released,
        author_runtime,
        children,
        command,
        environment,
        last_checkpoint,
        wait_for,
    )

    path, profile = author_runtime(
        tmp_path / "official", ticks=1000000, evaluation=False, tracking="offline"
    )
    root = path.parent / "run"
    log = tmp_path / "driver.log"
    args = [sys.executable, "-m", "smartsom.experiments.cli"]
    with log.open("w") as stream:
        process = subprocess.Popen(
            [*args, "run", str(path), "--profile", str(profile), "--progress", "off"],
            stdout=stream,
            stderr=stream,
            env=environment(),
        )
        try:
            wait_for(lambda: last_checkpoint(root), process, log)
            owned = children(process.pid)
            assert owned
            command("pause", root)
            assert process.wait(timeout=90) == 0, log.read_text()
            assert_released(owned)
            assert json.loads((root / "batch.json").read_text())["status"] == "paused"
            checkpoint = last_checkpoint(root)
            metadata = json.loads((checkpoint / "smartsom-runtime.json").read_text())
            before = json.loads((root / "progress.json").read_text())[
                "algorithm_iteration"
            ]
            assert metadata["deadline_unix"] is not None
            process = subprocess.Popen(
                [*args, "resume", str(root), "--foreground", "--progress", "off"],
                stdout=stream,
                stderr=stream,
                env=environment(),
            )
            wait_for(
                lambda: (
                    json.loads((root / "progress.json").read_text()).get(
                        "algorithm_iteration", 0
                    )
                    > before
                ),
                process,
                log,
            )
            owned = children(process.pid)
            command("stop", root)
            assert process.wait(timeout=90) in (0, 1), log.read_text()
            assert_released(owned)
            assert json.loads((root / "batch.json").read_text())["status"] == "stopped"
            final = json.loads(
                (last_checkpoint(root) / "smartsom-runtime.json").read_text()
            )
            assert final["deadline_unix"] == metadata["deadline_unix"]
            assert any(
                event["event"] == "checkpoint_restored"
                for event in final["execution"]["events"].values()
            )
            events = [
                json.loads(line)
                for journal in (root / "tracking").glob("*.jsonl")
                for line in journal.read_text().splitlines()
            ]
            ids = [event["event_id"] for event in events if event.get("event_id")]
            assert ids and len(ids) == len(set(ids))
        finally:
            if process.poll() is None:
                command("stop", root)
                process.wait(timeout=90)
