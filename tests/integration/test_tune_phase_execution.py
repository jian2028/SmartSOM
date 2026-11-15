"""Official search censoring and public cross-process continuation."""

import json
import subprocess
import sys

import yaml


def test_official_optuna_censoring_and_public_state_restore(tmp_path):
    import optuna
    from ray.tune.search.optuna import OptunaSearch

    from smartsom.experiments.tune_search_adapter import CensoredEvaluationSearch

    storage = optuna.storages.InMemoryStorage()

    def search():
        return CensoredEvaluationSearch(
            OptunaSearch(
                space={"x": optuna.distributions.FloatDistribution(0, 1)},
                metric=["time", "delay", "quality"],
                mode=["min", "min", "max"],
                seed=42,
                study_name="censor-test",
                storage=storage,
            )
        )

    original = search()
    original.suggest("complete")
    original.on_trial_result("complete", {"phase": "bootstrap"})
    original.on_trial_complete("complete", {"time": 10, "delay": 2, "quality": 0.9})
    original.suggest("censored")
    original.on_trial_complete(
        "censored",
        {
            "time": None,
            "delay": None,
            "quality": None,
            "search_evaluation/status": "censored",
        },
    )
    original.suggest("error")
    original.on_trial_complete(
        "error", {"time": 9, "delay": 1, "quality": 0.95}, error=True
    )
    study = optuna.load_study(study_name="censor-test", storage=storage)
    assert [t.state.name for t in study.trials] == ["COMPLETE", "PRUNED", "FAIL"]
    assert [t.number for t in study.best_trials] == [0]
    original.save(str(tmp_path / "search.pkl"))
    next_suggestion = original.suggest("next")
    restored = search()
    restored.restore(str(tmp_path / "search.pkl"))
    assert restored.suggest("next") == next_suggestion


def test_pareto_candidates_exclude_censored_and_dominated_results():
    from types import SimpleNamespace

    from smartsom.experiments.tune_search import pareto_candidates

    search = SimpleNamespace(
        objectives=[SimpleNamespace(mode=m) for m in ("min", "min", "max")]
    )

    def result(path, values):
        return SimpleNamespace(
            path=path,
            error=None,
            metrics={
                "done": True,
                **{f"search_objective_{i}": value for i, value in enumerate(values)},
            },
        )

    results = [
        result("A", [100, 10, 0.95]),
        result("B", [110, 8, 0.98]),
        result("C", [120, 12, 0.94]),
        result("D", [None, None, None]),
    ]
    assert pareto_candidates(results, search) == ["A", "B"]


def test_interrupted_optuna_search_restores_with_offline_tracking(tmp_path):
    from smartsom.experiments.tune_hpo import _read_shared_calibration
    from tests.helpers.rllib_runtime import (
        author_runtime,
        command,
        environment,
        last_checkpoint,
        wait_for,
    )

    path, profile = author_runtime(tmp_path / "inputs", ticks=200, tracking="offline")
    algorithm_file = path.parent / "algorithm.yaml"
    algorithm = yaml.safe_load(algorithm_file.read_text())
    algorithm["production"]["policies"]["ppo"]["learner"]["params"]["lr"] = 0.0003
    algorithm_file.write_text(yaml.safe_dump(algorithm))
    calibration = tmp_path / "calibration"
    command(
        "run",
        path,
        "--profile",
        profile,
        "--progress",
        "off",
        "--set",
        "experiment.run.task=calibrate",
        "--set",
        f'experiment.run.output="{calibration}"',
    )
    proof = calibration / "calibration.json"
    original_calibration = proof.read_bytes()
    search = tmp_path / "search.yaml"
    search.write_text(
        yaml.safe_dump(
            {
                "algorithm": "optuna",
                "num_samples": 3,
                "seed": 42,
                "train_seed": 101,
                "objectives": [
                    {
                        "panel": "sample",
                        "metric": f"configured.all.{metric}",
                        "mode": mode,
                    }
                    for metric, mode in (
                        ("makespan", "min"),
                        ("mean_tardiness", "min"),
                        ("final_passing_rate", "max"),
                    )
                ],
                "parameters": {
                    "algorithm.production.policies.ppo.learner.params.lr": {
                        "distribution": "uniform",
                        "lower": 0.0001,
                        "upper": 0.001,
                    }
                },
            }
        )
    )
    output = path.parent / "run"
    log = tmp_path / "search-driver.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "smartsom.experiments.cli",
                "run",
                str(path),
                "--profile",
                str(profile),
                "--search",
                str(search),
                "--max-concurrent",
                "1",
                "--progress",
                "off",
                "--set",
                f'experiment.run.calibration="{proof}"',
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment(),
        )
        try:

            def completed_trial():
                for trial in (output / "trials").glob("*"):
                    checkpoint = last_checkpoint(trial)
                    if (
                        checkpoint
                        and json.loads(
                            (checkpoint / "smartsom-runtime.json").read_text()
                        )["phase"]
                        == "complete"
                    ):
                        return trial
                return None

            completed = wait_for(completed_trial, process, log, timeout=150)
            command("pause", output)
            assert process.wait(timeout=90) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    paused = json.loads((output / "batch.json").read_text())
    assert paused["status"] == "paused"
    assert 0 < sum(t["metrics"].get("done", False) for t in paused["trials"]) < 3
    suggestions = {
        p: p.read_bytes() for p in (output / "trials").glob("*/suggestion.json")
    }
    completed_progress = (completed / "progress.json").read_bytes()
    proof.unlink()  # Restoration depends only on frozen provenance.
    command("resume", output, "--foreground", "--progress", "off", timeout=240)
    report = json.loads((output / "batch.json").read_text())
    assert report["status"] == "completed" and len(report["trials"]) == 3
    assert all(t["metrics"]["done"] and not t["error"] for t in report["trials"])
    assert all(p.read_bytes() == value for p, value in suggestions.items())
    assert (completed / "progress.json").read_bytes() == completed_progress
    trials = list((output / "trials").iterdir())
    assert len(trials) == 3
    for trial in trials:
        derived = json.loads((trial / "inputs/search-calibration.json").read_text())
        retained, _ = _read_shared_calibration(trial, derived)
        assert retained == original_calibration
        assert derived["drain_ticks"] == json.loads(original_calibration)["drain_ticks"]
        events = [
            json.loads(line)
            for p in (trial / "tracking").glob("*.jsonl")
            for line in p.read_text().splitlines()
        ]
        identities = [event["event_id"] for event in events if event.get("event_id")]
        assert identities and len(identities) == len(set(identities))
        assert list((trial / "tracking/wandb").glob("offline-run-*"))
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "retry",
            str(output),
            "--all",
            "--foreground",
            "--progress",
            "off",
        ],
        capture_output=True,
        text=True,
        env=environment(),
        timeout=30,
    )
    assert refused.returncode != 0 and "cannot be retried in place" in refused.stderr
