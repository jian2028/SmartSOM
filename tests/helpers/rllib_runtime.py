"""Small integration inputs derived from the maintained SL-attention recipes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]


def author_runtime(
    root, method="ppo", *, ticks=80, evaluation=True, tracking="disabled"
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = (
        ROOT / f"experiments/current/small/sl-attention-shared-{method}/experiment.yaml"
    )
    parent = yaml.safe_load(source.read_text())
    scenario_path = (source.parent / parent["scenario"]).resolve()
    scenario = yaml.safe_load(scenario_path.read_text())
    scenario["jobs"]["orders"].update(total=2, initial=1)
    scenario["jobs"]["arrivals"]["mean_interval_ticks"] = 4
    scenario["jobs"]["processing"]["base_ticks"] = dict.fromkeys(
        scenario["jobs"]["processing"]["base_ticks"], 2
    )
    (root / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    algorithm = yaml.safe_load((source.parent / parent["algorithm"]).read_text())
    policy = algorithm["production"]["policies"][algorithm["production"]["default"]]
    policy["learner"]["params"] = (
        {"train_batch_size_per_learner": 64, "minibatch_size": 32, "num_epochs": 1}
        if method == "ppo"
        else {
            "train_batch_size_per_learner": 32,
            "num_steps_sampled_before_learning_starts": 32,
        }
    )
    (root / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    document = {
        "schema": "smartsom.base-experiment/v1",
        "scenario": "scenario.yaml",
        "algorithm": "algorithm.yaml",
        "drain_frozen": True,
        "drain_ticks": 2000,
        "train_ticks": ticks,
        "train_seeds": [101],
        "periodic_evaluation": {
            "unit": "physical_ticks",
            "milestones": [ticks // 2, ticks] if evaluation else [],
            "modes": ["sample" if method == "ppo" else "argmax"],
            "sample_action_seeds": 1,
        },
        "periodic_evaluation_cases": 4,
        "test_cases": 4,
        "final_evaluation": evaluation,
        "tracking": {"mode": tracking},
        "run": {
            "task": "train",
            "methods": ["configured"],
            "seconds": 120,
            "output": str(root / "run"),
        },
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(document))
    profile = root / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "resources": {"cpus": 4},
                "rllib": {
                    "env_runners": {"num_env_runners": 0},
                    "evaluation_num_env_runners": 1,
                },
            }
        )
    )
    return path, profile


def command(*args, timeout=120):
    result = subprocess.run(
        [sys.executable, "-m", "smartsom.experiments.cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def environment():
    return dict(
        os.environ,
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        RAY_USAGE_STATS_ENABLED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="offline",
    )


def wait_for(predicate, process, log, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        if process.poll() is not None:
            raise AssertionError(Path(log).read_text())
        time.sleep(0.05)
    raise AssertionError("timed out: " + Path(log).read_text())


def last_checkpoint(root):
    references = list(Path(root).glob("checkpoint-refs-*.json"))
    if not references:
        return None
    row = json.loads(references[0].read_text())["last"]
    path = Path(row["path"])
    return path if (path / "smartsom-runtime.json").exists() else None


def children(pid):
    return [
        (p.pid, p.create_time()) for p in psutil.Process(pid).children(recursive=True)
    ]


def assert_released(identities):
    def alive(identity):
        try:
            process = psutil.Process(identity[0])
            return (
                process.create_time() == identity[1]
                and process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return False

    deadline = time.monotonic() + 15
    while any(map(alive, identities)) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(map(alive, identities)), identities
