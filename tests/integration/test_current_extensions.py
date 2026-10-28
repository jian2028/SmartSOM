"""Explicit extension source, schema and nonstandard input shape acceptance."""

import hashlib
from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.reservation import compile_reservation

ROOT = Path(__file__).resolve().parents[2]


def authored(tmp_path, method):
    origin = ROOT / "configs/base-diagnostic/small.yaml"
    experiment = yaml.safe_load(origin.read_text())
    for name in ("factory", "workload"):
        experiment[name] = str(origin.parent / experiment[name])
    algorithm = {
        "schema": "smartsom.base-algorithm/v1",
        "method": method,
        "epochs": 1,
        "observation": {"path": str(ROOT / "examples/current/observation.py")},
        "network": {"path": str(ROOT / "examples/current/network.py")},
    }
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    experiment["algorithm"] = "algorithm.yaml"
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    return tmp_path / "experiment.yaml"


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_custom_shape_freezes_and_continues(tmp_path, method):
    pytest.importorskip("torch")
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.experiments.reservation_runtime import episode
    from smartsom.learning.base_ppo import recorded_request, request_from_dict

    path = authored(tmp_path, method)
    plan = compile_reservation(
        path,
        sets=[
            "workload.orders=8",
            "workload.initial_orders=4",
            "algorithm.width=8",
        ],
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    path.write_text("invalid: later edit")
    _, factory, workload, options, _ = load_inputs(frozen / "experiment.yaml")
    policy = build_policy(options, 9)
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        1000,
        max_ticks=80,
        metrics_level="off",
    )
    request = policy.records[0]["request"]
    assert len(request.inputs["context"]) == 3
    assert request_from_dict(recorded_request(request)).inputs == request.inputs
    policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    checkpoint = tmp_path / "current.pt"
    policy.save(checkpoint)
    restored = build_policy(options, 9)
    restored.load(checkpoint)
    assert restored.updates == policy.updates > 0
    assert (
        options.observation.sha256
        == hashlib.sha256((frozen / "modules/observation.py").read_bytes()).hexdigest()
    )
    (frozen / "modules/observation.py").write_text("# modified frozen source")
    from smartsom.algorithms.observation import build_observation

    with pytest.raises(ValueError, match="source mismatch"):
        build_observation(options)


def test_schema_mismatch_rejected_before_run(tmp_path):
    path = authored(tmp_path, "ppo")
    algorithm = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    network = tmp_path / "network.py"
    network.write_text(
        (ROOT / "examples/current/network.py")
        .read_text()
        .replace("example.public-counts/v1", "different/v1")
    )
    algorithm["network"]["path"] = str(network)
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    with pytest.raises(ValueError, match="INPUT_SCHEMA mismatch"):
        compile_reservation(path)
