"""Native four-file validation stays framework-free and does not launch work."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartsom.config.reservation import compile_reservation

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "configs/base-diagnostic/small.yaml"


def test_check_without_torch_and_cli_override(tmp_path):
    script = """
import sys
from smartsom import api
print(api.check_experiment(sys.argv[1])["orders"])
assert "torch" not in sys.modules
"""
    checked = subprocess.run(
        [sys.executable, "-c", script, str(PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert checked.stdout.strip() == "128"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "check",
            str(PATH),
            "--set",
            "workload.orders=256",
            "--progress",
            "off",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(cli.stdout)["orders"] == 256
    assert not list(tmp_path.iterdir())


def test_invalid_and_unsupported_controls_are_rejected():
    with pytest.raises(ValueError):
        compile_reservation(PATH, sets=["algorithm.unknown=1"])
    with pytest.raises(ValueError, match="calibration"):
        compile_reservation(PATH, task="train")
    with pytest.raises(ValueError):
        compile_reservation(PATH, sets=["experiment.run.methods=[]"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "check",
            str(PATH),
            "--background",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["execution"] == "background"


def test_plan_is_detached_from_later_file_edits(tmp_path):
    import yaml

    original = yaml.safe_load(PATH.read_text())
    for name in ("factory", "workload", "algorithm"):
        original[name] = str(PATH.parent / original[name])
    local = tmp_path / "experiment.yaml"
    local.write_text(yaml.safe_dump(original))
    plan = compile_reservation(local)
    local.write_text("invalid: replacement")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    path = plan.materialize(frozen)
    assert compile_reservation(path).orders == 128


def test_provider_is_frozen_and_returns_semantic_identity(tmp_path):
    import hashlib

    import yaml

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector

    provider = tmp_path / "custom.py"
    provider.write_text("""class Policy:
    def choose(self, request):
        return request.candidates[-1].identity
    def get_state(self):
        return {}
    def set_state(self, state):
        pass

def create_policy(*, seed):
    return Policy()
""")
    checksum = hashlib.sha256(provider.read_bytes()).hexdigest()
    algorithm = tmp_path / "algorithm.yaml"
    algorithm.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-algorithm/v1",
                "method": "rule",
                "bindings": {"dispatcher.pickup": "custom"},
                "providers": {"custom": {"path": "custom.py"}},
            }
        )
    )
    document = yaml.safe_load(PATH.read_text())
    for name in ("factory", "workload"):
        document[name] = str(PATH.parent / document[name])
    document["algorithm"] = str(algorithm)
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(document))
    plan = compile_reservation(experiment)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    provider.write_text("raise RuntimeError('original file must not be loaded')")
    _, _, _, options, _ = load_inputs(frozen / "experiment.yaml")
    assert options.providers["custom"].sha256 == checksum
    policy = build_policy(options)
    request = BaseDecision(
        0,
        "dispatcher",
        "vehicle",
        "pickup",
        (
            BaseCandidate("request:a", ("a",), vector()),
            BaseCandidate("request:b", ("b",), vector()),
        ),
        vector(),
        ((), (), (), ()),
    )
    assert policy.choose(request) == ("b",)
    assert not policy.needs_critic(request)
    assert policy.learner is None
    (frozen / "providers/custom.py").write_text("# changed")
    with pytest.raises(ValueError, match="source mismatch"):
        build_policy(options)


def test_all_effective_rule_bindings_do_not_require_learning_checkpoint():
    from smartsom.algorithms.base_policy import DECISION_POINTS, build_policy
    from smartsom.config.base_diagnostic import BaseAlgorithm

    bindings = {point: "rule" for point in DECISION_POINTS}
    options = BaseAlgorithm(schema="smartsom.base-algorithm/v1", bindings=bindings)
    assert build_policy(options).learner is None
    shared = [
        "algorithm.bindings=" + json.dumps(bindings),
        "experiment.run.methods=[configured]",
        "experiment.run.calibration=calibration.json",
    ]
    assert (
        compile_reservation(PATH, task="evaluate", sets=shared).experiment.run.task
        == "evaluate"
    )
    with pytest.raises(ValueError, match="train requires PPO"):
        compile_reservation(PATH, task="train", sets=shared)
