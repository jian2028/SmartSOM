"""Native four-file validation stays framework-free and does not launch work."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartsom.config.algorithm_migration import migrate_algorithm
from smartsom.config.reservation import compile_reservation
from tests.helpers.algorithm import algorithm_options

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/examples/small/default/experiment.yaml"


@pytest.mark.parametrize("mode", [[], ["--verbose"], ["--json"]])
def test_check_display_never_imports_training_or_tracking(tmp_path, mode):
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'wandb', 'ray', 'gymnasium'}:
            raise AssertionError('check imported optional runtime: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from smartsom.experiments.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "check",
            str(PATH),
            "--output-root",
            str(tmp_path / "results"),
            "--wandb",
            "disabled",
            *mode,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert not (tmp_path / "results").exists()


def test_check_without_torch_and_cli_override(tmp_path):
    script = """
import sys
from smartsom import api
print(api.check_experiment(sys.argv[1])["files"][0]["orders"])
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
            "--json",
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
    assert json.loads(cli.stdout)["files"][0]["orders"] == 256
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
            "--json",
            str(PATH),
            "--background",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["files"][0]["execution"] == "background"


def test_plan_is_detached_from_later_file_edits(tmp_path):
    import yaml

    original = yaml.safe_load(PATH.read_text())
    for name in (
        ("scenario", "algorithm")
        if "scenario" in original
        else ("factory", "workload", "algorithm")
    ):
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

    import yaml

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector

    provider = tmp_path / "custom.py"
    provider.write_text("""class Policy:
    def choose(self, request, critic=None):
        return request.candidates[-1].identity
    def get_state(self):
        return {}
    def set_state(self, state):
        pass

def create_policy(*, seed, control="base"):
    return Policy()
create_policy.CONTRACT = "smartsom.policy/v1"
""")
    from smartsom.config.components import digest_files

    checksum = digest_files({provider.name: provider.read_bytes()})
    algorithm = tmp_path / "algorithm.yaml"
    algorithm.write_text(
        yaml.safe_dump(
            migrate_algorithm(
                {
                    "schema": "smartsom.base-algorithm/v1",
                    "method": "rule",
                    "bindings": {"dispatcher.pickup": "custom"},
                    "providers": {"custom": {"path": "custom.py"}},
                }
            )
        )
    )
    document = yaml.safe_load(PATH.read_text())
    for name in ("scenario",) if "scenario" in document else ("factory", "workload"):
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
    assert options.production.policies["custom"].implementation.sha256 == checksum
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
    Path(options.production.policies["custom"].implementation.path).write_text(
        "# changed"
    )
    with pytest.raises(ValueError, match="source mismatch"):
        build_policy(options)


def test_all_effective_rule_bindings_do_not_require_learning_checkpoint():
    from smartsom.algorithms.base_policy import DECISION_POINTS, build_policy

    bindings = {point: "rule" for point in DECISION_POINTS}
    options = algorithm_options(schema="smartsom.base-algorithm/v1", bindings=bindings)
    assert build_policy(options).learner is None
    shared = [
        "algorithm.production.bindings=" + json.dumps(bindings),
        "algorithm.production.policies.rule={kind: policy, implementation: {target: smartsom.policies.rule.create_policy}}",
        "experiment.run.methods=[configured]",
        "experiment.run.calibration=calibration.json",
    ]
    assert (
        compile_reservation(PATH, task="evaluate", sets=shared).experiment.run.task
        == "evaluate"
    )
    with pytest.raises(ValueError, match="train requires a configured"):
        compile_reservation(PATH, task="train", sets=shared)


@pytest.mark.parametrize("factor", [0.5, 2])
def test_nonunit_normal_reference_is_rejected_before_allocation(tmp_path, factor):
    from dataclasses import replace
    from decimal import Decimal

    import yaml

    from smartsom.config.base_layout import base_factory
    from smartsom.config.factory_design import save_factory_design

    factory = base_factory("small")
    machine = factory.machines[-1]
    changed = replace(
        machine,
        quality_modes=tuple(
            replace(mode, time_scale=Decimal(str(factor)))
            if mode.quality_mode_id == "normal"
            else mode
            for mode in machine.quality_modes
        ),
    )
    save_factory_design(
        tmp_path / "factory.yaml",
        replace(factory, machines=(*factory.machines[:-1], changed)),
    )
    document = yaml.safe_load(PATH.read_text())
    document.pop("scenario", None)
    document["workload"] = str(ROOT / "configs/workloads/small.yaml")
    document["factory"] = "factory.yaml"
    for name in ("workload", "algorithm"):
        document[name] = str(PATH.parent / document[name])
    output = tmp_path / "must-not-run"
    document["run"] = {"output": str(output)}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="normal reference mode requires time_scale=1"):
        compile_reservation(path)
    assert not output.exists()


def test_calibration_limit_cannot_precede_initial_drain():
    with pytest.raises(ValueError, match="max_drain must be at least"):
        compile_reservation(
            PATH, task="calibrate", sets=["experiment.run.max_drain=1999"]
        )
    assert (
        compile_reservation(
            PATH, task="calibrate", sets=["experiment.run.max_drain=2000"]
        ).experiment.run.max_drain
        == 2000
    )
