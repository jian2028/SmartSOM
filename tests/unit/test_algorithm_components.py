"""Explicit component and action contracts, independent of optional ML libraries."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.algorithm_migration import migrate, migrate_algorithm
from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
from smartsom.config.reservation import compile_reservation
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector

ROOT = Path(__file__).resolve().parents[2]


def author(tmp_path, algorithm):
    source = ROOT / "experiments/examples/small/default/experiment.yaml"
    ex = yaml.safe_load(source.read_text())
    for key in ("factory", "workload"):
        ex[key] = str(source.parent / ex[key])
    ex["algorithm"] = "algorithm.yaml"
    ex["tracking"] = {"mode": "disabled"}
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(ex))
    return tmp_path / "experiment.yaml"


def recipe(method="rule"):
    return migrate_algorithm({"schema": "smartsom.base-algorithm/v1", "method": method})


def test_current_parser_rejects_old_schema_and_protocol_is_scientific(tmp_path):
    old = {"schema": "smartsom.base-algorithm/v1", "method": "rule"}
    path = author(tmp_path, old)
    with pytest.raises(ValueError, match="config migrate"):
        compile_reservation(path)
    path = author(tmp_path, recipe())
    base = compile_reservation(path)
    changed = compile_reservation(
        path, sets=["algorithm.decision_protocol.pre.accept.allow_stop=false"]
    )
    assert base.scientific_sha256 != changed.scientific_sha256
    assert (
        changed.component_report["traffic.mover"]
        == base.component_report["traffic.mover"]
    )
    with pytest.raises(ValueError):
        compile_reservation(
            path, sets=["algorithm.decision_protocol.pre.service.allow_wait=true"]
        )
    central = recipe()
    central["control"] = "centralized"
    with pytest.raises(ValueError, match="rejects distributed"):
        BaseAlgorithm.model_validate(central)


def test_explicit_migration_is_non_destructive_and_reports_defaults(tmp_path):
    old = tmp_path / "old.yaml"
    text = "schema: smartsom.base-algorithm/v1\nmethod: dqn\nallow_stop: false\nallow_empty_wait: true\n"
    old.write_text(text)
    report = migrate(old, tmp_path / "new")
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((tmp_path / "new/algorithm.yaml").read_text())
    )
    assert not options.decision_protocol.pre.accept.allow_stop
    assert not options.decision_protocol.post.accept.allow_stop
    assert options.allow_empty_wait
    assert (
        options.learning_policy.learner.target == "smartsom.learning.dqn.create_learner"
    )
    assert report["checkpoint_compatibility"].startswith("new experiment")
    assert old.read_text() == text
    with pytest.raises(ValueError, match="already exists"):
        migrate(old, tmp_path / "new")


def test_multifile_policy_is_frozen_and_loads_without_original_tree(tmp_path):
    bundle = tmp_path / "policy"
    bundle.mkdir()
    (bundle / "helper.py").write_text(
        "def select(request):\n    return request.candidates[-1].identity\n"
    )
    (bundle / "entry.py").write_text("""from .helper import select
class Policy:
    def choose(self, request, critic=None): return select(request)
    def get_state(self): return {}
    def set_state(self, state): pass

def create_policy(*, seed, control): return Policy()
create_policy.CONTRACT = "smartsom.policy/v1"
""")
    data = recipe()
    data["production"] = {
        "default": "custom",
        "policies": {
            "custom": {
                "kind": "policy",
                "implementation": {
                    "path": "policy/entry.py",
                    "source_root": "policy",
                    "factory": "create_policy",
                },
            }
        },
    }
    plan = compile_reservation(author(tmp_path, data))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    shutil.rmtree(bundle)
    script = """from smartsom.config.base_diagnostic import load_inputs
from smartsom.algorithms.base_policy import build_policy
import sys
_,_,_,options,_ = load_inputs(sys.argv[1])
print(build_policy(options).policies["custom"].__class__.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(frozen / "experiment.yaml")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "Policy"
    _, _, _, options, _ = load_inputs(frozen / "experiment.yaml")
    request = BaseDecision(
        0,
        "machine",
        "m",
        "process",
        (
            BaseCandidate("first", ("a",), vector()),
            BaseCandidate("last", ("b",), vector()),
        ),
        vector(),
        ((), (), (), ()),
    )
    assert build_policy(options).choose(request) == ("b",)
    spec = options.production.policies["custom"].implementation
    (Path(spec.source_root) / "helper.py").write_text("# changed dependency")
    with pytest.raises(ValueError, match="source mismatch"):
        build_policy(options)


def test_single_file_cannot_smuggle_local_imports(tmp_path):
    (tmp_path / "helper.py").write_text("value = 1\n")
    (tmp_path / "policy.py").write_text("import helper\n")
    data = recipe()
    data["production"]["policies"]["rule"]["implementation"] = {
        "path": "policy.py",
        "factory": "create_policy",
    }
    path = author(tmp_path, data)
    with pytest.raises(ValueError, match="local import"):
        compile_reservation(path)


def test_mover_state_roundtrip_and_all_production_recipes_share_it():
    from smartsom.config.base_diagnostic import generate_scenario
    from smartsom.engine.production import ProductionSimulator
    from smartsom.traffic import build_mover
    from tests.unit.test_current_runtime import inputs

    factory, workload, _ = inputs()
    states = []
    for method in ("rule", "random", "ppo", "dqn"):
        options = BaseAlgorithm.model_validate(recipe(method))
        sim = ProductionSimulator(
            generate_scenario(factory, workload, 11, "normal", 500), contract="base"
        )
        mover = build_mover(options, sim.protocol.routes)
        moves = mover.moves(sim.protocol.traffic_view(), 0)
        state = json.loads(json.dumps(mover.get_state()))
        restored = build_mover(options, sim.protocol.routes)
        restored.set_state(state)
        assert restored.get_state() == mover.get_state()
        states.append(moves)
    assert all(s == states[0] for s in states)


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("post", [False, True])
def test_stop_permissions_apply_only_to_the_selected_acceptance_stage(pre, post):
    from smartsom.experiments.reservation_runtime import episode
    from tests.unit.test_current_runtime import inputs

    factory, workload, _ = inputs()
    data = recipe()
    data["decision_protocol"]["pre"]["accept"]["allow_stop"] = pre
    data["decision_protocol"]["post"]["accept"]["allow_stop"] = post
    options = BaseAlgorithm.model_validate(data)
    seen = set()

    class Observer:
        def decision(self, request, action):
            stops = any(c.action == ("STOP",) for c in request.candidates)
            if request.stage == "accept" and request.group in {"pre", "post"}:
                seen.add(request.group)
                assert stops == {"pre": pre, "post": post}[request.group]
            else:
                assert not stops

        def transition(self, elapsed, reward):
            pass

    report, _, _ = episode(
        factory,
        workload,
        options,
        build_policy(options),
        12,
        "normal",
        500,
        max_ticks=300,
        observer=Observer(),
        metrics_level="off",
    )
    assert seen == {"pre", "post"}
    assert report["tick"] > 0


def test_migration_uses_only_effective_strategies_and_keeps_central_reward():
    from smartsom.algorithms.base_policy import DECISION_POINTS

    central = BaseAlgorithm.model_validate(
        migrate_algorithm(
            {
                "schema": "smartsom.base-algorithm/v1",
                "method": "ppo",
                "control": "centralized",
                "bindings": {"central.assign": "rule"},
                "quality_penalty": 17,
            }
        )
    )
    assert central.quality_penalty == 17
    assert central.learning_policy is None
    assert set(central.production.policies) == {"rule"}
    assert central.decision_protocol.pre is None
    override = migrate_algorithm(
        {
            "schema": "smartsom.base-algorithm/v1",
            "method": "ppo",
            "bindings": dict.fromkeys(DECISION_POINTS, "dqn"),
        }
    )
    assert override["production"]["default"] == "dqn"
    assert set(override["production"]["policies"]) == {"dqn"}


def test_migration_checks_legacy_raw_hash_before_importing(tmp_path):
    (tmp_path / "policy.py").write_text(
        "raise RuntimeError('must not import tampered source')\n"
    )
    source = tmp_path / "old.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-algorithm/v1",
                "method": "rule",
                "bindings": {"machine.process": "custom"},
                "providers": {"custom": {"path": "policy.py", "sha256": "0" * 64}},
            }
        )
    )
    with pytest.raises(ValueError, match="legacy source mismatch"):
        migrate(source, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_source_bundle_identity_includes_relative_entry(tmp_path):
    from smartsom.config.components import absolute_components, freeze_components

    bundle = tmp_path / "package"
    bundle.mkdir()
    body = """class Policy:
    def choose(self, request, critic=None): return request.candidates[INDEX].identity
    def get_state(self): return {}
    def set_state(self, state): pass

def create_policy(*, seed, control): return Policy()
create_policy.CONTRACT = "smartsom.policy/v1"
"""
    for name, index in (("a", 0), ("b", -1)):
        (bundle / f"{name}.py").write_text(body.replace("INDEX", str(index)))
    options = []
    for entry in ("a", "b"):
        data = recipe()
        data["production"]["policies"]["rule"]["implementation"] = {
            "path": f"package/{entry}.py",
            "source_root": "package",
            "factory": "create_policy",
        }
        frozen, _, _ = freeze_components(BaseAlgorithm.model_validate(data), tmp_path)
        options.append(BaseAlgorithm.model_validate(frozen))
    assert (
        options[0].production.policies["rule"].implementation.sha256
        == options[1].production.policies["rule"].implementation.sha256
    )
    assert options[0].science_identity() != options[1].science_identity()
    # Relocation changes neither the relative entry nor scientific identity.
    assert (
        options[0].science_identity()
        == absolute_components(options[0], tmp_path / "elsewhere").science_identity()
    )


def test_migrate_experiment_keeps_coupled_axes_and_paths(tmp_path):
    legacy = {"schema": "smartsom.base-algorithm/v1", "method": "ppo"}
    source = author(tmp_path, legacy)
    raw = yaml.safe_load(source.read_text())
    raw["matrix"] = {
        "algorithm.allow_stop": [True, False],
        "algorithm.learning_rate": [0.001, 0.002],
    }
    raw.setdefault("run", {})["calibration"] = "calibration/calibration.json"
    source.write_text(yaml.safe_dump(raw))
    report = migrate(source, tmp_path / "new")
    converted = yaml.safe_load((tmp_path / "new/experiment.yaml").read_text())
    axes = converted["matrix"]
    assert set(axes) == {"algorithm.production", "algorithm.decision_protocol"}
    assert len(axes["algorithm.decision_protocol"]) == 2
    for variant in axes["algorithm.decision_protocol"]:
        assert variant["pre"] == variant["post"]
    assert converted["run"]["calibration"] == str(
        tmp_path / "calibration/calibration.json"
    )
    assert report["matrix_axes"]
    from smartsom.experiments.reservation_batch import compile_members

    plans = list(compile_members([tmp_path / "new/experiment.yaml"], {}))
    assert len(plans) == 4


def test_policy_observation_projection_and_preflight_signature(tmp_path):
    data = recipe()
    (tmp_path / "observation.py").write_text("""def create_observation():
    return lambda facts, request: {"custom": request.tick}
create_observation.OUTPUT_SCHEMA = "custom/test"
""")
    data["production"]["policies"]["rule"]["observation"] = {
        "path": "observation.py",
        "factory": "create_observation",
    }
    plan = compile_reservation(author(tmp_path, data))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    _, _, _, options, _ = load_inputs(frozen / "experiment.yaml")
    policy = build_policy(options)
    request = BaseDecision(5, "machine", "m", "process", (), vector(), ((), (), (), ()))
    assert policy.observation_transform({}, request).inputs["custom"] == 5
    (tmp_path / "observation.py").write_text("""def create_observation(required): pass
create_observation.OUTPUT_SCHEMA = "custom/test"
""")
    with pytest.raises(ValueError, match="required"):
        compile_reservation(tmp_path / "experiment.yaml")


def test_migration_copies_components_enabled_only_by_matrix(tmp_path):
    shutil.copyfile(
        ROOT / "examples/current/observation.py", tmp_path / "observation.py"
    )
    (tmp_path / "network.py").write_text(
        'def create_network(*, group, options, value_head):\n    raise AssertionError("configuration checking must not construct a tensor model")\ncreate_network.INPUT_SCHEMA = "example.public-counts/v1"\ncreate_network.pack = lambda requests, critics, options: None\n'
    )
    legacy = {
        "schema": "smartsom.base-algorithm/v1",
        "method": "rule",
        "observation": {"path": "observation.py"},
        "network": {"path": "network.py"},
    }
    path = author(tmp_path, legacy)
    raw = yaml.safe_load(path.read_text())
    raw["matrix"] = {
        "algorithm.method": ["rule", "ppo"],
        "experiment.run.calibration": ["cal/a.json", "~/cal/b.json"],
    }
    raw["run"] = {"output": "~/runs/example", "checkpoint": "~/checkpoints/model.pt"}
    path.write_text(yaml.safe_dump(raw))
    migrate(path, tmp_path / "new")
    raw = yaml.safe_load((tmp_path / "new/experiment.yaml").read_text())
    assert raw["run"]["checkpoint"] == str(Path.home() / "checkpoints/model.pt")
    assert raw["run"]["output"] == str(Path.home() / "runs/example")
    assert raw["matrix"]["experiment.run.calibration"] == [
        str(tmp_path / "cal/a.json"),
        str(Path.home() / "cal/b.json"),
    ]
    (tmp_path / "observation.py").unlink()
    (tmp_path / "network.py").unlink()
    from smartsom.experiments.reservation_batch import compile_members

    plans = list(compile_members([tmp_path / "new/experiment.yaml"], {}))
    assert len(plans) == 4
    assert (
        sum(
            BaseAlgorithm.model_validate(p.summary()["algorithm"]).learning_policy
            is not None
            for _, _, p in plans
        )
        == 2
    )


def test_learning_diagnostics_weights_visits_and_separates_stages():
    from types import SimpleNamespace

    from smartsom.learning.diagnostics import UpdateDiagnostics

    def row(stage):
        return {"request": SimpleNamespace(group="dispatcher", stage=stage)}

    stats = UpdateDiagnostics()
    stats.add(
        [row("pickup"), row("pickup"), row("dropoff")],
        {"actor_loss": [1.0, 100.0, 3.0]},
        mask=[True, False, True],
    )
    stats.add([row("pickup")], {"actor_loss": [5.0]})
    stats.add(
        [row("pickup"), row("pickup"), row("dropoff")], {"critic_loss": [2.0, 8.0, 5.0]}
    )
    result = stats.summary("dispatcher")
    assert result["dispatcher"]["actor_loss"] == {"mean": 3.0, "sample_visits": 3}
    assert result["dispatcher.pickup"]["actor_loss"] == {
        "mean": 3.0,
        "sample_visits": 2,
    }
    assert result["dispatcher.dropoff"]["actor_loss"] == {
        "mean": 3.0,
        "sample_visits": 1,
    }
    assert result["dispatcher"]["critic_loss"] == {"mean": 5.0, "sample_visits": 3}
    assert stats.summary("machine") == {}
    # A finite clipped PPO loss can coexist with an overflowing likelihood ratio.
    # Preserve the diagnostic failure without inserting JSON Infinity or a fake mean.
    stats.add([row("pickup"), row("pickup")], {"approx_kl": [float("inf"), 1.0]})
    measured = stats.summary("dispatcher")["dispatcher"]["approx_kl"]
    assert measured == {"mean": None, "sample_visits": 2, "nonfinite_visits": 1}
    json.dumps(stats.summary("dispatcher"), allow_nan=False)
