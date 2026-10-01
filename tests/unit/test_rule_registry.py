"""Student rules preserve semantic actions and explicit implementation identity."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from smartsom.algorithms import rule_registry
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.algorithms.rule_registry import (
    freeze_rule,
    load_rule_modules,
    public_rule_request,
    register_rule,
    verify_rule_modules,
)
from smartsom.config.policies import PolicyFile
from smartsom.domain.production_decisions import (
    Candidate,
    DecisionRequest,
    DispatchTarget,
)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(rule_registry, "_REGISTRY", {})
    monkeypatch.setattr(rule_registry, "_MODULES", {})


@pytest.mark.parametrize(
    "role,name,parameters,filename",
    [
        ("mover", "clearance_shortest_path", {}, "agv_planner.py"),
        ("dispatcher", "nearest", {"fleet_admission": "traffic"}, "agv_dispatcher.py"),
    ],
)
def test_builtin_identity_includes_controller_helpers(
    monkeypatch, role, name, parameters, filename
):
    original = freeze_rule(role, name, parameters=parameters)
    read_bytes = Path.read_bytes

    def modified(path):
        data = read_bytes(path)
        return data + b"\n# changed helper\n" if path.name == filename else data

    monkeypatch.setattr(Path, "read_bytes", modified)
    updated = freeze_rule(role, name, parameters=parameters)
    assert original["code_sha256"] != updated["code_sha256"]
    with pytest.raises(ValueError, match="digest changed"):
        freeze_rule(
            role, name, parameters=parameters, code_sha256=original["code_sha256"]
        )


@pytest.mark.parametrize(
    "role,name,parameters",
    [
        ("machine", "spt", {"ignored": True}),
        ("dispatcher", "nearest", {"ignored": True}),
        ("dispatcher", "nearest", {"fleet_admission": "invalid"}),
        ("dispatcher", "nearest", {"fleet_admission": "traffic", "max_active": 0}),
        ("dispatcher", "nearest", {"fleet_admission": "traffic", "max_active": True}),
        ("dispatcher", "nearest", {"max_active": 2}),
        ("dispatcher", "nearest", {"work_in_progress_first": "yes"}),
    ],
)
def test_builtin_parameters_reject_unused_or_invalid_settings(role, name, parameters):
    with pytest.raises(ValueError):
        freeze_rule(role, name, parameters=parameters)
    with pytest.raises(ValueError):
        RulePolicy(role, name, parameters=parameters)


def request(role="machine"):
    actions = {
        "machine": (("job-a", "normal"), ("job-b", "fast")),
        "buffer": ("job-a", "job-b"),
        "dispatcher": (None, DispatchTarget("buffer-a", "port-a")),
        "mover": ("WAIT", "RIGHT"),
    }[role]
    return DecisionRequest(
        tick=3,
        stage="proposals",
        role=role,
        owner="entity-a",
        candidates=(
            Candidate("candidate-a", actions[0], (0.0,) * 12),
            Candidate("candidate-b", actions[1], (1.0,) * 12),
            Candidate("illegal", "not-legal", (2.0,) * 12, legal=False),
        ),
        observation={
            "tick": 3,
            "jobs": {"job-a": {"due_at": 10, "rush": True}},
            "feature_time_scale": 100.0,
        },
    )


class FirstRule:
    def __init__(self, parameters, seed):
        self.parameters = parameters

    def choose(self, request):
        return request.candidates[0].action


class CountingRule:
    def __init__(self, parameters, seed):
        self.initial = parameters.get("initial", 0)

    def reset(self):
        self.calls = self.initial

    def choose(self, request):
        self.calls += 1
        return request.candidates[self.calls % len(request.candidates)].action

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class IllegalRule(FirstRule):
    def choose(self, request):
        return "not-legal"


@pytest.mark.parametrize(
    ("role", "feature"),
    [
        ("machine", "processing_time_scaled"),
        ("buffer", "due_slack_scaled"),
        ("dispatcher", "travel_time_scaled"),
        ("mover", "distance_to_target_scaled"),
    ],
)
def test_named_features_detached_and_legal(role, feature):
    original = request(role)
    public = public_rule_request(original)
    assert len(public.candidates) == 2
    assert public.candidates[0].feature(feature) == 0.0
    assert public.candidates[1].feature(feature) == 1.0
    with pytest.raises(ValueError, match="unavailable public"):
        public.candidates[0].feature("future_arrival")
    with pytest.raises(TypeError):
        public.candidates[0].features[feature] = 123
    public.observation["tick"] = 999
    job = public.job("job-a")
    job["due_at"] = 999
    assert original.observation["tick"] == 3
    assert original.observation["jobs"]["job-a"]["due_at"] == 10
    with pytest.raises(ValueError, match="not publicly revealed"):
        public.job("future-job")
    if role in ("machine", "buffer"):
        assert public.candidates[0].feature("rush") == 1.0
    assert not hasattr(public, "step")
    assert not hasattr(public, "core")


def test_custom_rule_freeze_and_schema_with_legacy_json_preserved():
    register_rule("student.first", "2", FirstRule, roles=("machine",))
    impl = {"kind": "rule", "name": "spt", "parameters": {}}
    legacy = PolicyFile.model_validate(
        {"schema": "smartsom.policy/v1", "role": "machine", "implementation": impl}
    )
    assert legacy.model_dump(by_alias=True)["implementation"] == impl
    custom = PolicyFile.model_validate(
        {
            "schema": "smartsom.policy/v1",
            "role": "machine",
            "implementation": {"kind": "rule", "name": "student.first", "version": "2"},
        }
    )
    assert custom.implementation.version == "2"
    identity = freeze_rule("machine", "student.first", "2", {"knob": 1})
    policy = RulePolicy(
        "machine",
        "student.first",
        parameters={"knob": 1},
        version="2",
        frozen_identity=identity,
    )
    assert policy.choose(request()).action == ("job-a", "normal")
    with pytest.raises(ValueError, match="identity changed"):
        RulePolicy("machine", "student.first", version="2", frozen_identity=identity)
    with pytest.raises(ValueError, match="does not support buffer"):
        RulePolicy("buffer", "student.first", version="2")
    with pytest.raises(ValueError, match="unregistered"):
        RulePolicy("machine", "student.first", version="3")


def test_rule_registration_validation_and_duplicate():
    with pytest.raises(ValueError, match="cannot replace builtin"):
        register_rule("spt", "1", FirstRule)
    with pytest.raises(ValueError, match="not a Python import path"):
        register_rule("pkg:Factory", "1", FirstRule)
    with pytest.raises(ValueError, match="explicit nonempty version"):
        register_rule("student.first", "", FirstRule)
    with pytest.raises(ValueError, match="supported production roles"):
        register_rule("student.first", "1", FirstRule, roles=("central",))
    assert register_rule("student.first", "1", FirstRule) == register_rule(
        "student.first", "1", FirstRule
    )
    with pytest.raises(ValueError, match="already registered"):
        register_rule("student.first", "1", IllegalRule)


def test_illegal_output_and_finite_json_parameters():
    register_rule("student.illegal", "1", IllegalRule)
    with pytest.raises(ValueError, match="illegal semantic action"):
        RulePolicy("machine", "student.illegal").choose(request())
    with pytest.raises(ValueError):
        freeze_rule("machine", "student.illegal", parameters={"bad": float("nan")})
    with pytest.raises(TypeError):
        freeze_rule("machine", "student.illegal", parameters={"bad": object()})


def test_state_round_trip_and_reset_reject_parameter_or_code_drift():
    register_rule("student.counting", "1", CountingRule, stateful=True)
    first = RulePolicy("machine", "student.counting", parameters={"initial": 1})
    first.choose(request())
    saved = json.loads(json.dumps(first.state_dict()))
    resumed = RulePolicy("machine", "student.counting", parameters={"initial": 1})
    resumed.load_state_dict(saved)
    assert first.choose(request()).action == resumed.choose(request()).action
    assert first.state_dict() == resumed.state_dict()
    resumed.reset()
    assert resumed.state_dict()["state"] == {"calls": 1}
    other_parameters = RulePolicy(
        "machine", "student.counting", parameters={"initial": 2}
    )
    with pytest.raises(ValueError, match="different code or parameter identity"):
        other_parameters.load_state_dict(saved)
    changed = copy.deepcopy(saved)
    changed["identity"]["code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different code or parameter identity"):
        resumed.load_state_dict(changed)


def test_declared_state_protocol_required():
    register_rule("student.incomplete", "1", FirstRule, stateful=True)
    with pytest.raises(ValueError, match="declared state protocol"):
        RulePolicy("machine", "student.incomplete")


def test_builtin_behavior_rng_and_state_json_unchanged():
    policy = RulePolicy("machine", "random", seed=71)
    saved = json.loads(json.dumps(policy.state_dict()))
    assert set(saved) == {"rng"}
    expected = [policy.choose(request()).action for _ in range(10)]
    resumed = RulePolicy("machine", "random", seed=2)
    resumed.load_state_dict(saved)
    assert [resumed.choose(request()).action for _ in range(10)] == expected
    with pytest.raises(ValueError, match="builtin rule version"):
        RulePolicy("machine", "spt", version="2")


def test_explicit_module_identity_and_helper_source_drift(tmp_path, monkeypatch):
    name = "smartsom_test_student_rule"
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n")
    source = tmp_path / (name + ".py")
    source.write_text(
        "from pathlib import Path\n"
        "from smartsom.algorithms.rule_registry import register_rule\n"
        "class First:\n"
        "    def __init__(self, parameters, seed): pass\n"
        "    def choose(self, request): return request.candidates[0].action\n"
        "register_rule('student.module', '1', First, "
        "source_files=(Path(__file__).with_name('helper.py'),))\n"
    )
    monkeypatch.delitem(sys.modules, name, raising=False)
    records = load_rule_modules([name])
    identity = freeze_rule("machine", "student.module")
    assert identity["modules"] == list(records)
    assert identity["extension_module"] == name
    verify_rule_modules(records)
    verify_rule_modules(records, load=True)
    with pytest.raises(ValueError, match="missing or its identity changed"):
        verify_rule_modules([{"module": "unloaded", "source_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="implementation digest changed"):
        RulePolicy("machine", "student.module", code_sha256="0" * 64)
    helper.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source changed after loading"):
        freeze_rule("machine", "student.module")
    source.write_text(source.read_text() + "# source drift\n")
    with pytest.raises(ValueError, match="source changed"):
        verify_rule_modules(records)
    with pytest.raises(ValueError, match="source changed after loading"):
        load_rule_modules([name])


def test_module_loader_never_accepts_import_paths():
    with pytest.raises(ValueError, match="Python module name"):
        load_rule_modules(["/tmp/student.py"])
    with pytest.raises(ValueError, match="Python module name"):
        load_rule_modules(["student:factory"])


def test_rule_import_boundary_needs_no_learning_framework():
    source = Path(__file__).resolve().parents[2] / "src"
    script = """
import builtins
real_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ray', 'numpy', 'gymnasium', 'pettingzoo'}:
        raise AssertionError('optional learning import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.policies import PolicyFile
PolicyFile.model_validate({'schema': 'smartsom.policy/v1', 'role': 'machine',
    'implementation': {'kind': 'rule', 'name': 'spt'}})
RulePolicy('machine', 'spt')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changed", ["module", "package", "helper"])
def test_fresh_worker_checks_all_frozen_sources_before_import(
    tmp_path, monkeypatch, changed
):
    package_name = "frozen_student_" + uuid4().hex
    package = tmp_path / package_name
    package.mkdir()
    parent = package / "__init__.py"
    parent.write_text("# Frozen package initializer.\n")
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n")
    leaf = package / "rules.py"
    module_name = package_name + ".rules"
    rule_name = "student.frozen_" + uuid4().hex
    leaf.write_text(
        "from pathlib import Path\n"
        "from .helper import VALUE\n"
        "from smartsom.algorithms.rule_registry import register_rule\n"
        "class First:\n"
        "    def __init__(self, parameters, seed): pass\n"
        "    def choose(self, request): return request.candidates[0].action\n"
        f"register_rule({rule_name!r}, '1', First, "
        "source_files=(Path(__file__).with_name('helper.py'),))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    records = load_rule_modules([module_name])
    assert records[0]["dependencies"] == [
        {"module": package_name, "source_sha256": rule_registry._hash_file(parent)}
    ]
    assert records[0]["source_files"] == [
        {
            "relative_path": "helper.py",
            "source_sha256": rule_registry._hash_file(helper),
        }
    ]
    marker = tmp_path / "must-not-execute.txt"
    target = {"module": leaf, "package": parent, "helper": helper}[changed]
    target.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        + target.read_text()
    )
    source = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import json, sys\n"
        "from smartsom.algorithms.rule_registry import verify_rule_modules\n"
        f"records = json.loads({json.dumps(records)!r})\n"
        "try:\n"
        "    verify_rule_modules(records, load=True)\n"
        "except ValueError as exc:\n"
        f"    assert {package_name!r} not in sys.modules\n"
        f"    assert {module_name!r} not in sys.modules\n"
        "    print(str(exc))\n"
        "else:\n"
        "    raise AssertionError('changed code was accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "source changed" in result.stdout
    assert not marker.exists()


def test_unchanged_qualified_module_and_namespace_package_load_in_fresh_worker(
    tmp_path, monkeypatch
):
    name = "namespace_student_" + uuid4().hex
    package = tmp_path / name
    package.mkdir()  # Namespace package intentionally has no __init__.py.
    leaf = package / "rules.py"
    rule_name = "student.namespace_" + uuid4().hex
    leaf.write_text(
        "from smartsom.algorithms.rule_registry import register_rule\n"
        "class First:\n"
        "    def __init__(self, parameters, seed): pass\n"
        "    def choose(self, request): return request.candidates[0].action\n"
        f"register_rule({rule_name!r}, '1', First)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    records = load_rule_modules([name + ".rules"])
    assert records[0]["dependencies"] == []
    source = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import json\n"
        "from smartsom.algorithms.rule_registry import verify_rule_modules, freeze_rule\n"
        f"records = json.loads({json.dumps(records)!r})\n"
        "verify_rule_modules(records, load=True)\n"
        f"assert freeze_rule('machine', {rule_name!r})['modules'] == records\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # Adding a namespace package initializer changes the import execution graph.
    (package / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "package dependencies changed" in result.stderr
    assert "RuntimeError: must not import" not in result.stderr


def test_factory_helper_package_is_pinned_before_its_initializer_executes(
    tmp_path, monkeypatch
):
    suffix = uuid4().hex
    module_name, helper_name = "entry_" + suffix, "provider_" + suffix
    helper = tmp_path / helper_name
    helper.mkdir()
    parent = helper / "__init__.py"
    parent.write_text("# Helper package.\n")
    (helper / "factory.py").write_text(
        "class First:\n"
        "    def __init__(self, parameters, seed): pass\n"
        "    def choose(self, request): return request.candidates[0].action\n"
    )
    leaf = tmp_path / (module_name + ".py")
    leaf.write_text(
        f"from {helper_name}.factory import First\n"
        "from smartsom.algorithms.rule_registry import register_rule\n"
        f"register_rule('student.factory_{suffix}', '1', First)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    records = load_rule_modules([module_name])
    assert {item["module"] for item in records[0]["dependencies"]} == {
        helper_name,
        helper_name + ".factory",
    }
    assert records[0]["source_files"][0]["relative_path"] == helper_name + "/factory.py"
    marker = tmp_path / "helper-must-not-execute.txt"
    parent.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    source = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import json, sys\n"
        "from smartsom.algorithms.rule_registry import verify_rule_modules\n"
        f"records = json.loads({json.dumps(records)!r})\n"
        "try:\n"
        "    verify_rule_modules(records, load=True)\n"
        "except ValueError as exc:\n"
        f"    assert {module_name!r} not in sys.modules\n"
        f"    assert {helper_name!r} not in sys.modules\n"
        "    print(str(exc))\n"
        "else:\n"
        "    raise AssertionError('changed helper package was accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "source changed" in result.stdout
    assert not marker.exists()
