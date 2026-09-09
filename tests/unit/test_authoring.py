"""Scenario projects keep existing input semantics and survive relocation."""

import importlib.resources
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from smartsom.config import ConfigurationError, resolve_run, resolve_training_run
from smartsom.config.authoring import (
    create_template,
    import_fjs_project,
    list_templates,
    preview_scenario,
)
from smartsom.config.codec import primitive
from smartsom.engine import Simulator
from smartsom.workloads import import_fjs

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ("minimal_jsp", "generated_fjsp", "transport_buffers", "marl_micro")


def test_templates_are_packaged_and_describe_the_existing_examples():
    listed = list_templates()
    assert tuple(row["name"] for row in listed) == TEMPLATES
    assert [row["name"] for row in listed if row["learning"]] == ["marl_micro"]
    assert all(row["description"] for row in listed)
    resource = importlib.resources.files("smartsom.config").joinpath(
        "authoring_templates.json"
    )
    templates = json.loads(resource.read_text())["templates"]
    for template in templates.values():
        for name, original in template["source_inputs"].items():
            assert template["documents"][name] == yaml.safe_load(
                (ROOT / original).read_text()
            )


@pytest.mark.parametrize("name", TEMPLATES)
def test_created_project_is_legal_portable_and_matches_its_original(
    name, tmp_path, monkeypatch
):
    templates = json.loads(
        importlib.resources.files("smartsom.config")
        .joinpath("authoring_templates.json")
        .read_text()
    )["templates"]
    original = preview_scenario(ROOT / templates[name]["source_scenario"])
    directory = create_template(name, tmp_path / "created")
    assert directory == tmp_path / "created"
    assert not (directory / "runs").exists()
    moved = tmp_path / "relocated project"
    directory.rename(moved)
    monkeypatch.chdir(tmp_path)
    preview = preview_scenario(moved)
    resolved = resolve_run(moved / "run.yaml")
    assert resolved.algorithm.algorithm.provider == "builtin.spt"
    assert resolved.run.output_root == str(moved / "runs")
    for field in ("counts", "modules", "input_sha256", "effective_seeds"):
        assert preview[field] == original[field]
    assert preview["input_sha256"]["workload"] == resolved.workload_sha256
    assert all(Path(row["path"]).is_relative_to(moved) for row in preview["sources"])
    assert all(source.path.is_relative_to(moved) for source in resolved.sources)
    for path in moved.iterdir():
        assert str(ROOT) not in path.read_text()
    if name == "marl_micro":
        monkeypatch.setattr(
            "smartsom.learning.checkpoint.require_backend", lambda p: {}
        )
        training = resolve_training_run(moved / "train.yaml")
        baseline = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
        assert training.algorithm == baseline.algorithm
        assert training.run.budget == baseline.run.budget
        assert training.run.seed == baseline.run.seed
        assert training.episode(0) == baseline.episode(0)
        assert all(source.path.is_relative_to(moved) for source in training.sources)
    else:
        assert not (moved / "train.yaml").exists()


@pytest.mark.parametrize("name", TEMPLATES)
def test_preview_does_not_instantiate_a_simulator_or_write_output(
    name, tmp_path, monkeypatch
):
    directory = create_template(name, tmp_path / "case")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("authoring preview started a simulator")

    monkeypatch.setattr(Simulator, "__init__", forbidden)
    result = preview_scenario(directory / "scenario.yaml")
    assert result["status"] == "valid" and result["simulation_executed"] is False
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert json.loads(json.dumps(result)) == result


def test_preview_materializes_generated_inputs_for_the_requested_seed(tmp_path):
    directory = create_template("generated_fjsp", tmp_path / "case")
    first = preview_scenario(directory, seed=101)
    assert first == preview_scenario(directory, seed=101)
    second = preview_scenario(directory, seed=102)
    assert first["input_sha256"]["factory"] == second["input_sha256"]["factory"]
    assert first["input_sha256"]["workload"] != second["input_sha256"]["workload"]
    assert first["counts"]["machines"] == 3
    assert first["counts"]["jobs"] == 2
    assert first["workload_source"] == "profile"


def test_preview_resolves_references_from_a_nested_declaration(tmp_path, monkeypatch):
    directory = create_template("minimal_jsp", tmp_path / "project")
    nested = directory / "declarations"
    nested.mkdir()
    path = nested / "scenario.yaml"
    scenario = yaml.safe_load((directory / "scenario.yaml").read_text())
    scenario["factory"] = "../factory.yaml"
    scenario["workload"]["path"] = "../workload.json"
    path.write_text(yaml.safe_dump(scenario))
    monkeypatch.chdir(tmp_path)
    assert (
        preview_scenario(path)["input_sha256"]
        == preview_scenario(directory)["input_sha256"]
    )
    scenario["factory"] = "missing.yaml"
    path.write_text(yaml.safe_dump(scenario))
    with pytest.raises(ConfigurationError, match="missing.yaml"):
        preview_scenario(path)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_creation_refuses_existing_targets_including_empty_ones(tmp_path, kind):
    target = tmp_path / "existing"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("preserve")
    else:
        target.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(FileExistsError):
        create_template("minimal_jsp", target)
    assert not (tmp_path / "missing-target").exists()
    if kind == "file":
        assert target.read_text() == "preserve"


def test_unknown_template_fails_before_creating_any_directory(tmp_path):
    target = tmp_path / "missing" / "case"
    with pytest.raises(ConfigurationError, match="unknown scenario template"):
        create_template("not-a-template", target)
    assert not target.parent.exists()


def test_fjs_import_produces_a_movable_complete_project_and_retains_provenance(
    tmp_path, monkeypatch
):
    source = tmp_path / "custom-instance.fjs"
    source.write_text("2 2 1.5\n2 2 1 3 2 4 1 2 2\n1 1 1 1\n")
    imported = import_fjs(source, instance_id=source.stem)
    target = import_fjs_project(source, tmp_path / "project")
    moved = tmp_path / "moved"
    target.rename(moved)
    source.unlink()
    monkeypatch.chdir(tmp_path)
    resolved = resolve_run(moved / "run.yaml")
    assert resolved.factory == imported.factory
    assert resolved.workload == imported.workload
    assert resolved.provenance == imported.provenance
    assert yaml.safe_load((moved / "workload.json").read_text())["provenance"] == (
        primitive(imported.provenance)
    )
    assert import_fjs(moved / "source.fjs", instance_id="custom-instance") == imported
    preview = preview_scenario(moved)
    assert preview["counts"] == {
        "machines": 2,
        "agvs": 0,
        "orders": 1,
        "jobs": 2,
        "operations": 3,
        "base_processing_modes": 4,
    }
    assert not any(preview["modules"].values())
    with pytest.raises(FileExistsError):
        import_fjs_project(moved / "source.fjs", moved)


@pytest.mark.parametrize("raw", [b"", b"2 2\n1 1 1 3\n", b"\xff\xfe"])
def test_invalid_fjs_never_allocates_a_project(tmp_path, raw):
    source = tmp_path / "broken.fjs"
    source.write_bytes(raw)
    target = tmp_path / "new-parent" / "project"
    with pytest.raises(ConfigurationError):
        import_fjs_project(source, target)
    assert not target.parent.exists()


def test_authoring_and_marl_preview_do_not_import_optional_frameworks(tmp_path):
    code = """
import sys
from pathlib import Path
class NoFrameworks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ray', 'torch', 'gymnasium', 'pettingzoo', 'numpy', 'ortools'}:
            raise AssertionError('optional framework imported: ' + fullname)
sys.meta_path.insert(0, NoFrameworks())
from smartsom.config.authoring import create_template, preview_scenario
directory = create_template('marl_micro', Path(sys.argv[1]) / 'project')
assert preview_scenario(directory)['counts']['agvs'] == 4
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
