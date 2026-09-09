"""Combined acceptance rejects incomplete providers, pairing, and recipe drift."""

import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import yaml

from smartsom.config import resolve_study, resolve_training_run
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gates(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda _: {})
    return importlib.import_module("validation.usability_acceptance")


@pytest.mark.parametrize("name", ["rllib", "sb3", "marl"])
def test_legacy_training_recipes_remain_frozen(gates, name):
    resolved = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
    gates.require_frozen_training(name, resolved)
    changed = copy.deepcopy(resolved)
    object.__setattr__(changed, "run", changed.run.model_copy(update={"seed": 999}))
    with pytest.raises(ValueError, match="frozen"):
        gates.require_frozen_training(name, changed)


@pytest.fixture
def paired(gates):
    return [
        {
            "provider": provider,
            "replication": replication,
            "world_sha256": str(replication),
            "makespan": 200,
            "audit_status": "passed",
        }
        for replication in range(5)
        for provider in sorted(gates.CENTRAL_PROVIDERS)
    ]


def test_complete_central_pairing(gates, paired):
    gates.require_central_coverage(
        {"status": "passed", "completed": 15, "failed": 0}, paired
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "rllib.ppo"),
        ("world_sha256", "drift"),
        ("makespan", float("nan")),
        ("audit_status", "failed"),
        ("replication", 5),
    ],
)
def test_false_success_is_rejected(gates, paired, field, value):
    paired[0][field] = value
    with pytest.raises(ValueError):
        gates.require_central_coverage(
            {"status": "passed", "completed": 15, "failed": 0}, paired
        )


def test_failed_batch_cannot_become_acceptance(gates, paired):
    with pytest.raises(ValueError):
        gates.require_central_coverage(
            {"status": "failed", "completed": 15, "failed": 0}, paired
        )


@pytest.mark.parametrize("child", ["skipped", "error", "failure"])
def test_required_features_cannot_pass_by_skipping(gates, tmp_path, child):
    path = tmp_path / "results.xml"
    path.write_text(
        f"<testsuites><testsuite><testcase><{child}/></testcase></testsuite></testsuites>"
    )
    with pytest.raises(ValueError, match="execute and pass"):
        gates.require_feature_results(path, ["test_feature.py::test_required"])


def feature_xml(path, rows):
    root = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(root, "testsuite")
    for classname, name in rows:
        ElementTree.SubElement(suite, "testcase", classname=classname, name=name)
    ElementTree.ElementTree(root).write(path, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "nodeid,identity",
    [
        ("tests/unit/test_a.py::test_a", ("tests.unit.test_a", "test_a")),
        (
            "tests/unit/test_a.py::TestGroup::test_a[param::value[slot]]",
            ("tests.unit.test_a.TestGroup", "test_a[param::value[slot]]"),
        ),
        (
            'tests/unit/test_a.py::test_a[quote"<&>]',
            ("tests.unit.test_a", 'test_a[quote"<&>]'),
        ),
    ],
)
def test_feature_xml_matches_parameterized_pytest_identity(
    gates, tmp_path, nodeid, identity
):
    path = feature_xml(tmp_path / "results.xml", [identity])
    assert gates.require_feature_results(path, [nodeid]) == {
        "tests": 1,
        "passed": 1,
        "skipped": 0,
    }


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "wrong_class"])
def test_feature_xml_requires_the_exact_collected_multiset(gates, tmp_path, change):
    expected = ["test_a.py::test_one", "test_a.py::test_two"]
    rows = [("test_a", "test_one"), ("test_a", "test_two")]
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(("test_a", "test_extra"))
    elif change == "duplicate":
        rows[1] = rows[0]
    else:
        rows[1] = ("test_b", "test_two")
    with pytest.raises(ValueError, match="coverage differs"):
        gates.require_feature_results(
            feature_xml(tmp_path / "results.xml", rows), expected
        )


def test_feature_xml_preserves_expected_collection_multiplicity(gates, tmp_path):
    expected = ["test_a.py::test_one"] * 2
    path = feature_xml(tmp_path / "results.xml", [("test_a", "test_one")] * 2)
    assert gates.require_feature_results(path, expected)["passed"] == 2
    with pytest.raises(ValueError, match="coverage differs"):
        gates.require_feature_results(path, expected[:1])


@pytest.mark.parametrize(
    "expected", [[], [None], [""], ["missing_separator"], "test_a.py::test_a"]
)
def test_feature_collection_identity_must_be_present_and_valid(
    gates, tmp_path, expected
):
    path = feature_xml(tmp_path / "results.xml", [("test_a", "test_a")])
    with pytest.raises(ValueError, match="feature"):
        gates.require_feature_results(path, expected)


def test_collection_plugin_writes_session_items_and_is_optional(
    gates, tmp_path, monkeypatch
):
    plugin = importlib.import_module("validation.feature_collection")
    expected = ["test_a.py::test_b[param::value]", "test_a.py::test_a"]
    session = SimpleNamespace(
        items=[SimpleNamespace(nodeid=nodeid) for nodeid in expected]
    )
    path = tmp_path / "collection.json"
    monkeypatch.delenv("SMARTSOM_FEATURE_COLLECTION", raising=False)
    plugin.pytest_collection_finish(session)
    assert not path.exists()
    monkeypatch.setenv("SMARTSOM_FEATURE_COLLECTION", str(path))
    plugin.pytest_collection_finish(session)
    assert json.loads(path.read_text()) == expected


def test_real_pytest_collection_and_xml_reject_deselection(gates, tmp_path):
    (tmp_path / "test_features.py").write_text(
        "import pytest\n"
        '@pytest.mark.parametrize("value", [1, 2], ids=["nested[slot]::member", "quote&<tag>"])\n'
        "def test_parameter(value):\n    assert value\n"
        "class TestGroup:\n"
        '    @pytest.mark.parametrize("value", [1], ids=["literal::value[slot]"])\n'
        "    def test_class(self, value):\n        assert value\n"
    )
    collection = tmp_path / "collection.json"
    xml = tmp_path / "results.xml"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "SMARTSOM_FEATURE_COLLECTION"}
    } | {
        "PYTHONPATH": str(ROOT / "scripts"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "SMARTSOM_FEATURE_COLLECTION": str(collection),
    }

    def run(*arguments):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--noconftest",
                "-o",
                "addopts=",
                *arguments,
                "test_features.py",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("--collect-only", "-p", "validation.feature_collection")
    expected = json.loads(collection.read_text())
    assert len(expected) == 3
    environment.pop("SMARTSOM_FEATURE_COLLECTION")
    run("--junitxml", str(xml))
    assert gates.require_feature_results(xml, expected)["passed"] == 3
    run("--junitxml", str(xml), "-k", "test_parameter")
    with pytest.raises(ValueError, match="coverage differs"):
        gates.require_feature_results(xml, expected)


@pytest.fixture
def central_runs(gates, tmp_path, monkeypatch):
    # Use the real resolver and original inputs; only checkpoint disk I/O is absent.
    monkeypatch.setattr("smartsom.learning.checkpoint.file_hash", lambda _: "a" * 64)
    monkeypatch.setattr(
        "smartsom.learning.checkpoint.validate_checkpoint", lambda _: None
    )
    path = ROOT / "configs/studies/learning_evaluation.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["cases"][0]["scenario"] = str(
        (path.parent / spec["cases"][0]["scenario"]).resolve()
    )
    filenames = {
        "RLlib-PPO": "rllib_ppo.yaml",
        "SB3-MaskablePPO": "sb3_maskable_ppo.yaml",
    }
    for row in spec["algorithms"]:
        if row["id"] == "SPT":
            row["config"] = str((path.parent / row["config"]).resolve())
            continue
        algorithm = yaml.safe_load(
            (ROOT / "configs/algorithms" / filenames[row["id"]]).read_text()
        )
        algorithm["algorithm"]["checkpoint"] = str(tmp_path / row["id"])
        target = tmp_path / f"{row['id']}.yaml"
        target.write_text(yaml.safe_dump(algorithm))
        row["config"] = str(target)
    spec["output_root"] = str(tmp_path / "output")
    target = tmp_path / "study.yaml"
    target.write_text(yaml.safe_dump(spec))
    return [entry.resolved for entry in resolve_study(target).entries]


def test_original_central_recipe_ignores_locations_weights_and_order(
    gates, central_runs
):
    gates.require_central_recipe(central_runs)
    relocated = []
    for run in central_runs:
        algorithm = run.algorithm.algorithm
        if algorithm.provider != "builtin.spt":
            algorithm = algorithm.model_copy(
                update={"checkpoint": "/relocated/model", "checkpoint_sha256": "b" * 64}
            )
        relocated.append(
            replace(
                run,
                run=run.run.model_copy(update={"output_root": "/relocated/output"}),
                algorithm=run.algorithm.model_copy(update={"algorithm": algorithm}),
            )
        )
    gates.require_central_recipe(reversed(relocated))


@pytest.mark.parametrize(
    "change", ["seed", "module", "budget", "algorithm", "world", "missing", "duplicate"]
)
def test_central_recipe_rejects_coordinated_or_individual_drift(
    gates, central_runs, change
):
    changed = list(central_runs)
    index = next(
        i
        for i, run in enumerate(changed)
        if run.algorithm.algorithm.provider == "rllib.ppo"
    )
    run = changed[index]
    if change == "seed":
        changed = [
            replace(
                row, study_seed_origin=replace(row.study_seed_origin, study_seed=203)
            )
            for row in changed
        ]
    elif change == "module":
        changed = [replace(row, buffers_enabled=False) for row in changed]
    elif change == "budget":
        changed[index] = replace(
            run,
            run=run.run.model_copy(
                update={
                    "budget": run.run.budget.model_copy(update={"max_decisions": 100})
                }
            ),
        )
    elif change == "algorithm":
        algorithm = run.algorithm.algorithm
        changed[index] = replace(
            run,
            algorithm=run.algorithm.model_copy(
                update={
                    "algorithm": algorithm.model_copy(
                        update={
                            "parameters": algorithm.parameters.model_copy(
                                update={"gamma": 0.9}
                            )
                        }
                    )
                }
            ),
        )
    elif change == "world":
        times = ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    op.operation_id,
                    mode.processing_mode_id,
                    mode.nominal_ticks,
                    mode.nominal_ticks + 1,
                )
                for op in run.workload.operations
                for mode in op.modes
            )
        )
        changed = [replace(row, processing_times=times) for row in changed]
    elif change == "missing":
        changed.pop()
    else:
        changed[-1] = changed[0]
    with pytest.raises(ValueError, match="frozen centralized"):
        gates.require_central_recipe(changed)
