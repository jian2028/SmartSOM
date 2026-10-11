"""Selection infrastructure must fail closed without importing optional frameworks."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def isolated_scopes(tmp_path, monkeypatch, files, selectors=None, prefix=None):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    runner = module("runner")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for name, source in files.items():
        (tmp_path / name).write_text(source)
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("CI_SCOPE_SENTINEL", None)
    env.update(
        PYTHONPATH=str(ROOT / "scripts/ci"),
        SMARTSOM_CI_PLATFORM=runner.actual_platform(),
        SMARTSOM_CI_ALLOWED_SKIPS="{}",
    )
    junit, report = tmp_path / "combined.xml", tmp_path / "coverage.json"
    code = runner.isolated_pytest(
        prefix or [sys.executable, "-m", "pytest", "-p", "pytest_plugin", "-q"],
        selectors or list(files),
        env,
        tmp_path / "fixtures",
        junit,
        report,
    )
    return code, json.loads(report.read_text()), junit


def test_isolated_selectors_release_process_state_and_preserve_all_cases(
    tmp_path, monkeypatch
):
    import xml.etree.ElementTree as ET

    first, second = tmp_path / "first.pid", tmp_path / "second.pid"
    code, report, junit = isolated_scopes(
        tmp_path,
        monkeypatch,
        {
            "test_first.py": (
                "import os,pytest\nfrom pathlib import Path\n"
                "@pytest.mark.parametrize('value',[1,2])\n"
                "def test_allocate(value,tmp_path):\n"
                "    assert tmp_path.is_dir()\n"
                "    os.environ['CI_SCOPE_SENTINEL']='retained framework state'\n"
                f"    Path({str(first)!r}).write_text(str(os.getpid()))\n"
            ),
            "test_second.py": (
                "import os\nfrom pathlib import Path\n"
                "def test_fresh():\n"
                "    assert 'CI_SCOPE_SENTINEL' not in os.environ\n"
                f"    Path({str(second)!r}).write_text(str(os.getpid()))\n"
            ),
        },
    )
    assert code == 0 and report["errors"] == []
    assert first.read_text() != second.read_text()
    assert report["collected"] == report["executed"] == report["passed"] == 3
    assert report["missing_selectors"] == []
    assert len(report["selected_nodeids"]) == 3
    assert len(list(ET.parse(junit).getroot().iter("testcase"))) == 3
    assert list(tmp_path.rglob("*.json")) == [tmp_path / "coverage.json"]


@pytest.mark.parametrize("failure", ["assertion", "skip"])
def test_isolated_selectors_preserve_failures_and_execute_later_scopes(
    tmp_path, monkeypatch, failure
):
    marker = tmp_path / "later"
    bad = (
        "assert False"
        if failure == "assertion"
        else "pytest.skip('unexpected dependency skip')"
    )
    code, report, _ = isolated_scopes(
        tmp_path,
        monkeypatch,
        {
            "test_bad.py": f"import pytest\ndef test_bad():\n    {bad}\n",
            "test_later.py": f"from pathlib import Path\ndef test_later():\n    Path({str(marker)!r}).touch()\n",
        },
    )
    assert code == 1 and report["errors"]
    assert marker.exists() and report["passed"] == 1
    assert report["collected"] == 2 and report["missing_selectors"] == []
    if failure == "skip":
        assert report["unexpected_skips"] == report["skipped"] == 1


def test_isolated_selectors_reject_zero_exit_without_accounting(tmp_path, monkeypatch):
    code, report, _ = isolated_scopes(
        tmp_path,
        monkeypatch,
        {"test_empty.py": "def test_case(): pass\n"},
        prefix=[sys.executable, "-c", "pass"],
    )
    assert code == 1 and report["executed"] == 0
    assert report["missing_selectors"] == ["test_empty.py"]
    assert any("missing coverage/JUnit" in error for error in report["errors"])


def test_isolated_selectors_reject_double_counted_case(tmp_path, monkeypatch):
    code, report, _ = isolated_scopes(
        tmp_path,
        monkeypatch,
        {"test_case.py": "def test_case(): pass\n"},
        selectors=["test_case.py", "test_case.py::test_case"],
    )
    assert code == 1
    assert report["collected"] == report["executed"] == 1
    assert any("duplicate selected test" in error for error in report["errors"])


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/ci" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


selection = module("selection")
plugin = module("pytest_plugin")


@pytest.fixture
def manifest():
    return selection.load_manifest()


def selected(plan):
    return {name for name, suite in plan["suites"].items() if suite["selected"]}


def test_complete_primary_ownership_and_no_duplicate_online_cases(manifest):
    selection.validate_manifest(manifest, ROOT)
    lifecycle = "tests/integration/test_tuning_ray_online_lifecycle.py"
    owners = [
        name
        for name, suite in manifest["suites"].items()
        if lifecycle in suite["selectors"]
    ]
    assert owners == ["tuning"]
    assert not manifest["suites"]["fast"]["extras"]
    assert (
        "tests/unit/test_composition_results.py"
        in manifest["suites"]["usability"]["selectors"]
    )
    assert "reports" in manifest["suites"]["usability"]["extras"]


def test_docs_only_always_runs_fast(manifest):
    plan = selection.make_plan(manifest, [("M", "docs/guide.md")], "pull_request")
    assert selected(plan) == {"fast"}


@pytest.mark.parametrize(
    "path",
    [
        "src/smartsom/engine/production.py",
        "src/smartsom/config/codec.py",
        "uv.lock",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "scripts/ci/selection.py",
        "tests/helpers/probe.py",
        "tests/conftest.py",
        "configs/test/new.yaml",
        "src/smartsom/unknown.py",
        "scripts/new_tool.py",
    ],
)
def test_high_impact_and_unknown_force_full(manifest, path):
    plan = selection.make_plan(manifest, [("M", path)], "pull_request")
    assert selected(plan) == set(manifest["suites"])
    assert plan["tier"] == "full"
    assert plan["full_reasons"]


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_main_scheduled_and_manual_are_full(manifest, event):
    assert selected(selection.make_plan(manifest, [], event)) == set(manifest["suites"])


@pytest.mark.parametrize("status", ["D", "R100", "R052", "C100"])
def test_deleted_renamed_copied_force_full(manifest, status):
    plan = selection.make_plan(
        manifest, [(status, "tests/unit/test_api_workflow.py")], "pull_request"
    )
    assert selected(plan) == set(manifest["suites"])


def test_empty_or_failed_diff_never_means_no_testing(manifest):
    assert selected(selection.make_plan(manifest, [], "pull_request")) == set(
        manifest["suites"]
    )
    plan = selection.make_plan(
        manifest, [("M", "README.md")], "pull_request", diff_error="missing base"
    )
    assert selected(plan) == set(manifest["suites"])


def test_changed_test_always_forced_even_full_tier(manifest):
    changed = "tests/integration/test_tuning_ray_online_lifecycle.py"
    manifest["suites"]["tuning"]["tier"] = "full"
    plan = selection.make_plan(manifest, [("A", changed)], "pull_request")
    assert selected(plan) == set(manifest["suites"])
    assert "changed test forced" in " ".join(plan["suites"]["tuning"]["reasons"])


@pytest.mark.parametrize("filename", ["test_new.py", "new_contract_test.py"])
def test_new_unclassified_test_rejected(tmp_path, manifest, filename):
    # The repository audit fails before a new test can become a green omission.
    for suite in manifest["suites"].values():
        for selector in suite["selectors"]:
            path = tmp_path / selector.split("::")[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    (tmp_path / "tests" / filename).touch()
    with pytest.raises(ValueError, match="unowned="):
        selection.validate_manifest(manifest, tmp_path)
    plan = selection.make_plan(manifest, [("A", "tests/" + filename)], "pull_request")
    assert selected(plan) == set(manifest["suites"])


def test_duplicate_primary_owner_rejected(manifest):
    manifest["suites"]["learning"]["selectors"].append(
        manifest["suites"]["fast"]["selectors"][0]
    )
    with pytest.raises(ValueError, match="duplicate primary owner"):
        selection.validate_manifest(manifest, ROOT)


def test_public_and_tuning_changes_keep_real_backends_and_portability(manifest):
    for path in (
        "src/smartsom/experiments/cli.py",
        "src/smartsom/experiments/tuning_broker.py",
    ):
        plan = selection.make_plan(manifest, [("M", path)], "pull_request")
        assert {
            "fast",
            "learning",
            "gym",
            "tuning",
            "resource-learning",
            "usability",
            "control",
            "diagnostics",
            "macos-learning",
        } <= selected(plan)


def test_changed_portable_test_forces_every_platform_owner(manifest):
    path = "tests/unit/test_checkpoint_portability.py"
    plan = selection.make_plan(manifest, [("M", path)], "pull_request")
    owners = {
        name
        for name, suite in manifest["suites"].items()
        if any(s.split("::")[0] == path for s in suite["selectors"])
    }
    assert owners <= selected(plan)
    platforms = {
        platform
        for name in owners
        for platform in manifest["suites"][name]["platforms"]
    }
    assert platforms == selection.PLATFORMS


def gate_fixture(manifest):
    plan = selection.make_plan(manifest, [("M", "README.md")], "pull_request")
    results = []
    for name, suite in plan["suites"].items():
        for platform in suite["platforms"]:
            results.append(
                {
                    "suite": name,
                    "platform": platform,
                    "status": "passed" if suite["selected"] else "not_required",
                    "selectors": suite["selectors"],
                    "exit_code": 0,
                    "collected": 2,
                    "executed": 2,
                    "passed": 2,
                    "allowed_skips": 0,
                    "unexpected_skips": 0,
                    "deselected": 0,
                }
            )
    needs = {
        name: {"result": "success"}
        for name in set(plan["suites"]) | {"plan", "cp", "resource-protocol"}
    }
    return plan, results, needs


def test_gate_accepts_only_complete_expected_success(manifest):
    selection.validate_gate(*gate_fixture(manifest))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "cancelled",
        "failure",
        "skipped_job",
        "missing_job",
        "zero",
        "skip",
        "deselected",
        "incomplete",
        "duplicate",
        "wrong_selector",
    ],
)
def test_gate_rejects_false_green(manifest, mutation):
    plan, results, needs = gate_fixture(manifest)
    fast = next(item for item in results if item["suite"] == "fast")
    if mutation == "missing":
        results.remove(fast)
    elif mutation in {"cancelled", "failure", "skipped_job"}:
        needs["fast"]["result"] = "skipped" if mutation == "skipped_job" else mutation
    elif mutation == "missing_job":
        del needs["fast"]
    elif mutation == "zero":
        fast.update(collected=0, executed=0, passed=0)
    elif mutation == "skip":
        fast.update(unexpected_skips=1, passed=1)
    elif mutation == "deselected":
        fast["deselected"] = 1
    elif mutation == "incomplete":
        fast["passed"] = 1
    elif mutation == "duplicate":
        results.append(copy.deepcopy(fast))
    elif mutation == "wrong_selector":
        fast["selectors"] = []
    with pytest.raises(ValueError):
        selection.validate_gate(plan, results, needs)


def test_allowlist_requires_exact_node_platform_and_reason():
    exceptions = {
        "tests/test_os.py::test_windows": {
            "platforms": ["ubuntu-latest"],
            "reason": "Windows only",
        }
    }
    assert plugin.allowed_skip(
        "tests/test_os.py::test_windows[x]", "Windows only", "ubuntu-latest", exceptions
    )
    assert not plugin.allowed_skip(
        "tests/test_os.py::test_windows", "missing torch", "ubuntu-latest", exceptions
    )
    assert not plugin.allowed_skip(
        "tests/test_os.py::test_windows", "Windows only", "windows-latest", exceptions
    )
    assert not plugin.allowed_skip(
        "tests/test_os.py::test_windows_other",
        "Windows only",
        "ubuntu-latest",
        exceptions,
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("def test_pass(): assert True\n", 0),
        ("import pytest\ndef test_skip(): pytest.skip('missing native backend')\n", 1),
        (
            "import pytest\npytest.importorskip('smartsom_nonexistent_required_dependency')\n",
            1,
        ),
        ("# nothing collected\n", 1),
    ],
)
def test_real_pytest_plugin_accounting(tmp_path, source, expected):
    path = tmp_path / "test_sample.py"
    path.write_text(source)
    report = tmp_path / "coverage.json"
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "scripts/ci"),
        SMARTSOM_CI_REPORT=str(report),
        SMARTSOM_CI_PLATFORM="ubuntu-latest",
        SMARTSOM_CI_SELECTORS=json.dumps(["test_sample.py"]),
        SMARTSOM_CI_ALLOWED_SKIPS="{}",
    )
    env.pop("PYTEST_ADDOPTS", None)
    outcome = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "pytest_plugin", "-q", "test_sample.py"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert outcome.returncode == expected, outcome.stdout + outcome.stderr
    data = json.loads(report.read_text())
    if expected == 0:
        assert data["collected"] == data["executed"] == data["passed"] == 1
    else:
        assert data["errors"]


def test_runner_creates_ordinary_basetemp_parent(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    runner = module("runner")
    root = tmp_path / "missing-parent"
    result = runner.basetemp_path(root, "learning")
    assert result == root / "ci-tests" / "learning"
    assert result.parent.is_dir()


@pytest.mark.parametrize(
    "helper",
    [
        "tests/unit/test_production_runtime.py",
        "tests/integration/test_training_sampling.py",
        "tests/integration/test_author_learning.py",
    ],
)
def test_shared_test_module_changes_cover_transitive_platform_consumers(
    manifest, helper
):
    plan = selection.make_plan(manifest, [("M", helper)], "pull_request")
    assert selected(plan) == set(manifest["suites"])
    assert any("cross-suite helpers" in reason for reason in plan["full_reasons"])


@pytest.mark.parametrize(
    "options",
    [
        {"testpaths": ["tests", "acceptance"]},
        {"python_files": ["check_*.py"]},
    ],
)
def test_changed_pytest_discovery_requires_explicit_inventory_update(options):
    with pytest.raises(ValueError, match="pytest discovery changed"):
        selection.validate_discovery({"tool": {"pytest": {"ini_options": options}}})


@pytest.mark.parametrize(
    "filename",
    [
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
        "pytest.toml",
        ".pytest.toml",
    ],
)
def test_alternate_pytest_config_cannot_bypass_inventory(tmp_path, manifest, filename):
    (tmp_path / filename).write_text("# alternate discovery")
    with pytest.raises(ValueError, match="alternate pytest configuration"):
        selection.validate_manifest(manifest, tmp_path)


@pytest.mark.parametrize(
    "host,expected", [("linux", "ubuntu-latest"), ("darwin", "macos-latest")]
)
def test_runner_requires_actual_supported_python(monkeypatch, host, expected):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    runner = module("runner")
    monkeypatch.setattr(runner.sys, "platform", host)
    assert runner.actual_platform() == expected


@pytest.mark.parametrize("host", ["win32", "freebsd14"])
def test_runner_rejects_unsupported_python(monkeypatch, host):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    runner = module("runner")
    monkeypatch.setattr(runner.sys, "platform", host)
    with pytest.raises(ValueError, match="Linux or macOS Python"):
        runner.actual_platform()


def test_manifest_rejects_native_windows_platform(manifest):
    manifest["suites"]["control"]["platforms"].append("windows-latest")
    with pytest.raises(ValueError, match="unknown platform"):
        selection.validate_manifest(manifest, ROOT)


def test_diagnostic_staging_never_follows_checkpoint_links(tmp_path):
    staging = module("stage_diagnostics")
    source = tmp_path / "source"
    source.mkdir()
    (source / "batch.json").write_text("{}")
    (source / "loop").symlink_to(source, target_is_directory=True)
    (source / "report.json").symlink_to(source / "batch.json")
    (source / "probe.log").write_bytes(b"x" * 33)
    result = staging.stage(source, tmp_path / "stage", max_file_bytes=32)
    assert result["files"] == ["batch.json"]
    assert result["bytes"] == 2 and result["limited"]
    assert not any(path.is_symlink() for path in (tmp_path / "stage").rglob("*"))


def test_diagnostic_staging_caps_traversal_and_publication(tmp_path):
    staging = module("stage_diagnostics")
    source = tmp_path / "source"
    source.mkdir()
    for i in range(8):
        (source / f"evaluation-{i}.log").write_bytes(b"1234")
    result = staging.stage(
        source, tmp_path / "stage", max_files=1, max_bytes=4, max_entries=3
    )
    assert len(result["files"]) == 1
    assert result["bytes"] == 4 and result["scanned"] == 3 and result["limited"]
    empty = staging.stage(tmp_path / "absent", tmp_path / "empty")
    assert empty["files"] == []
