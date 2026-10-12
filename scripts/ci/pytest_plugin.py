"""Account for every selected test and reject accidental optional-dependency skips."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def skip_reason(report) -> str:
    value = report.longrepr
    if isinstance(value, tuple):
        value = value[2]
    value = str(value)
    return value.removeprefix("Skipped: ")


def allowed_skip(nodeid: str, reason: str, platform: str, exceptions: dict) -> bool:
    for prefix, exception in exceptions.items():
        if (
            nodeid == prefix or nodeid.startswith(prefix + "[")
        ) and platform in exception["platforms"]:
            if reason == exception["reason"]:
                return True
    return False


class Coverage:
    def __init__(self, config):
        self.config = config
        self.selectors = json.loads(os.environ["SMARTSOM_CI_SELECTORS"])
        self.exceptions = json.loads(os.environ.get("SMARTSOM_CI_ALLOWED_SKIPS", "{}"))
        self.platform = os.environ["SMARTSOM_CI_PLATFORM"]
        self.output = Path(os.environ["SMARTSOM_CI_REPORT"])
        self.nodes = []
        self.passed = set()
        self.executed = set()
        self.skips = {}
        self.collection_skips = []
        self.failed = set()
        self.failed_test_paths = {}
        self.deselected = 0
        self.collection_errors = []

    def pytest_collection_finish(self, session):
        self.nodes = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(report.nodeid)
        elif report.skipped:
            self.collection_skips.append(
                {"nodeid": report.nodeid, "reason": skip_reason(report)}
            )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        if report.failed or getattr(report, "wasxfail", None):
            temporary = item.funcargs.get("tmp_path")
            if temporary is not None:
                self.failed_test_paths[item.nodeid] = str(temporary.resolve())
        return report

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.executed.add(report.nodeid)
            if report.passed and not getattr(report, "wasxfail", None):
                self.passed.add(report.nodeid)
        if report.failed or getattr(report, "wasxfail", None):
            self.failed.add(report.nodeid)
        if report.skipped:
            self.skips[report.nodeid] = skip_reason(report)

    def pytest_sessionfinish(self, session, exitstatus):
        missing = [
            selector
            for selector in self.selectors
            if not any(
                node == selector
                or node.startswith(selector + "::")
                or node.startswith(selector + "[")
                for node in self.nodes
            )
        ]
        allowed = {
            node
            for node, reason in self.skips.items()
            if allowed_skip(node, reason, self.platform, self.exceptions)
        }
        unexpected = set(self.skips) - allowed
        accounted = self.passed | allowed
        errors = []
        if not self.nodes or not self.executed:
            errors.append("empty collection or no executed test calls")
        if missing:
            errors.append(f"selectors collected no tests: {missing}")
        if unexpected or self.collection_skips:
            errors.append("unexpected test/module skips")
        if self.deselected or set(self.nodes) != accounted:
            errors.append(
                "selected tests were deselected or did not pass/legitimately skip"
            )
        if self.failed or self.collection_errors:
            errors.append("test/collection failures")
        report = {
            "collected": len(self.nodes),
            "executed": len(self.executed),
            "passed": len(self.passed),
            "skipped": len(self.skips) + len(self.collection_skips),
            "allowed_skips": len(allowed),
            "unexpected_skips": len(unexpected) + len(self.collection_skips),
            "deselected": self.deselected,
            "missing_selectors": missing,
            "selected_nodeids": self.nodes,
            "failed_test_paths": self.failed_test_paths,
            "passed_nodeids": sorted(self.passed),
            "skip_reasons": self.skips,
            "collection_skips": self.collection_skips,
            "errors": errors,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "\nCI coverage: "
            + json.dumps({k: v for k, v in report.items() if isinstance(v, int)})
        )
        for error in errors:
            print("CI coverage error: " + error)
        if errors:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    if os.environ.get("SMARTSOM_CI_REPORT"):
        config.pluginmanager.register(Coverage(config), "smartsom-ci-coverage")
