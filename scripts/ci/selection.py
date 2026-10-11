"""Fail-closed CI impact planning and aggregate validation (stdlib only)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github/ci/test-ownership.json"
PLATFORMS = {"ubuntu-latest", "macos-latest", "windows-latest"}
FULL_PREFIXES = (
    ".github/",
    "scripts/ci/",
    "tests/helpers/",
    "src/smartsom/config/",
    "src/smartsom/domain/",
    "src/smartsom/engine/",
    "src/smartsom/modules/",
    "src/smartsom/dispatch/",
    "src/smartsom/workloads/",
    "src/smartsom/trace/",
    "configs/",
    "data/",
)
FULL_FILES = {"pyproject.toml", "uv.lock", ".python-version", "AGENTS.md"}
NATIVE_DOMAINS = {
    "learning",
    "tuning",
    "resource",
    "experiments",
    "public_api",
    "batch",
    "checkpoint",
    "control",
    "reports",
    "search",
    "tracking",
}


def load_manifest(path: Path = MANIFEST, root: Path = ROOT) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, root)
    return manifest


def validate_discovery(config: dict) -> None:
    options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    if options.get("testpaths", ["tests"]) != ["tests"] or options.get(
        "python_files", ["test_*.py", "*_test.py"]
    ) != ["test_*.py", "*_test.py"]:
        raise ValueError(
            "pytest discovery changed: update ownership inventory rules explicitly"
        )


def validate_manifest(manifest: dict, root: Path) -> None:
    alternate_configs = (
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
        "pytest.toml",
        ".pytest.toml",
    )
    if any((root / name).exists() for name in alternate_configs):
        raise ValueError(
            "alternate pytest configuration requires explicit inventory support"
        )
    project = root / "pyproject.toml"
    if project.is_file():
        validate_discovery(tomllib.loads(project.read_text(encoding="utf-8")))
    if manifest.get("version") != 1 or not manifest.get("suites"):
        raise ValueError("unsupported or empty ownership manifest")
    owners: dict[str, str] = {}
    for name, suite in manifest["suites"].items():
        for key in (
            "tier",
            "cost",
            "domains",
            "platforms",
            "extras",
            "requires",
            "selectors",
            "primary",
            "allow_skips",
        ):
            if key not in suite:
                raise ValueError(f"{name}: missing {key}")
        if suite["tier"] not in {"fast", "native", "full"}:
            raise ValueError(f"{name}: invalid tier")
        if not suite["selectors"] or not suite["platforms"]:
            raise ValueError(f"{name}: empty selectors/platforms")
        if not set(suite["platforms"]) <= PLATFORMS:
            raise ValueError(f"{name}: unknown platform")
        if suite["primary"] and "ubuntu-latest" not in suite["platforms"]:
            raise ValueError(f"{name}: primary owner must execute on Linux")
        for selector in suite["selectors"]:
            filename = selector.split("::")[0]
            path = Path(filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not filename.startswith("tests/")
            ):
                raise ValueError(f"{name}: unsafe selector {selector}")
            if not (root / filename).is_file():
                raise ValueError(f"{name}: missing test {filename}")
            if suite["primary"]:
                if "::" in selector:
                    raise ValueError("primary ownership must cover complete files")
                if filename in owners:
                    raise ValueError(f"duplicate primary owner: {filename}")
                owners[filename] = name
        for selector, exception in suite["allow_skips"].items():
            if not exception.get("reason") or not exception.get("platforms"):
                raise ValueError(f"{name}: unbounded skip exception {selector}")
            if not set(exception["platforms"]) <= set(suite["platforms"]):
                raise ValueError(f"{name}: skip exception outside suite platforms")
    tests = {
        p.relative_to(root).as_posix()
        for p in (root / "tests").rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py")
    }
    if tests != set(owners):
        raise ValueError(
            f"ownership mismatch: unowned={sorted(tests - set(owners))}, "
            f"stale={sorted(set(owners) - tests)}"
        )


def changed_paths(base: str, head: str, root: Path = ROOT) -> list[tuple[str, str]]:
    if not base or not head:
        raise ValueError("both base and head revisions are required for PR selection")
    # Git validates revisions separately; '--' prevents path/option ambiguity.
    for revision in (base, head):
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    merge_base = subprocess.check_output(
        ["git", "merge-base", base, head], cwd=root, text=True
    ).strip()
    raw = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            merge_base,
            head,
            "--",
        ],
        cwd=root,
    ).decode("utf-8")
    fields = iter(raw.split("\0"))
    result = []
    for status in fields:
        if not status:
            continue
        path = next(fields)
        result.append((status, path))
        if status.startswith(("R", "C")):
            result.append((status, next(fields)))
    return result


def make_plan(
    manifest: dict,
    changes: list[tuple[str, str]],
    event: str,
    *,
    diff_error: str | None = None,
) -> dict:
    suites = manifest["suites"]
    reasons = {name: [] for name in suites}
    full_reasons = []
    domains: set[str] = set()
    changed_tests = []
    if event != "pull_request":
        full_reasons.append(f"{event}: full verification")
    if diff_error:
        full_reasons.append(f"diff unavailable: {diff_error}")
    if event == "pull_request" and not changes:
        full_reasons.append(
            "empty/unavailable change set: conservative full verification"
        )
    for status, path in changes:
        if status.startswith(("D", "R", "C")):
            full_reasons.append(f"{status} {path}: deleted/renamed/copied impact")
        if (
            path in FULL_FILES
            or path.startswith(FULL_PREFIXES)
            or Path(path).name == "conftest.py"
        ):
            full_reasons.append(f"{path}: shared contract/dependency/selection input")
        elif path.startswith("tests/"):
            if path.endswith(".py") and (
                Path(path).name.startswith("test_")
                or Path(path).name.endswith("_test.py")
            ):
                changed_tests.append(path)
                full_reasons.append(
                    f"{path}: test modules can supply cross-suite helpers"
                )
                found = False
                for name, suite in suites.items():
                    if any(s.split("::")[0] == path for s in suite["selectors"]):
                        reasons[name].append(f"changed test forced: {path}")
                        found = True
                if not found:
                    full_reasons.append(f"{path}: unknown test ownership")
            else:
                full_reasons.append(f"{path}: shared test support")
        elif path.startswith("src/smartsom/experiments/"):
            domains.update(NATIVE_DOMAINS)
        elif path.startswith("src/smartsom/learning/"):
            domains.update(NATIVE_DOMAINS | {"studio"})
        elif path.startswith("src/smartsom/telemetry/"):
            domains.update({"reports", "search", "tracking", "control", "learning"})
        elif path.startswith("src/smartsom/studio/"):
            domains.update({"studio", "rendering", "control"})
        elif path in {"src/smartsom/_filesystem.py", "src/smartsom/api.py"}:
            domains.update(NATIVE_DOMAINS)
        elif (path.startswith("docs/") and path.endswith(".md")) or path in {
            "README.md",
            "CONTRIBUTING.md",
        }:
            pass
        else:
            full_reasons.append(f"{path}: unmapped impact")
    for name, suite in suites.items():
        if suite["tier"] == "fast":
            reasons[name].append("always-on lean contracts")
        if full_reasons:
            reasons[name].extend(full_reasons)
        elif domains.intersection(suite["domains"]):
            reasons[name].append(
                "affected domains: "
                + ", ".join(sorted(domains.intersection(suite["domains"])))
            )
    return {
        "version": 1,
        "event": event,
        "tier": "full" if full_reasons else "affected",
        "changes": [{"status": s, "path": p} for s, p in changes],
        "changed_tests": sorted(set(changed_tests)),
        "full_reasons": full_reasons,
        "suites": {
            name: {
                "selected": bool(reasons[name]),
                "reasons": reasons[name],
                "platforms": suite["platforms"],
                "selectors": suite["selectors"],
                "tier": suite["tier"],
                "cost": suite["cost"],
            }
            for name, suite in suites.items()
        },
    }


def summary(plan: dict) -> str:
    lines = [f"CI selection: {plan['tier']} ({plan['event']})"]
    for name, suite in plan["suites"].items():
        selected = "REQUIRED" if suite["selected"] else "not affected"
        lines.append(
            f"- {name}: {selected}; {suite['cost']}; " + "; ".join(suite["reasons"])
        )
        if suite["selected"]:
            lines.extend(f"  - {selector}" for selector in suite["selectors"])
    return "\n".join(lines) + "\n"


def validate_gate(plan: dict, results: list[dict], needs: dict) -> None:
    errors = []
    required_jobs = set(plan["suites"]) | {"plan", "cp", "resource-protocol"}
    if not required_jobs <= set(needs) or any(
        value.get("result") != "success" for value in needs.values()
    ):
        errors.append("a prerequisite job is missing, skipped, failed, or cancelled")
    indexed = {}
    for result in results:
        key = (result.get("suite"), result.get("platform"))
        if key in indexed:
            errors.append(f"duplicate result: {key}")
        indexed[key] = result
    expected = set()
    for name, suite in plan["suites"].items():
        for platform in suite["platforms"]:
            key = (name, platform)
            expected.add(key)
            result = indexed.get(key)
            if result is None:
                errors.append(f"missing result: {key}")
                continue
            if suite["selected"]:
                if result.get("status") != "passed" or result.get("exit_code") != 0:
                    errors.append(f"required suite did not pass: {key}")
                if result.get("selectors") != suite["selectors"]:
                    errors.append(f"selector mismatch: {key}")
                if result.get("collected", 0) <= 0 or result.get("executed", 0) <= 0:
                    errors.append(f"empty collection/execution: {key}")
                if (
                    result.get("deselected", 0) != 0
                    or result.get("unexpected_skips", 0) != 0
                ):
                    errors.append(f"lost required coverage: {key}")
                if result.get("collected") != result.get("passed", 0) + result.get(
                    "allowed_skips", 0
                ):
                    errors.append(f"incomplete selected coverage: {key}")
            elif result.get("status") != "not_required":
                errors.append(f"invalid unselected result: {key}")
    if set(indexed) != expected:
        errors.append("unexpected or missing result identities")
    if errors:
        raise ValueError("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plan", choices=["plan", "gate"])
    parser.add_argument("--event", default="workflow_dispatch")
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("ci-plan.json"))
    parser.add_argument("--plan", type=Path, default=Path("ci-plan.json"))
    parser.add_argument("--results", type=Path, default=Path("ci-results"))
    parser.add_argument("--needs-json", default="{}")
    args = parser.parse_args()
    if args.command == "gate":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        results = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in args.results.rglob("*.json")
        ]
        validate_gate(plan, results, json.loads(args.needs_json))
        print("Every planned environment/platform contract passed")
        return
    manifest = load_manifest()
    changes, error = [], None
    if args.event == "pull_request":
        try:
            changes = changed_paths(args.base, args.head)
        except (
            ValueError,
            subprocess.SubprocessError,
            UnicodeError,
            StopIteration,
        ) as exc:
            error = str(exc)
    plan = make_plan(manifest, changes, args.event, diff_error=error)
    plan["base"] = args.base
    plan["head"] = args.head
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    rendered = summary(plan)
    print(rendered)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(rendered)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for name, suite in plan["suites"].items():
                stream.write(f"{name}={str(suite['selected']).lower()}\n")


if __name__ == "__main__":
    main()
