"""Execute one planned dependency/platform owner and retain fail-closed evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from selection import ROOT, load_manifest


def basetemp_path(root: Path, suite_name: str) -> Path:
    # Create the shared temporary parent before pytest allocates its fixture root.
    base_temp = root / "ci-tests" / suite_name
    base_temp.parent.mkdir(parents=True, exist_ok=True)
    return base_temp


def actual_platform() -> str:
    platforms = {"linux": "ubuntu-latest", "darwin": "macos-latest"}
    if sys.platform not in platforms:
        raise ValueError(
            "CI requires Linux or macOS Python; use Ubuntu under WSL2 on Windows hardware"
        )
    return platforms[sys.platform]


def isolated_pytest(
    prefix: list[str],
    selectors: list[str],
    env: dict,
    base_temp: Path,
    junit: Path,
    report_path: Path,
) -> int:
    """Release framework state between full selectors, preserving all accounting."""
    aggregate = {
        key: 0
        for key in (
            "collected",
            "executed",
            "passed",
            "skipped",
            "allowed_skips",
            "unexpected_skips",
            "deselected",
        )
    }
    aggregate.update(
        selected_nodeids=[],
        passed_nodeids=[],
        skip_reasons={},
        failed_test_paths={},
        collection_skips=[],
        missing_selectors=[],
        errors=[],
        scopes=[],
    )
    combined = ET.Element("testsuites", name="pytest tests")
    base_temp.mkdir(parents=True, exist_ok=True)
    code = 0
    for index, selector in enumerate(selectors):
        scope = Path(
            tempfile.mkdtemp(
                prefix=f"{report_path.stem}-scope-{index:03}-", dir=report_path.parent
            )
        )
        # The gate recursively loads *.json owner results. Keep internal scope
        # accounting distinct so it cannot masquerade as another owner record.
        scope_report, scope_junit = scope / "coverage.tmp", scope / "junit.xml"
        scope_env = dict(env)
        scope_env.update(
            SMARTSOM_CI_SELECTORS=json.dumps([selector]),
            SMARTSOM_CI_REPORT=str(scope_report.resolve()),
        )
        scope_code = subprocess.run(
            [
                *prefix,
                f"--basetemp={base_temp / scope.name}",
                f"--junitxml={scope_junit.resolve()}",
                selector,
            ],
            cwd=ROOT,
            env=scope_env,
            check=False,
        ).returncode
        code = max(code, int(scope_code != 0))
        if not scope_report.is_file() or not scope_junit.is_file():
            aggregate["errors"].append(f"{selector}: missing coverage/JUnit")
            code = 1
            continue
        try:
            coverage = json.loads(scope_report.read_text())
            tree = ET.parse(scope_junit).getroot()
            suites = [tree] if tree.tag == "testsuite" else list(tree)
            if tree.tag not in {"testsuite", "testsuites"} or not suites:
                raise ValueError("empty/invalid JUnit")
            if len(list(tree.iter("testcase"))) != coverage["collected"]:
                raise ValueError("JUnit/coverage test count mismatch")
            if set(aggregate["selected_nodeids"]) & set(coverage["selected_nodeids"]):
                raise ValueError("duplicate selected test across scopes")
            for key in (
                "collected",
                "executed",
                "passed",
                "skipped",
                "allowed_skips",
                "unexpected_skips",
                "deselected",
            ):
                aggregate[key] += coverage[key]
            for key in ("selected_nodeids", "passed_nodeids", "collection_skips"):
                aggregate[key].extend(coverage[key])
            aggregate["skip_reasons"].update(coverage["skip_reasons"])
            aggregate["failed_test_paths"].update(coverage.get("failed_test_paths", {}))
            aggregate["errors"].extend(f"{selector}: {e}" for e in coverage["errors"])
            aggregate["scopes"].append(
                {
                    "selector": selector,
                    "exit_code": scope_code,
                    "collected": coverage["collected"],
                    "executed": coverage["executed"],
                }
            )
            combined.extend(suites)
        except (KeyError, ValueError, TypeError, ET.ParseError) as exc:
            aggregate["errors"].append(f"{selector}: invalid evidence: {exc}")
            code = 1
    aggregate["missing_selectors"] = [
        selector
        for selector in selectors
        if not any(
            node == selector
            or node.startswith(selector + "::")
            or node.startswith(selector + "[")
            for node in aggregate["selected_nodeids"]
        )
    ]
    if aggregate["missing_selectors"] or not aggregate["executed"]:
        aggregate["errors"].append("missing selectors or no executed tests")
    if aggregate["errors"]:
        code = 1
    ET.ElementTree(combined).write(junit, encoding="utf-8", xml_declaration=True)
    report_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return code


def run_suite(
    suite_name: str,
    platform: str,
    plan_path: Path,
    *,
    sync: bool,
    results_dir: Path = Path("ci-results"),
    isolate_selectors: bool = False,
    qualify_host: bool = False,
) -> int:
    manifest = load_manifest()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    suite = manifest["suites"][suite_name]
    selected = plan["suites"][suite_name]
    if (
        platform not in suite["platforms"]
        or selected["selectors"] != suite["selectors"]
    ):
        raise ValueError("plan differs from checked-out ownership manifest")
    actual = actual_platform()
    if platform != actual:
        raise ValueError(f"requested {platform} but running on {actual}")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / f"{suite_name}-{platform}.json"
    report_path = results_dir / f"{suite_name}-{platform}.coverage.tmp"
    junit = results_dir / f"{suite_name}-{platform}.xml"
    result = {
        "suite": suite_name,
        "platform": platform,
        "selectors": suite["selectors"],
        "status": "failed",
        "exit_code": 1,
        "reasons": selected["reasons"],
    }
    if not selected["selected"]:
        result.update(status="not_required", exit_code=0)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{suite_name}/{platform}: intentionally not required by impact plan")
        return 0
    code = 1
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    env.update(
        PYTHONUTF8="1",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        WANDB_MODE="disabled",
        SMARTSOM_CI_PLATFORM=platform,
        SMARTSOM_CI_SELECTORS=json.dumps(suite["selectors"]),
        SMARTSOM_CI_ALLOWED_SKIPS=json.dumps(suite["allow_skips"]),
        SMARTSOM_CI_REPORT=str(report_path.resolve()),
    )
    # Preserve repository-required dependency switches as well as generic skip accounting.
    flags = {
        "gym": "GYM",
        "learning": "LEARNING",
        "reports": "REPORTS",
        "search": "SEARCH",
        "wandb": "TRACKING",
        "learning-marl": "MARL",
    }
    for extra, suffix in flags.items():
        if extra in suite["extras"]:
            env[f"SMARTSOM_REQUIRE_{suffix}"] = "1"
    if suite_name == "resource-learning":
        env["SMARTSOM_REQUIRE_RESOURCE_PROTOCOL"] = "1"
        env["SMARTSOM_REQUIRE_MARL"] = "1"
    if suite_name == "usability":
        env["SMARTSOM_REQUIRE_EXTENSIONS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(ROOT / "scripts/ci"), str(ROOT / "src"), env.get("PYTHONPATH", "")],
        )
    )
    # External filtering/addopts would silently weaken the committed selector contract.
    env.pop("PYTEST_ADDOPTS", None)
    base_temp = basetemp_path(
        Path(env.get("RUNNER_TEMP", str(ROOT / ".ci-tmp"))), suite_name
    )
    try:
        if sync:
            command = ["uv", "sync", "--locked"]
            for extra in suite["extras"]:
                command.extend(["--extra", extra])
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        preflight = "import importlib; " + "; ".join(
            f"importlib.import_module({name!r})" for name in suite["requires"]
        )
        subprocess.run(
            ["uv", "run", "--no-sync", "python", "-c", preflight],
            cwd=ROOT,
            env=env,
            check=True,
        )
        if suite_name == "usability" and shutil.which("node") is None:
            raise RuntimeError(
                "report browser-contract tests require Node; missing Node may not skip"
            )
        if suite_name == "fast":
            boundary = "import importlib.util; assert all(importlib.util.find_spec(m) is None for m in ('ray','torch','gymnasium','PySide6','sb3_contrib')), 'fast requires a truly lean environment'"
            subprocess.run(
                ["uv", "run", "--no-sync", "python", "-c", boundary],
                cwd=ROOT,
                env=env,
                check=True,
            )
            for args in (["ruff", "check", "."], ["ruff", "format", "--check", "."]):
                subprocess.run(
                    ["uv", "run", "--no-sync", *args], cwd=ROOT, env=env, check=True
                )
        if qualify_host:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "scripts/ci/host_ready.py",
                    *(
                        ["--mode", "performance", "--required-cpus", "2.25"]
                        if suite_name == "tuning"
                        else []
                    ),
                    "--output",
                    str(
                        (
                            results_dir / f"{suite_name}-{platform}-host-readiness.log"
                        ).resolve()
                    ),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "pytest",
            "-p",
            "pytest_plugin",
            "--strict-markers",
            "-q" if suite["tier"] == "fast" else "-v",
            "--durations=20",
        ]
        print("Required selectors: " + json.dumps(suite["selectors"]), flush=True)
        if isolate_selectors:
            code = isolated_pytest(
                command, suite["selectors"], env, base_temp, junit, report_path
            )
        else:
            command.extend(
                [
                    f"--basetemp={base_temp}",
                    f"--junitxml={junit.resolve()}",
                    *suite["selectors"],
                ]
            )
            code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
        if not report_path.is_file():
            raise RuntimeError("pytest did not emit coverage accounting")
        result.update(json.loads(report_path.read_text(encoding="utf-8")))
        if result.get("errors"):
            code = 1
        result.update(status="passed" if code == 0 else "failed", exit_code=code)
    except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
        result["error"] = str(exc)
        code = 1
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if report_path.exists():
            report_path.unlink()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"selectors", "selected_nodeids", "passed_nodeids"}
                },
                indent=2,
            )
        )
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(
                    f"\n{suite_name}/{platform}: {result['status']}; collected={result.get('collected', 0)}, executed={result.get('executed', 0)}, skipped={result.get('skipped', 0)}, deselected={result.get('deselected', 0)}\n"
                )
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--plan", type=Path, default=Path("ci-plan.json"))
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--isolate-selectors", action="store_true")
    parser.add_argument("--qualify-host", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_suite(
            args.suite,
            args.platform,
            args.plan,
            sync=args.sync,
            isolate_selectors=args.isolate_selectors,
            qualify_host=args.qualify_host,
        )
    )


if __name__ == "__main__":
    main()
