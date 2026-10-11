"""Execute one planned dependency/platform owner and retain fail-closed evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from selection import ROOT, load_manifest


def run_suite(
    suite_name: str,
    platform: str,
    plan_path: Path,
    *,
    sync: bool,
    results_dir: Path = Path("ci-results"),
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
    actual = {"win32": "windows-latest", "darwin": "macos-latest"}.get(
        sys.platform, "ubuntu-latest"
    )
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
        WANDB_MODE="offline",
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
    base_temp = (
        Path(env.get("RUNNER_TEMP", str(ROOT / ".ci-tmp"))) / "ci-tests" / suite_name
    )
    base_temp.parent.mkdir(parents=True, exist_ok=True)
    if platform == "windows-latest":
        # Match the existing Windows native lane: pytest owns deep checkpoint roots.
        base_temp = "\\\\?\\" + str(base_temp.resolve())
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
            "-q",
            "--durations=20",
            f"--basetemp={base_temp}",
            f"--junitxml={junit.resolve()}",
            *suite["selectors"],
        ]
        print("Required selectors: " + json.dumps(suite["selectors"]), flush=True)
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
    args = parser.parse_args()
    raise SystemExit(run_suite(args.suite, args.platform, args.plan, sync=args.sync))


if __name__ == "__main__":
    main()
