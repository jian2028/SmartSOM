"""Fresh fixed-recipe and feature acceptance from a retained, integrated checkout."""

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from validation.resource_acceptance import (
    require_integrated_source,
    require_matching_source,
)
from validation.usability_acceptance import (
    require_central_coverage,
    require_central_recipe,
    require_feature_results,
    require_frozen_training,
)

import smartsom
from smartsom import api
from smartsom.config import load_resolved_run
from smartsom.config.codec import digest, primitive
from smartsom.config.experiment import prepare
from smartsom.config.training import episode_input
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.references import protect_model_reference
from smartsom.experiments.training_audit import load_training_snapshot
from smartsom.learning.checkpoint import file_hash

ROOT = Path(__file__).resolve().parents[1]
FEATURE_TESTS = (
    "tests/integration/test_training_lifecycle.py",
    "tests/integration/test_training_sampling.py",
    "tests/integration/test_training_extensions.py",
    "tests/integration/test_training_reward_isolation.py",
    "tests/integration/test_training_probe.py",
    "tests/integration/test_experiment_api.py",
    "tests/integration/test_learning_study_integration.py",
    "tests/unit/test_training_display.py",
    "tests/unit/test_optuna_search.py",
    "tests/unit/test_extension_examples.py",
    "tests/unit/test_doctor.py",
    "tests/unit/test_report.py",
    "tests/unit/test_report_evaluation.py",
    "tests/unit/test_packaging.py",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        WANDB_MODE="offline",
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True)  # Existing evidence is never overwritten.
    logs = output / "logs"
    logs.mkdir()
    source = source_identity()
    record = {
        "schema": "smartsom.usability-acceptance/v1",
        "status": "running",
        "evidence_kind": "development" if args.development else "formal",
        "started_at": datetime.now(UTC).isoformat(),
        "source": source,
        "implementation_sha": source.get("git", {}).get("commit"),
        "checkout": str(ROOT),
        "imported_package": str(Path(smartsom.__file__).resolve()),
        "python_executable": sys.executable,
        "lock_sha256": file_hash(ROOT / "uv.lock"),
        "dependencies": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            )
        ),
        "platform_scope": "macOS CPU",
        "linux_cuda": "not_executed",
        "stages": {},
    }

    def save():
        write_json(output / "report.json", record)

    def start(name):
        print(
            json.dumps({"stage": name, "status": "running", "output": str(output)}),
            flush=True,
        )
        record["stages"][name] = {"status": "running"}
        save()

    def run(name, arguments, env=None):
        start(name)
        log = logs / f"{name}.log"
        with log.open("x") as stream:
            result = subprocess.run(
                [sys.executable, *arguments],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["stages"][name] = {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "log": str(log),
            "log_sha256": file_hash(log),
        }
        save()
        if result.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {result.returncode}; see {log}"
            )

    try:
        if not Path(smartsom.__file__).resolve().is_relative_to(ROOT / "src"):
            raise ValueError("acceptance must import the retained checkout's source")
        if platform.system() != "Darwin" or sys.version_info[:2] != (3, 12):
            raise ValueError("this acceptance requires macOS and Python 3.12")
        commit = source["git"]["commit"]
        if not args.development:
            require_integrated_source(ROOT, source)
        missing = [p for p in FEATURE_TESTS if not (ROOT / p).is_file()]
        if missing:
            raise ValueError(f"required feature tests are missing: {missing}")
        training = {}
        for name in ("rllib", "sb3", "marl"):
            start(f"training_{name}")
            config = api.load_config(ROOT / f"configs/runs/learning_{name}.yaml")
            config.output.root = str(output / "training")
            config.output.name = f"frozen-{name}"
            require_frozen_training(name, prepare(config).resolved)
            log = logs / f"training_{name}.log"
            with (
                log.open("x") as stream,
                contextlib.redirect_stdout(stream),
                contextlib.redirect_stderr(stream),
            ):
                training[name] = api.train(config)
            result = training[name]
            if result.status != "completed":
                raise ValueError(f"frozen {name} training stopped at {result.status}")
            require_frozen_training(
                name,
                load_training_snapshot(result.training_dir / "resolved_training.json"),
            )
            protect_model_reference(result.last_checkpoint, output / "report.json")
            if not args.development:
                manifest = json.loads(
                    (result.training_dir / "manifest.json").read_text()
                )
                require_matching_source(manifest["source"], commit, f"{name} training")
            record["stages"][f"training_{name}"] = {
                "status": "passed",
                "result": primitive(result),
                "log": str(log),
                "log_sha256": file_hash(log),
            }
            save()
        run(
            "item12",
            [
                "scripts/validate_learning.py",
                "--rllib-training-dir",
                str(training["rllib"].run_dir),
                "--sb3-training-dir",
                str(training["sb3"].run_dir),
                "--output-dir",
                str(output / "item12"),
                "--workers",
                "1",
            ],
        )
        central = json.loads((output / "item12/report.json").read_text())
        rows, actual_inputs = [], []
        for row in central["evaluation"]:
            directory = Path(row["run_dir"])
            manifest = json.loads((directory / "manifest.json").read_text())
            if not args.development:
                require_matching_source(manifest["source"], commit, "item12 evaluation")
            resolved = load_resolved_run(directory / "resolved_run.yaml")
            actual_inputs.append(resolved)
            rows.append(
                {
                    "provider": resolved.algorithm.algorithm.provider,
                    "replication": resolved.study_seed_origin.replication,
                    "world_sha256": digest(episode_input(resolved)),
                    "makespan": row.get("makespan"),
                    "audit_status": row.get("status"),
                }
            )
        require_central_coverage(central, rows)
        require_central_recipe(actual_inputs)
        record["stages"]["item12"]["full_input_pairing"] = rows
        run(
            "item13",
            [
                "scripts/validate_resource_learning.py",
                "--training-dir",
                str(training["marl"].run_dir),
                "--output-dir",
                str(output / "item13"),
                "--workers",
                "1",
                *(["--development"] if args.development else []),
            ],
        )
        environment = os.environ | {
            "SMARTSOM_REQUIRE_LEARNING": "1",
            "SMARTSOM_REQUIRE_MARL": "1",
            "SMARTSOM_REQUIRE_EXTENSIONS": "1",
            "SMARTSOM_REQUIRE_TRACKING": "1",
            "SMARTSOM_REQUIRE_REPORTS": "1",
            "SMARTSOM_REQUIRE_SEARCH": "1",
            "WANDB_MODE": "offline",
            "MPLBACKEND": "Agg",
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTEST_ADDOPTS": "",
            "PYTEST_PLUGINS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        collection_path = output / "feature-collection.json"
        run(
            "feature_collection",
            [
                "-m",
                "pytest",
                "-q",
                "--collect-only",
                "-p",
                "scripts.validation.feature_collection",
                *FEATURE_TESTS,
            ],
            environment | {"SMARTSOM_FEATURE_COLLECTION": str(collection_path)},
        )
        expected_tests = json.loads(collection_path.read_text())
        if (
            not isinstance(expected_tests, list)
            or not all(isinstance(node, str) for node in expected_tests)
            or {node.split("::", 1)[0] for node in expected_tests} != set(FEATURE_TESTS)
        ):
            raise ValueError("every required feature file must collect test cases")
        record["stages"]["feature_collection"].update(
            nodeids=str(collection_path),
            sha256=file_hash(collection_path),
            tests=len(expected_tests),
        )
        save()
        run(
            "features",
            [
                "-m",
                "pytest",
                "-q",
                *FEATURE_TESTS,
                "--basetemp",
                str(output / "feature-evidence"),
                "--junitxml",
                str(output / "features.xml"),
            ],
            environment,
        )
        record["stages"]["features"].update(
            require_feature_results(output / "features.xml", expected_tests)
        )
        record["final_source"] = source_identity()
        if not args.development:
            require_matching_source(
                record["final_source"], commit, "acceptance completion"
            )
            require_integrated_source(ROOT, record["final_source"])
            record["accepted_source_sha"] = commit
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        record["error"] = f"{type(exc).__name__}: {exc}"
        for row in record["stages"].values():
            if row["status"] == "running":
                row["status"] = record["status"]
    record["finished_at"] = datetime.now(UTC).isoformat()
    save()
    (output / "report.md").write_text(
        f"# SmartSOM macOS acceptance\n\nStatus: **{record['status']}**. Evidence: {record['evidence_kind']}.\n\nImplementation: `{record['implementation_sha']}`.\n\nFixed item 12/13 recipes and retained feature evidence are recorded in `report.json`. Linux/CUDA execution was not performed.\n"
        + (f"\nFailure: {record['error']}\n" if "error" in record else "")
    )
    print(
        json.dumps({"status": record["status"], "report": str(output / "report.json")}),
        flush=True,
    )
    return (
        0
        if record["status"] == "passed"
        else 130
        if record["status"] == "interrupted"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
