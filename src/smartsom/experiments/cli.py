"""Minimal single-run CLI; no scientific overrides or alternate engine path."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from smartsom.config import ConfigurationError, resolve_study
from smartsom.config.codec import digest, primitive
from smartsom.config.models import FactoryFile, InstanceFile
from smartsom.config.snapshots import load_run_input
from smartsom.experiments.batch import run_batch
from smartsom.experiments.evidence import write_json
from smartsom.experiments.runner import RunFailedError, run_one
from smartsom.workloads import import_fjs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smartsom")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        commands.add_parser(name).add_argument("run_config")
    commands.add_parser("plan").add_argument("study")
    batch = commands.add_parser("batch")
    batch.add_argument("study", nargs="?")
    batch.add_argument("--resume")
    batch.add_argument("--retry-failed", action="store_true")
    batch.add_argument("--workers", type=int, default=2)
    importer = commands.add_parser("import-fjs")
    importer.add_argument("input")
    importer.add_argument("--instance-id", required=True)
    importer.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-fjs":
            try:
                imported = import_fjs(args.input, instance_id=args.instance_id)
            except (ValueError, OSError) as exc:
                raise ConfigurationError(str(exc)) from exc
            factory = FactoryFile(
                schema="smartsom.factory/v1", factory=imported.factory
            )
            instance = InstanceFile(
                schema="smartsom.workload-instance/v1",
                workload=imported.workload,
                content_sha256=digest(imported.workload),
                provenance=imported.provenance,
            )
            output = Path(args.output_dir)
            output.mkdir(parents=True)  # Refuse existing destinations, even empty ones.
            (output / "factory.yaml").write_text(
                yaml.safe_dump(primitive(factory)), encoding="utf-8"
            )
            write_json(output / "workload.json", instance)
            print(
                f"imported workload_sha256={instance.content_sha256} output_dir={output}"
            )
            return 0
        if args.command == "plan":
            study = resolve_study(args.study)
            print(
                json.dumps(
                    {
                        "runs": len(study.entries),
                        "plan_sha256": study.plan_sha256,
                        "entries": [
                            {
                                "entry_id": e.entry_id,
                                "case": e.case_id,
                                "algorithm": e.algorithm_id,
                                "variant": e.variant_id,
                                "replication": e.replication,
                                "seeds": primitive(e.resolved.seeds),
                                "workload_sha256": e.resolved.workload_sha256,
                            }
                            for e in study.entries
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "batch":
            if bool(args.study) == bool(args.resume):
                raise ConfigurationError(
                    "provide exactly one STUDY or --resume STUDY_DIR"
                )
            result = run_batch(
                resolve_study(args.study) if args.study else None,
                resume=args.resume,
                workers=args.workers,
                retry_failed=args.retry_failed,
                on_progress=lambda p: print(
                    json.dumps(p, sort_keys=True), file=sys.stderr
                ),
            )
            print(
                f"study_dir={result.study_dir} completed={result.completed} failed={result.failed} pending={result.pending}"
            )
            return (
                130
                if result.interrupted
                else int(bool(result.failed or result.pending))
            )
        resolved = load_run_input(args.run_config)
        if args.command == "validate":
            print(f"valid workload_sha256={resolved.workload_sha256}")
            return 0
        result = run_one(
            resolved,
            on_progress=lambda p: print(
                json.dumps(primitive(p), sort_keys=True), file=sys.stderr
            ),
        )
    except (ConfigurationError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (RunFailedError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"completed makespan={result.simulation_result.makespan} run_dir={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
