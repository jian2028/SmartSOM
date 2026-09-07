"""Minimal single-run CLI; no scientific overrides or alternate engine path."""

import argparse
import sys
from pathlib import Path

import yaml

from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.config.models import FactoryFile, InstanceFile
from smartsom.experiments.evidence import write_json
from smartsom.experiments.runner import RunFailedError, run_one
from smartsom.workloads import import_fjs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smartsom")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        commands.add_parser(name).add_argument("run_config")
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
        resolved = resolve_run(args.run_config)
        if args.command == "validate":
            print(f"valid workload_sha256={resolved.workload_sha256}")
            return 0
        result = run_one(resolved)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (RunFailedError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"completed makespan={result.simulation_result.makespan} run_dir={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
