"""Minimal single-run CLI; no scientific overrides or alternate engine path."""

import argparse
import sys

from smartsom.config import ConfigurationError, resolve_run
from smartsom.experiments.runner import RunFailedError, run_one


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smartsom")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        commands.add_parser(name).add_argument("run_config")
    args = parser.parse_args(argv)
    try:
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
