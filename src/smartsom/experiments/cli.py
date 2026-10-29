"""Command line interface for one current experiment runtime."""

import argparse
import json
import sys
from pathlib import Path

from smartsom.config.codec import ConfigurationError


def _display_arguments(parser):
    parser.add_argument(
        "--verbose", nargs="?", const=1, type=int, choices=(0, 1, 2), default=None
    )
    parser.add_argument("--no-verbose", dest="verbose", action="store_const", const=0)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", choices=("auto", "on", "off"))
    parser.add_argument("--log-format", choices=("text", "json"))
    parser.add_argument("--summary-interval", type=float)
    parser.add_argument(
        "--progress-title", help="presentation title; does not rename experiment files"
    )


def _launch_arguments(command, *, advanced=False):
    command.add_argument("--profile")
    group = command.add_mutually_exclusive_group()
    group.add_argument("--foreground", dest="background", action="store_false")
    group.add_argument("--background", dest="background", action="store_true")
    command.set_defaults(background=None)
    command.add_argument("--max-concurrent", type=int)
    command.add_argument("--concurrency-mode", choices=("manual", "adaptive"))
    command.add_argument("--cpus", type=float)
    command.add_argument("--memory-gib", type=float)
    command.add_argument("--wandb", choices=("online", "offline", "disabled"))
    command.add_argument("--wandb-project")
    command.add_argument("--wandb-entity")
    command.add_argument(
        "--tensorboard", action=argparse.BooleanOptionalAction, default=None
    )
    command.add_argument("--output-root")
    _display_arguments(command)
    if advanced:
        group = command.add_argument_group(
            "Advanced scientific overrides (recorded in frozen inputs)"
        )
        group.add_argument("--set", action="append", default=[])
        group.add_argument(
            "--task", choices=("smoke", "calibrate", "train", "evaluate")
        )
        group.add_argument("--seed", type=int)
        group.add_argument("--steps", type=int)
        group.add_argument("--learning-rate", type=float)
        group.add_argument("--checkpoint")
        group.add_argument("--name")
        group.add_argument("--tune", choices=("off", "recommend", "auto"))


def _operation_overrides(args):
    out = {}
    for arg, section, key in (
        ("max_concurrent", "concurrency", "max_concurrent"),
        ("concurrency_mode", "concurrency", "mode"),
        ("cpus", "resources", "cpus"),
        ("memory_gib", "resources", "memory_gib"),
        ("wandb", "tracking", "mode"),
        ("wandb_project", "tracking", "project"),
        ("wandb_entity", "tracking", "entity"),
        ("tensorboard", "tracking", "tensorboard"),
        ("progress", "display", "progress"),
        ("log_format", "display", "log_format"),
        ("summary_interval", "display", "summary_interval"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            out.setdefault(section, {})[key] = value
    if getattr(args, "background", None) is not None:
        out["launch"] = {"mode": "background" if args.background else "foreground"}
    if getattr(args, "output_root", None) is not None:
        out["output_root"] = str(Path(args.output_root).expanduser().resolve())
    return out


def _parser():
    parser = argparse.ArgumentParser(
        prog="smartsom",
        description="One current experiment runtime; four scientific files, optional operation profile.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run", "batch-run"):
        command = commands.add_parser(name)
        command.add_argument("run_config", nargs="?")
        command.add_argument("--config")
        _launch_arguments(command, advanced=True)
        command.add_argument("--id", action="append", dest="batch_ids")
        command.add_argument("--tag", action="append", dest="batch_tags")
        command.add_argument("--keep-open", action="store_true")
    command = commands.add_parser("resume")
    command.add_argument("source")
    command.add_argument("--root", action="append")
    command.add_argument("--retry-failed", action="store_true")
    _launch_arguments(command)
    for name in (
        "status",
        "pause",
        "stop",
        "monitor",
        "attach",
        "tensorboard",
        "results",
    ):
        command = commands.add_parser(name)
        command.add_argument("source")
        command.add_argument("--root", action="append")
        if name == "stop":
            command.add_argument("--timeout", type=float, default=30)
            command.add_argument("--force", action="store_true")
        if name in {"monitor", "attach"}:
            command.add_argument("--once", action="store_true")
            _display_arguments(command)
        if name == "tensorboard":
            command.add_argument("--port", type=int, default=6006)
        if name == "results":
            command.add_argument("--algorithm")
            command.add_argument("--seed", type=int)
            command.add_argument("--status")
    command = commands.add_parser("runs")
    sub = command.add_subparsers(dest="runs_command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--root", action="append")
    show = sub.add_parser("show")
    show.add_argument("source")
    show.add_argument("--root", action="append")
    command = commands.add_parser("config")
    sub = command.add_subparsers(dest="config_command", required=True)
    migration = sub.add_parser("migrate")
    migration.add_argument("source")
    migration.add_argument("--output", required=True)
    diff = sub.add_parser("diff")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--root", action="append")
    command = commands.add_parser("completion")
    command.add_argument("shell", choices=("bash", "zsh"))
    command = commands.add_parser("_complete", help=argparse.SUPPRESS)
    command.add_argument("prefix", nargs="?", default="")
    command = commands.add_parser("studio")
    command.add_argument("paths", nargs="*", type=Path)
    command = commands.add_parser("playback")
    command.add_argument("source", type=Path)
    command.add_argument(
        "--seed", type=int, help="historical seed; inferred only when unique"
    )
    for action in (
        "add",
        "limit",
        "close",
        "pause",
        "resume",
        "remove",
        "stop",
        "status",
        "mode",
        "budget",
    ):
        command = commands.add_parser("batch-" + action)
        command.add_argument("queue", type=Path)
        command.add_argument("--command-id")
        command.add_argument("--expected-revision", type=int)
        if action in {"pause", "resume", "remove", "stop"}:
            command.add_argument("--task", dest="task_id", required=action == "remove")
        if action == "add":
            command.add_argument("directory", type=Path)
            command.add_argument("--id", action="append", dest="batch_ids")
            command.add_argument("--tag", action="append", dest="batch_tags")
        if action == "limit":
            command.add_argument("limit", type=int)
        if action == "mode":
            command.add_argument("mode", choices=("manual", "adaptive"))
        if action == "budget":
            command.add_argument("--cpus", type=float)
            command.add_argument("--memory-gib", type=float)
        if action == "remove":
            command.add_argument("--show-target", action="store_true")
            command.add_argument("--batch-id")
            command.add_argument("--confirm-target")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = None
    try:
        args = parser.parse_args(argv)
        return _dispatch(args, parser)
    except ModuleNotFoundError as exc:
        extra = {
            "PySide6": "studio",
            "torch": "learning",
            "ray": "tuning",
            "gymnasium": "gym",
            "wandb": "wandb",
            "tensorboard": "tensorboard",
        }.get((exc.name or "").split(".")[0])
        if extra is None:
            raise
        _error(args, f"optional dependency missing; install uv sync --extra {extra}")
        return 2
    except (ConfigurationError, ValueError) as exc:
        _error(args, f"configuration error: {exc}")
        return 2


def _error(args, message):
    if getattr(args, "log_format", None) == "json":
        print(json.dumps({"type": "error", "message": message}), file=sys.stderr)
    else:
        print(message, file=sys.stderr)


def _dispatch(args, parser):
    from smartsom.telemetry.runtime import display_options

    verbose = getattr(args, "verbose", None)
    options = {
        "verbose": None if verbose is None else bool(verbose),
        "legacy_verbose": verbose == 2,
        "debug": getattr(args, "debug", None),
        "progress": getattr(args, "progress", None),
        "format": getattr(args, "log_format", None),
        "every_seconds": getattr(args, "summary_interval", None),
        "title": getattr(args, "progress_title", None),
    }
    if (
        verbose == 2
        and options["debug"] is None
        and (not hasattr(args, "set") or getattr(args, "resume", None))
    ):
        options["debug"] = True
    with display_options(**options):
        return _execute_args(args, parser)


def _remove_task(args):
    from smartsom.experiments import runtime_control

    if args.command_id and not args.show_target:
        prior = (
            runtime_control.snapshot(args.queue)
            .get("commands", {})
            .get(args.command_id)
        )
        if prior is not None:
            payload = prior["payload"]
            if (
                payload.get("kind") != "remove"
                or payload.get("task_id") != args.task_id
                or payload.get("value") is not None
            ):
                raise ValueError("command ID reused with different payload")
            confirmation = payload["confirmation"]
            revision = prior["revision"] - 1
            batch_id = Path(args.queue).expanduser().resolve().name
            if (
                (
                    (not sys.stdin.isatty() or args.batch_id is not None)
                    and args.batch_id != batch_id
                )
                or (
                    (not sys.stdin.isatty() or args.confirm_target is not None)
                    and args.confirm_target != confirmation
                )
                or (
                    (not sys.stdin.isatty() or args.expected_revision is not None)
                    and args.expected_revision != revision
                )
            ):
                raise ValueError(
                    "removal retry requires the original target confirmation"
                )
            return runtime_control.submit(
                args.queue,
                "remove",
                task_id=args.task_id,
                command_id=args.command_id,
                expected_revision=revision,
                confirmation=confirmation,
            )
    target = runtime_control.removal_target(args.queue, args.task_id)
    if args.show_target:
        return target
    if sys.stdin.isatty():
        print(json.dumps(target, ensure_ascii=False, indent=2), file=sys.stderr)
        print(
            "Remove this task from future scheduling; running work waits for a safe checkpoint. Files remain. Type the exact short_id to confirm; blank input cancels:",
            file=sys.stderr,
        )
        try:
            typed = input()
        except (EOFError, KeyboardInterrupt):
            return {"status": "cancelled", "task_id": args.task_id}
        if typed != target["short_id"]:
            return {"status": "cancelled", "task_id": args.task_id}
        confirmation, revision = target["confirmation"], target["revision"]
    else:
        if (
            args.batch_id != target["batch_id"]
            or args.confirm_target != target["confirmation"]
            or args.expected_revision != target["revision"]
        ):
            raise ValueError(
                "noninteractive removal requires exact --batch-id, --task, --confirm-target and --expected-revision from --show-target"
            )
        confirmation, revision = args.confirm_target, args.expected_revision
    return runtime_control.submit(
        args.queue,
        "remove",
        task_id=args.task_id,
        command_id=args.command_id,
        expected_revision=revision,
        confirmation=confirmation,
    )


def _execute_args(args, parser):
    if args.command in {
        "status",
        "pause",
        "stop",
        "monitor",
        "attach",
        "tensorboard",
        "results",
    }:
        from smartsom.experiments.catalog import require_current, resolve_run

        args.source = (
            Path(args.source).resolve()
            if args.command == "results" and Path(args.source).is_dir()
            else resolve_run(args.source, args.root)
        )
        if args.command in {"pause", "stop", "attach"}:
            require_current(args.source)
    if args.command == "_complete":
        from smartsom.experiments.completion import candidates

        print("\n".join(candidates(args.prefix)))
        return 0
    if args.command == "completion":
        from smartsom.experiments.completion import script

        print(script(args.shell))
        return 0
    if args.command == "tensorboard":
        from smartsom.telemetry.tensorboard import serve

        return serve(args.source, args.port)
    if args.command in {"status", "runs", "results", "pause", "config"}:
        from smartsom.experiments import catalog

        if args.command == "config":
            if args.config_command == "migrate":
                from smartsom.config.algorithm_migration import migrate

                payload = migrate(args.source, args.output)
            else:
                from smartsom.experiments.launch import compare

                payload = compare(args.left, args.right, roots=args.root)
        elif args.command == "pause":
            from smartsom.experiments import control, runtime_control

            payload = (
                runtime_control.submit(args.source, "pause")
                if runtime_control.snapshot(args.source)
                else control.request_pause(args.source)
            )
        elif args.command == "results":
            payload = catalog.results(
                args.source,
                algorithm=args.algorithm,
                seed=args.seed,
                status=args.status,
            )
        elif args.command == "runs" and args.runs_command == "list":
            payload = catalog.list_runs(args.root)
        else:
            payload = catalog.describe(catalog.resolve_run(args.source, args.root))
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command in {
        "batch-add",
        "batch-limit",
        "batch-close",
        "batch-pause",
        "batch-resume",
        "batch-remove",
        "batch-stop",
        "batch-status",
        "batch-mode",
        "batch-budget",
    }:
        from smartsom.experiments import runtime_control
        from smartsom.experiments.catalog import require_current, resolve_run

        args.queue = resolve_run(args.queue)
        if args.command != "batch-status":
            require_current(args.queue)
        if args.command == "batch-add":
            payload = runtime_control.append(
                args.queue,
                args.directory,
                ids=args.batch_ids,
                tags=args.batch_tags,
                command_id=args.command_id,
                expected_revision=args.expected_revision,
            )
        elif args.command == "batch-status":
            payload = runtime_control.snapshot(args.queue)
        elif args.command == "batch-remove":
            payload = _remove_task(args)
        elif args.command == "batch-budget":
            import math

            budget = {}
            for key, value in (("cpus", args.cpus), ("memory", args.memory_gib)):
                if value is not None:
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(
                            "budget values must be positive finite numbers"
                        )
                    budget[key] = value if key == "cpus" else int(value * 1024**3)
            if not budget:
                raise ValueError("batch-budget requires --cpus or --memory-gib")
            payload = runtime_control.submit(
                args.queue,
                "budget",
                value=budget,
                command_id=args.command_id,
                expected_revision=args.expected_revision,
            )
        else:
            payload = runtime_control.submit(
                args.queue,
                args.command.removeprefix("batch-"),
                value=getattr(args, "limit", getattr(args, "mode", None)),
                task_id=getattr(args, "task_id", None),
                command_id=getattr(args, "command_id", None),
                expected_revision=getattr(args, "expected_revision", None),
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if payload.get("status") == "rejected" else 0

    if args.command in {"check", "run", "batch-run"}:
        from smartsom.experiments.launch import compile_launch, run_launch

        if bool(args.config) == bool(args.run_config):
            raise ConfigurationError("provide one Experiment file, directory or name")
        changes = list(args.set)
        if args.steps is not None and args.task != "train":
            raise ConfigurationError("--steps requires --task train")
        for key, target in (
            ("steps", "experiment.train_ticks"),
            ("checkpoint", "experiment.run.checkpoint"),
        ):
            if getattr(args, key) is not None:
                changes.append(target + "=" + json.dumps(getattr(args, key)))
        if args.name is not None:
            changes.append(
                "experiment.run.output="
                + json.dumps(
                    str((Path(args.output_root or "runs") / args.name).resolve())
                )
            )
        plan = compile_launch(
            args.config or args.run_config,
            ids=args.batch_ids,
            tags=args.batch_tags,
            sets=changes,
            task=args.task,
            seed=args.seed,
            tuning=args.tune,
            learning_rate=args.learning_rate,
            profile=args.profile,
            operations=_operation_overrides(args),
        )
        if args.command == "check":
            payload = plan.summary()
        else:
            print(json.dumps({"launch": plan.summary()}, ensure_ascii=False, indent=2))
            payload = run_launch(plan, keep_open=args.keep_open)
    elif args.command == "resume":
        from smartsom.api import resume

        payload = resume(
            args.source,
            background=args.background,
            retry_failed=args.retry_failed,
            roots=args.root,
            profile=args.profile,
            operations=_operation_overrides(args),
        )
    elif args.command == "stop":
        from smartsom.experiments.control import stop

        payload = stop(args.source, timeout=args.timeout, force=args.force)
    elif args.command in {"monitor", "attach"}:
        from smartsom.telemetry.monitor import monitor
        from smartsom.telemetry.runtime import OVERRIDES, DisplayOptions

        return monitor(
            args.source,
            once=args.once,
            controlling=args.command == "attach",
            options=DisplayOptions.from_value(OVERRIDES.get()),
        )
    elif args.command == "studio":
        from smartsom.studio.app import main as studio_main

        return studio_main(args.paths)
    elif args.command == "playback":
        root = Path(args.source).expanduser().resolve()
        manifest = json.loads((root / "run.json").read_text())
        if manifest.get("schema") == "smartsom.spatial-run/v1":
            seeds = sorted(
                int(p.stem.removeprefix("frames-"))
                for p in root.glob("frames-*.jsonl")
                if p.stem.removeprefix("frames-").isdecimal()
            )
            seed = args.seed
            if seed is None:
                if len(seeds) != 1:
                    raise ConfigurationError(
                        f"historical playback requires --seed; available: {seeds}"
                    )
                seed = seeds[0]
            if seed not in seeds:
                raise ConfigurationError(
                    f"historical playback seed unavailable: {seed}; available: {seeds}"
                )
            from smartsom.studio.historical_playback import historical_playback_window

            return historical_playback_window(root, seed=seed)
        if args.seed is not None:
            raise ConfigurationError("--seed is only for historical spatial recordings")
        from smartsom.studio.playback import playback_window

        return playback_window(root)
    else:
        raise ConfigurationError("historical command requires archived source")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return int(payload.get("status") in {"failed", "stopped", "partial_stopped"})


if __name__ == "__main__":
    raise SystemExit(main())
