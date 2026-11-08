"""Command line interface for one current experiment runtime."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from smartsom.config.codec import ConfigurationError


def _display_arguments(parser, *, check=False):
    output = parser.add_mutually_exclusive_group() if check else parser
    output.add_argument(
        "--verbose",
        nargs="?",
        const=1,
        type=int,
        choices=(0, 1),
        default=None,
        help="show resolved files, components and setting sources" if check else None,
    )
    output.add_argument("--no-verbose", dest="verbose", action="store_const", const=0)
    if check:
        output.add_argument(
            "--json",
            action="store_true",
            help="emit the complete machine-readable check result",
        )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", choices=("auto", "on", "off"))
    parser.add_argument("--log-format", choices=("text", "json"))
    parser.add_argument("--summary-interval", type=float)
    parser.add_argument(
        "--progress-title", help="presentation title; does not rename experiment files"
    )


def _launch_arguments(command, *, advanced=False, check=False):
    command.add_argument("--profile")
    group = command.add_mutually_exclusive_group()
    group.add_argument("--foreground", dest="background", action="store_false")
    group.add_argument("--background", dest="background", action="store_true")
    command.set_defaults(background=None)
    command.add_argument("--max-concurrent", type=int)
    command.add_argument(
        "--evaluation-processes",
        type=int,
        help="independent native evaluation cases per panel (default: 1)",
    )
    command.add_argument(
        "--evaluation-cpus",
        type=int,
        help="total CPU cap per experiment evaluation panel",
    )
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
    _display_arguments(command, check=check)
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
        ("evaluation_processes", "concurrency", "evaluation_processes"),
        ("evaluation_cpus", "concurrency", "evaluation_cpus"),
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
    from smartsom.experiments.batch_selection import EntryAction, SelectAction

    parser_type = partial(argparse.ArgumentParser, allow_abbrev=False)
    parser = parser_type(
        prog="smartsom",
        description="One experiment runtime; frozen scientific inputs and explicit operational controls.",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=parser_type
    )
    for name in ("check", "run"):
        command = commands.add_parser(name)
        command.set_defaults(_allow_file_select=True, file_selector=None)
        command.add_argument("run_config", nargs="?")
        command.add_argument("--config")
        _launch_arguments(command, advanced=True, check=name == "check")
        command.add_argument(
            "--id",
            action=EntryAction,
            dest="batch_ids",
            help="select entries in eligible launch order",
        )
        command.add_argument(
            "--select",
            action=SelectAction,
            dest="batch_selectors",
            help="YAML train_seeds execution subset for preceding --id (or run FILE); lists are actual seeds, integers count from 101",
        )
        command.add_argument("--tag", action="append", dest="batch_tags")
        command.add_argument("--keep-open", action="store_true")
        command.add_argument(
            "--first-train-seeds",
            type=int,
            help="execute only the first N authored training seeds; freeze the full batch",
        )
        command.add_argument(
            "--resume-missing",
            action="store_true",
            help="continue selected missing tasks in the same frozen batch",
        )
    command = commands.add_parser("resume")
    command.add_argument("source")
    command.add_argument("--root", action="append")
    command.add_argument("--task-id", dest="task_id")
    command.add_argument("--command-id")
    command.add_argument("--expected-revision", type=int)
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
            command.add_argument("--timeout", type=float)
            command.add_argument("--force", action="store_true")

        if name in {"pause", "stop", "status"}:
            command.add_argument("--task-id", dest="task_id")
            command.add_argument("--command-id")
        if name in {"pause", "stop"}:
            command.add_argument("--expected-revision", type=int)
        if name == "status":
            command.add_argument("--json", action="store_true")
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
    command.add_argument("--root", action="append")
    command = commands.add_parser("wandb")
    sub = command.add_subparsers(
        dest="wandb_command", required=True, parser_class=parser_type
    )
    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("source")
    dashboard.add_argument("--root", action="append")
    command = commands.add_parser(
        "paired-export", help="export explicit paired evaluation evidence"
    )
    command.add_argument("config", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("config")
    sub = command.add_subparsers(
        dest="config_command", required=True, parser_class=parser_type
    )
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
    for action in ("add", "close", "remove", "configure"):
        command = commands.add_parser(action)
        command.add_argument("queue", type=Path)
        command.add_argument("--command-id")
        command.add_argument("--expected-revision", type=int)
        if action in {"remove", "configure"}:
            command.add_argument(
                "--task-id", dest="task_id", required=action == "remove"
            )
        if action == "add":
            command.add_argument("directory", type=Path)
            command.add_argument("--first-train-seeds", type=int)
            command.add_argument("--output-root", type=Path)
            command.add_argument("--id", action=EntryAction, dest="batch_ids")
            command.add_argument(
                "--select", action=SelectAction, dest="batch_selectors"
            )
            command.add_argument("--tag", action="append", dest="batch_tags")
        if action == "configure":
            command.add_argument("--max-concurrent", type=int)
            command.add_argument("--concurrency-mode", choices=("manual", "adaptive"))
            command.add_argument("--evaluation-processes", type=int)
            command.add_argument("--evaluation-cpus", type=int)
            command.add_argument("--cpus", type=float)
            command.add_argument("--memory-gib", type=float)
        if action == "remove":
            command.add_argument("--show-target", action="store_true")
            command.add_argument("--batch-id")
            command.add_argument("--confirm-target")
    command = commands.add_parser(
        "retry", help="retry failed work; paused work uses resume"
    )
    command.add_argument("queue", type=Path)
    _launch_arguments(command)
    selection = command.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-id", dest="task_id", help="exact failed task ID in a live batch"
    )
    selection.add_argument(
        "--all",
        dest="all_tasks",
        action="store_true",
        help="select failed tasks in the batch",
    )
    command.add_argument(
        "--preview", action="store_true", help="inspect selection without submitting"
    )
    command = commands.add_parser(
        "clean",
        help="preview or delete a stopped run or removed task's outputs",
    )
    command.add_argument("queue", type=Path)
    command.add_argument("--root", action="append")
    command.add_argument("--task-id", dest="task_id")
    choice = command.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--show-target",
        action="store_true",
        help="preview paths, sizes and permanent deletion warning",
    )
    choice.add_argument(
        "--confirm-target", help="exact confirmation token from preview"
    )
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
    if getattr(args, "json", False) or getattr(args, "log_format", None) == "json":
        print(json.dumps({"type": "error", "message": message}), file=sys.stderr)
    else:
        print(message, file=sys.stderr)


def _dispatch(args, parser):
    from smartsom.telemetry.runtime import display_options

    verbose = getattr(args, "verbose", None)
    options = {
        "verbose": None if verbose is None else bool(verbose),
        "debug": getattr(args, "debug", None),
        "progress": getattr(args, "progress", None),
        "format": getattr(args, "log_format", None),
        "every_seconds": getattr(args, "summary_interval", None),
        "title": getattr(args, "progress_title", None),
    }
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
                "noninteractive removal requires exact --batch-id, --task-id, --confirm-target and --expected-revision from --show-target"
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


def _submit_control(args, kind):
    from smartsom.experiments import runtime_control

    if kind == "resume":
        from smartsom.experiments import control

        owner = control.read(args.source)
        if (
            not owner
            or owner.get("status") not in control.ACTIVE
            or not control.alive(owner["owner"], control.processes())
        ):
            raise ValueError(
                "batch driver is not running; resume the whole run before scoped/journal resume"
            )
    return runtime_control.submit(
        args.source,
        kind,
        value=None,
        task_id=args.task_id,
        command_id=args.command_id,
        expected_revision=args.expected_revision,
    )


def _status(args):
    from smartsom.experiments import catalog, runtime_control

    payload = catalog.describe(args.source)
    state = runtime_control.snapshot(args.source)
    payload["runtime"] = state
    if state:
        payload["status"] = state.get("status", payload["status"])
    if args.task_id:
        row = {**state.get("tasks", {}), **state.get("additions", {})}.get(args.task_id)
        if row is None:
            raise ValueError("unknown task ID")
        payload["task"] = {"id": args.task_id, **row}
    if args.command_id:
        receipt = state.get("commands", {}).get(args.command_id)
        if receipt is None:
            raise ValueError("unknown command ID")
        payload["command"] = receipt
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"{payload['id']}: {payload['status']}")
        if state:
            print(
                f"Experiments: active={state.get('active', 0)}, limit={state.get('limit')}, mode={state.get('mode')}"
            )
            print("Budget: " + json.dumps(state.get("budget", {}), ensure_ascii=False))
            print(
                "Evaluation targets: "
                + json.dumps(state.get("evaluation", {}), ensure_ascii=False)
            )
            rows = (
                {args.task_id: payload["task"]}
                if args.task_id
                else {**state.get("tasks", {}), **state.get("additions", {})}
            )
            for key, row in rows.items():
                print(
                    f"  {key}: {row.get('status', 'unknown')} | evaluation requested={row.get('evaluation_requested', row.get('evaluation_processes', '?'))} effective={row.get('evaluation_effective', '?')} live={row.get('evaluation_live', '?')} | reserved CPUs={row.get('cpus', '?')} | {row.get('blocking_reason') or ''}"
                )
        if args.command_id:
            print(json.dumps(payload["command"], ensure_ascii=False, indent=2))
    return 0


def _clean_run(args):
    from smartsom.config.codec import digest
    from smartsom.experiments import control
    from smartsom.experiments.catalog import resolve_run

    if Path(args.queue).expanduser().is_symlink():
        raise ValueError("clean requires the actual run directory, not a symlink")
    root = resolve_run(args.queue, args.root)
    control._validate_delete(root)
    owner = control.read(root)
    if not owner:
        raise ValueError("clean requires verified run ownership")
    target = {"directory": str(root), "owner_id": owner["id"]}
    target["confirmation"] = digest(target)
    if args.show_target:
        print(json.dumps(target, ensure_ascii=False, indent=2))
        return 0
    if args.confirm_target != target["confirmation"]:
        raise ValueError("clean requires the exact --confirm-target from --show-target")
    table = control.processes()
    if owner.get("status") in control.ACTIVE or any(
        control.alive(item, table)
        for item in [owner["owner"], *owner.get("members", [])]
    ):
        raise ValueError("run is not stopped; stop it and verify exit before clean")
    payload = control.stop(root, timeout=0, expected_owner_id=owner["id"], delete=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(not payload.get("deleted"))


def _execute_args(args, parser):
    if args.command == "paired-export":
        from smartsom.telemetry.paired_export import export, write_export

        payload = export(args.config)
        write_export(args.output, payload)
        print(json.dumps({"status": "exported", "output": str(args.output)}))
        return 0
    if args.command == "wandb":
        from smartsom.telemetry.wandb_dashboard import create_dashboard

        payload = create_dashboard(args.source, args.root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
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
    if args.command == "status":
        return _status(args)
    if args.command in {"runs", "results", "pause", "config"}:
        from smartsom.experiments import catalog

        if args.command == "config":
            from smartsom.experiments.launch import compare

            payload = compare(args.left, args.right, roots=args.root)
        elif args.command == "pause":
            from smartsom.experiments import control, runtime_control

            if args.task_id or runtime_control.snapshot(args.source):
                payload = _submit_control(args, "pause")
            else:
                if args.command_id or args.expected_revision is not None:
                    raise ValueError("journal options require a batch runtime")
                payload = control.request_pause(args.source)
        elif args.command == "results":
            payload = catalog.results(
                args.source,
                algorithm=args.algorithm,
                seed=args.seed,
                status=args.status,
            )
        elif args.command == "runs":
            payload = catalog.list_runs(args.root)
        else:
            payload = catalog.describe(catalog.resolve_run(args.source, args.root))
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "retry":
        from smartsom.experiments import batch_retry, runtime_control
        from smartsom.experiments.catalog import require_current, resolve_run

        queue = resolve_run(args.queue)
        require_current(queue)
        if not args.preview and not runtime_control._RETRY_ENABLED:
            raise ValueError("live retry is not yet enabled; task state was retained")
        from smartsom.experiments import control

        owner = control.read(queue)
        state = runtime_control.snapshot(queue)
        live = bool(
            owner
            and owner.get("status") in control.ACTIVE
            and control.alive(owner["owner"], control.processes())
        )
        if state and live and state.get("status") not in runtime_control.TERMINAL:
            if not args.all_tasks and not args.task_id:
                raise ValueError("batch retry requires --all or --task-id")
            if _operation_overrides(args) or args.profile:
                raise ValueError("live retry cannot change settings; use configure")
        else:
            if live:
                raise ValueError("run is still active; wait for failure recovery")
            if args.task_id:
                raise ValueError(
                    "ended batch recovery requires retry --all; exact task retry requires a live driver"
                )
            if state and not args.all_tasks:
                raise ValueError("ended batch retry requires --all")
            from smartsom.experiments.catalog import read_json

            failed = any(
                read_json(queue / name).get("status") == "failed"
                for name in ("batch.json", "progress.json")
            )
            if state:
                failed = failed or any(
                    job.get("status") == "failed"
                    for job in {
                        **state.get("tasks", {}),
                        **state.get("additions", {}),
                    }.values()
                )
            if args.preview or not failed:
                print(
                    json.dumps(
                        {
                            "directory": str(queue),
                            "status": "preview" if args.preview else "no_failed_tasks",
                            "failed": failed,
                        }
                    )
                )
                return 0
            from smartsom.api import resume

            payload = resume(
                queue,
                background=args.background,
                retry_failed=True,
                profile=args.profile,
                operations=_operation_overrides(args),
                interactive=sys.stdin.isatty() and sys.stdout.isatty(),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return int(
                payload.get("status") in {"failed", "stopped", "partial_stopped"}
            )
        preview = batch_retry.preview_failed(
            queue, [args.task_id] if args.task_id else None
        )
        if args.preview:
            payload = preview
        else:
            print(
                json.dumps({"preview": preview}, ensure_ascii=False, indent=2),
                file=sys.stderr,
                flush=True,
            )
            payload = batch_retry.submit_preview(queue, preview)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return (
            1
            if any(
                item.get("status") == "rejected" for item in payload.get("outcomes", [])
            )
            else 0
        )
    if args.command == "clean":
        from smartsom.experiments import batch_cleanup
        from smartsom.experiments.catalog import require_current, resolve_run

        if not args.task_id:
            return _clean_run(args)
        queue = resolve_run(args.queue, args.root)
        require_current(queue)
        payload = (
            batch_cleanup.preview(queue, args.task_id)
            if args.show_target
            else batch_cleanup.delete(queue, args.task_id, args.confirm_target)
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command in {"add", "close", "remove", "configure"}:
        from smartsom.experiments import runtime_control
        from smartsom.experiments.catalog import require_current, resolve_run

        args.queue = resolve_run(args.queue)
        require_current(args.queue)
        if args.command == "add":
            payload = runtime_control.append(
                args.queue,
                args.directory,
                first_train_seeds=args.first_train_seeds,
                output_root=args.output_root,
                ids=args.batch_ids,
                selectors=args.batch_selectors,
                tags=args.batch_tags,
                command_id=args.command_id,
                expected_revision=args.expected_revision,
            )
        elif args.command == "remove":
            payload = _remove_task(args)
        else:
            value = None
            if args.command == "configure":
                value = {
                    key: getattr(args, key)
                    for key in runtime_control.CONFIGURE_FIELDS
                    if getattr(args, key, None) is not None
                }
                runtime_control.validate_configuration(value, args.task_id)
            payload = runtime_control.submit(
                args.queue,
                args.command,
                value=value,
                task_id=getattr(args, "task_id", None),
                command_id=args.command_id,
                expected_revision=args.expected_revision,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return int(payload.get("status") == "rejected")

    if args.command in {"check", "run"}:
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
        source = args.config or args.run_config
        ids, selectors = args.batch_ids, args.batch_selectors
        if args.file_selector is not None or (args.resume_missing):
            from smartsom.config.experiment_layout import batch_directory, entry_id
            from smartsom.config.launch import resolve_experiment

            path = resolve_experiment(source)
            if args.file_selector is not None and not path.is_file():
                raise ConfigurationError("file --select requires one Experiment file")
            if path.is_file():
                if ids or selectors:
                    raise ConfigurationError(
                        "file selection cannot be combined with --id"
                    )
                source = batch_directory(path)
                ids = [entry_id(path, source)]
                selectors = (
                    [args.file_selector] if args.file_selector is not None else None
                )
        plan = compile_launch(
            source,
            ids=ids,
            selectors=selectors,
            tags=args.batch_tags,
            first_train_seeds=args.first_train_seeds,
            resume_missing=args.resume_missing,
            sets=changes,
            task=args.task,
            seed=args.seed,
            tuning=args.tune,
            learning_rate=args.learning_rate,
            profile=args.profile,
            operations=_operation_overrides(args),
        )
        if args.command == "check":
            if args.json:
                print(
                    json.dumps(
                        plan.summary(), ensure_ascii=False, indent=2, default=str
                    )
                )
            else:
                from smartsom.experiments.check_display import render_check

                print(render_check(plan, verbose=bool(args.verbose)))
            return 0
        else:
            print(json.dumps({"launch": plan.summary()}, ensure_ascii=False, indent=2))
            payload = run_launch(
                plan,
                keep_open=args.keep_open,
                interactive=sys.stdin.isatty() and sys.stdout.isatty(),
            )
    elif args.command == "resume":
        from smartsom.api import resume

        if args.task_id:
            from smartsom.experiments.catalog import require_current, resolve_run

            args.source = resolve_run(args.source, args.root)
            require_current(args.source)
            if _operation_overrides(args) or args.profile:
                raise ValueError("task resume cannot change settings; use configure")
            payload = _submit_control(args, "resume")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return int(payload.get("status") == "rejected")
        if args.command_id or args.expected_revision is not None:
            from smartsom.experiments.catalog import require_current, resolve_run

            args.source = resolve_run(args.source, args.root)
            require_current(args.source)
            if _operation_overrides(args) or args.profile:
                raise ValueError("journal resume cannot change settings; use configure")
            payload = _submit_control(args, "resume")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return int(payload.get("status") == "rejected")
        payload = resume(
            args.source,
            background=args.background,
            roots=args.root,
            profile=args.profile,
            operations=_operation_overrides(args),
            interactive=sys.stdin.isatty() and sys.stdout.isatty(),
        )
    elif args.command == "stop":
        from smartsom.experiments.control import stop

        if args.task_id:
            if args.force or args.timeout is not None:
                raise ValueError(
                    "task stop is cooperative; --force/--timeout apply only to the whole run"
                )
            payload = _submit_control(args, "stop")
        else:
            if args.command_id or args.expected_revision is not None:
                raise ValueError(
                    "whole-run stop verifies process exit; journal options require --task-id"
                )
            payload = stop(
                args.source,
                timeout=30 if args.timeout is None else args.timeout,
                force=args.force,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return int(
            bool(payload.get("remaining")) or payload.get("status") == "rejected"
        )
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
            raise ConfigurationError("historical recordings require archived source")
        if args.seed is not None:
            raise ConfigurationError(
                "current replay selects its recorded episode directory, not --seed"
            )
        from smartsom.studio.playback import playback_window

        return playback_window(root)
    else:
        raise ConfigurationError("historical command requires archived source")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return int(payload.get("status") in {"failed", "stopped", "partial_stopped"})


if __name__ == "__main__":
    raise SystemExit(main())
