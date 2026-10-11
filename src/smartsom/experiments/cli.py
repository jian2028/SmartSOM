"""Discoverable CLI over the typed Python experiment API."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import yaml

from smartsom.config.codec import (
    ConfigurationError,
    _UniqueLoader,
    canonical_json,
    primitive,
)
from smartsom.config.experiment import (
    PRESETS,
    apply_overrides,
    load_config,
    load_preset,
    preview,
)

FLAGS = {
    "seed": ("seed", int),
    "steps": ("training.total_steps", int),
    "num-envs": ("runtime.num_envs", int),
    "steps-per-update": ("training.steps_per_update", int),
    "sampling-processes": ("runtime.sampling_processes", int),
    "max-concurrent": ("runtime.max_concurrent", int),
    "threads": ("runtime.numerical_threads", int),
    "device": ("runtime.device", str),
    "learning-rate": ("algorithm.learning_rate", float),
    "name": ("output.name", str),
    "output-root": ("output.root", str),
    "progress-title": ("logging.title", str),
    "progress": ("logging.progress", str),
    "verbose": ("logging.verbose", int),
    "log-format": ("logging.format", str),
    "summary-interval": ("logging.every_seconds", float),
}


class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            raise ConfigurationError(f"duplicate option: {option_string}")
        setattr(namespace, self.dest, values)


class DisableOnce(Once):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, False, option_string)


def calibration_timeout(value):
    """Parse a positive wall-clock calibration budget such as 10m or 20m."""
    matched = re.fullmatch(r"([1-9][0-9]*)(s|m|h)", value)
    if matched is None:
        raise argparse.ArgumentTypeError("use a positive duration such as 10m or 20m")
    return int(matched[1]) * {"s": 1, "m": 60, "h": 3600}[matched[2]]


def _recipe_arguments(parser):
    parser.add_argument(
        "run_config",
        nargs="?",
        help="Experiment YAML (v4 has an explicit task) or legacy recipe",
    )
    parser.add_argument("--config", action=Once)
    parser.add_argument("--preset", choices=PRESETS, action=Once)
    for name, (field, kind) in FLAGS.items():
        if name == "verbose":
            parser.add_argument(
                "--verbose", nargs="?", const=True, type=int, action=Once
            )
            parser.add_argument(
                "--no-verbose", dest="verbose", action=DisableOnce, default=None
            )
            continue
        help_text = {
            "steps": "training budget: physical ticks for v3/v4, adapter decisions for legacy recipes",
            "steps-per-update": "training update interval: physical ticks for v3/v4, legacy adapter decisions",
            "max-concurrent": "native experiment concurrency; v4 execution.max_concurrent",
            "learning-rate": "learner learning rate; v4 algorithm.learner.parameters.learning_rate",
        }.get(name, f"override {field}")
        parser.add_argument(f"--{name}", type=kind, action=Once, help=help_text)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Typed override; v4 uses factory/workload/algorithm/experiment namespaces",
    )


def _recipe(args):
    if args.run_config and args.config:
        raise ConfigurationError("provide only one configuration file")
    path = args.config or args.run_config
    if path:
        config = load_config(path, preset=args.preset)
    elif args.preset:
        config = load_preset(args.preset)
    else:
        raise ConfigurationError(
            "provide --preset or --config; see smartsom presets list"
        )
    overrides = [
        (field, getattr(args, name.replace("-", "_")))
        for name, (field, _) in FLAGS.items()
        if getattr(args, name.replace("-", "_"), None) is not None
        and name != "progress-title"
    ]
    if getattr(args, "debug", None) is not None:
        overrides.append(("logging.debug", args.debug))
    for item in args.set:
        if "=" not in item:
            raise ConfigurationError("--set requires FIELD=VALUE")
        key, value = item.split("=", 1)
        overrides.append((key, yaml.load(value, Loader=_UniqueLoader)))
    return apply_overrides(config, overrides) if overrides else config


def _doctor(args):
    import smartsom
    from smartsom.experiments.evidence import source_identity
    from smartsom.learning.checkpoint import VERSIONS, require_backend

    packages = {}
    for name in (
        *VERSIONS,
        "rich",
        "tensorboard",
        "wandb",
        "optuna",
        "matplotlib",
        "pyjobshop",
        "ortools",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source": source_identity(),
        "imported_package": str(Path(smartsom.__file__).resolve()),
        "packages": packages,
        "backend_probe": "not requested",
        "status": "passed",
    }
    config = _recipe(args) if args.preset or args.config or args.run_config else None
    if config is None and (
        args.probe
        or args.set
        or any(
            getattr(args, name.replace("-", "_"), None) is not None for name in FLAGS
        )
    ):
        raise ConfigurationError(
            "doctor overrides and --probe require --preset or --config"
        )
    root = Path(config.output.root if config else "runs").resolve()
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    result["output"] = {
        "root": str(root),
        "exists": root.exists(),
        "writable_parent": str(ancestor),
        "writable": ancestor.is_dir() and os.access(ancestor, os.W_OK),
    }
    if sys.version_info[:2] != (3, 12):
        raise ConfigurationError("SmartSOM requires Python 3.12")
    if config:
        details = preview(config)
        result["experiment"] = details
        if details["provider"].startswith(("rllib.", "sb3.")):
            require_backend(details["provider"])
        elif details["provider"] == "pyjobshop.cp_sat":
            if any(packages[p] is None for p in ("pyjobshop", "ortools")):
                raise ConfigurationError("CP-SAT requires uv sync --locked --extra cp")
        result["device"] = {"requested": config.runtime.device, "available": True}
        if config.runtime.device == "cuda":
            import torch

            result["device"]["available"] = torch.cuda.is_available()
            if not result["device"]["available"]:
                raise ConfigurationError("CUDA was requested but is unavailable")
        from smartsom.telemetry.training import TrainingDisplay

        TrainingDisplay.preflight(config.logging)
    if args.probe:
        from smartsom.config.experiment import prepare
        from smartsom.experiments.training_probe import probe_training_backend

        prepared = prepare(config)
        result["backend_probe"] = probe_training_backend(
            prepared,
            config.runtime,
        )
    return result


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


def _parser():
    parser = argparse.ArgumentParser(
        prog="smartsom",
        description="Configure, train, evaluate and replay scheduling experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    tuning = commands.add_parser(
        "tune", help="Check, calibrate and run frozen v3 experiment batches"
    )
    tuning_actions = tuning.add_subparsers(dest="tune_action", required=True)
    for action in ("check", "recommend", "run"):
        command = tuning_actions.add_parser(action)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--batch", type=Path)
        source.add_argument("--study", type=Path)
        command.add_argument("--mode", choices=("balanced", "performance"))
        if action != "check":
            command.add_argument("--calibration-timeout", type=calibration_timeout)
            command.add_argument(
                "--background", action=argparse.BooleanOptionalAction, default=None
            )
            command.add_argument("--preflight", choices=("quick", "full"))
            command.add_argument(
                "--preflight-coverage", choices=("each", "representative")
            )
        command.add_argument("--execution", choices=("adaptive", "fixed"))
        _display_arguments(command)
    tune_resume = tuning_actions.add_parser("resume")
    tune_resume.add_argument("directory", type=Path)
    tune_resume.add_argument("--retry-failed", action="store_true")
    _display_arguments(tune_resume)
    studio = commands.add_parser(
        "studio", help="Browse and edit v2 factory designs in SmartSOM Studio"
    )
    studio.add_argument("paths", nargs="*", type=Path, help="Factory design YAML files")
    batch_run = commands.add_parser(
        "batch-run", help="Run a frozen directory of V4 Experiments"
    )
    batch_run.add_argument("directory", type=Path)
    batch_run.add_argument(
        "--background", action=argparse.BooleanOptionalAction, default=None
    )
    batch_run.add_argument(
        "--calibration-timeout",
        type=calibration_timeout,
        help="Override the shared calibration wall-clock budget",
    )
    batch_run.add_argument(
        "--calibration-level", choices=("online", "off", "quick", "full")
    )
    batch_run.add_argument("--calibration-candidate", default=None)
    _display_arguments(batch_run)
    for name in (
        "doctor",
        "show-config",
        "validate",
        "check",
        "run",
        "train",
        "train-evaluate",
    ):
        command = commands.add_parser(name)
        _recipe_arguments(command)
        if name in {"check", "run"}:
            command.add_argument(
                "--task",
                choices=("train", "evaluate", "train-evaluate"),
                help="Override the explicit v4 task; required for the new v3 task interface",
            )
            command.add_argument(
                "--factory",
                action=Once,
                help="v4 Factory selector; narrows a matrix axis",
            )
            command.add_argument(
                "--workload",
                action=Once,
                help="v4 Workload selector; narrows a matrix axis",
            )
            command.add_argument(
                "--algorithm", action=Once, help="v4 Algorithm selector"
            )
            command.add_argument(
                "--data-seed",
                type=int,
                action=Once,
                help="v4 data root, independent of policy/learning --seed",
            )
            command.add_argument(
                "--background",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Explicit v4 detached macOS/Linux execution (default foreground)",
            )
            command.add_argument(
                "--tune",
                choices=("off", "recommend", "auto"),
                action=Once,
                help="v4 training calibration: off, measure only, or adopt and execute",
            )
            command.add_argument(
                "--extension-module",
                action="append",
                default=[],
                help="Explicit versioned rule registration module; repeatable",
            )
            source = command.add_mutually_exclusive_group()
            source.add_argument("--source", type=Path)
            source.add_argument("--study", type=Path)
            command.add_argument("--initialize-from")
            command.add_argument("--checkpoint")
            command.add_argument("--preview", action="store_true")
            command.add_argument("--replications", type=int)
            command.add_argument("--baseline", action="append", default=[])
            command.add_argument("--scenario", action="append", default=[])
            command.add_argument(
                "--deterministic", action=argparse.BooleanOptionalAction, default=None
            )
            command.add_argument(
                "--replay", action=argparse.BooleanOptionalAction, default=None
            )
            command.add_argument("--render-case")
            command.add_argument("--render-replication", type=int)
            command.add_argument("--mode", choices=("balanced", "performance"))
            command.add_argument("--calibration-timeout", type=calibration_timeout)
            command.add_argument(
                "--calibration-level", choices=("online", "off", "quick", "full")
            )
            command.add_argument("--calibration-candidate")
            command.add_argument("--execution", choices=("adaptive", "fixed"))
            command.add_argument("--retry-failed", action="store_true")
            command.add_argument("--preflight", choices=("quick", "full"))
            command.add_argument(
                "--preflight-coverage", choices=("each", "representative")
            )
        if name == "doctor":
            command.add_argument("--probe", action="store_true")
        if name in {"train", "train-evaluate"}:
            command.add_argument("--initialize-from")
        if name in {"check", "run", "train-evaluate"}:
            command.add_argument("--render-mode", choices=("human",))
            command.add_argument(
                "--record",
                action=argparse.BooleanOptionalAction,
                default=True if name == "train-evaluate" else None,
            )
    presets = commands.add_parser("presets").add_subparsers(
        dest="action", required=True
    )
    presets.add_parser("list")
    presets.add_parser("show").add_argument("name", choices=PRESETS)
    init = commands.add_parser("init")
    init.add_argument("template")
    init.add_argument("destination", type=Path)
    init.add_argument("--name", default="example")
    importer = commands.add_parser("import-fjs")
    importer.add_argument("input", type=Path)
    importer.add_argument("--output-dir", type=Path, required=True)
    importer.add_argument("--instance-id")
    importer.add_argument("--factory", type=Path, required=True)
    importer.add_argument(
        "--machine-map",
        type=Path,
        help="JSON mapping from FJS M1, M2, ... to factory machine IDs",
    )
    migrate = commands.add_parser("migrate")
    migrate.add_argument("input", type=Path)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument("--to", choices=("v2", "v3"), default="v2")
    migrate.add_argument("--preview", action="store_true")
    migrate.add_argument("--total-ticks", type=int)
    migrate.add_argument("--ticks-per-update", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("source", nargs="?", type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--replications", type=int)
    evaluate.add_argument("--baseline", action="append", default=[])
    evaluate.add_argument("--scenario", action="append", default=[])
    evaluate.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=None
    )
    evaluate.add_argument(
        "--replay", action=argparse.BooleanOptionalAction, default=None
    )
    evaluate.add_argument("--output-root", type=Path)
    evaluate.add_argument("--render-mode", choices=("human",))
    evaluate.add_argument("--render-case")
    evaluate.add_argument("--render-replication", type=int)
    _display_arguments(evaluate)
    evaluate.add_argument(
        "--record", action=argparse.BooleanOptionalAction, default=None
    )
    commands.add_parser("playback").add_argument("source", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("source", type=Path)
    resume.add_argument("--retry-failed", action="store_true")
    resume.add_argument(
        "--background", action=argparse.BooleanOptionalAction, default=None
    )
    resume.add_argument("--max-concurrent", type=int)
    resume.add_argument("--extension-module", action="append", default=[])
    _display_arguments(resume)
    stop = commands.add_parser("stop", help="Safely stop one registered local run")
    stop.add_argument("source", type=Path)
    stop.add_argument("--timeout", type=float, default=60)
    stop.add_argument("--force", action="store_true")
    monitor = commands.add_parser("monitor", help="Read-only mainline runtime monitor")
    monitor.add_argument("source", type=Path)
    monitor.add_argument("--once", action="store_true")
    _display_arguments(monitor)
    attach = commands.add_parser("attach", help="Attach a controlling Rich view")
    attach.add_argument("source", type=Path)
    _display_arguments(attach)
    preflight = commands.add_parser(
        "preflight", help="Control optional preflight smoke"
    )
    preflight_actions = preflight.add_subparsers(dest="preflight_action", required=True)
    preflight_set = preflight_actions.add_parser("set")
    preflight_set.add_argument("source", type=Path)
    preflight_set.add_argument(
        "--coverage", choices=("representative", "skip"), required=True
    )
    audit = commands.add_parser("audit")
    audit.add_argument("source", type=Path)
    audit.add_argument("--training", action="store_true")
    audit.add_argument("--output", type=Path)
    report = commands.add_parser("report")
    report.add_argument("source", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument(
        "--format", choices=("html", "png", "svg", "pdf"), default="html"
    )
    export = commands.add_parser("export")
    export.add_argument("source", type=Path)
    export.add_argument("--kind", choices=("model", "experiment"), default="model")
    export.add_argument("--output", type=Path)
    export.add_argument("--group")
    export.add_argument("--checkpoint", default="last")
    runs = commands.add_parser("runs")
    runs.add_argument("action", choices=("list", "show"))
    runs.add_argument("query", nargs="?")
    runs.add_argument("--root", action="append", default=[])
    runs.add_argument("--name")
    runs.add_argument("--status")
    runs.add_argument("--algorithm")
    runs.add_argument("--tag", action="append", default=[])
    index = commands.add_parser("index")
    index.add_argument("action", choices=("rebuild",))
    index.add_argument("--root", action="append", default=[])
    index.add_argument("--views-dir", type=Path, default=Path("."))
    study = commands.add_parser("study")
    actions = study.add_subparsers(dest="study_action", required=True)
    prepare_study = actions.add_parser("prepare")
    prepare_study.add_argument("--config", required=True, type=Path)
    prepare_study.add_argument("--output", required=True, type=Path)
    for action in ("show", "run", "resume"):
        child = actions.add_parser(action)
        child.add_argument("directory", type=Path)
        if action in ("run", "resume"):
            child.add_argument("--retry-failed", action="store_true")
            _display_arguments(child)
    commands.add_parser("plan").add_argument("study")
    batch = commands.add_parser("batch")
    batch.add_argument("study", nargs="?")
    _display_arguments(batch)
    batch.add_argument("--resume")
    batch.add_argument("--retry-failed", action="store_true")
    batch.add_argument("--workers", type=int, default=1)
    for name in ("batch-train", "search"):
        learning = commands.add_parser(name)
        _recipe_arguments(learning)
        learning.add_argument("--resume", type=Path)
        learning.add_argument("--retry-failed", action="store_true")
        if name == "batch-train":
            learning.add_argument("--recipe", action="append", default=[])
            learning.add_argument("--seeds", type=int, nargs="+")
    return parser


def _evaluation_options(args, defaults):
    """Only explicit CLI flags override a frozen run's evaluation recipe."""
    values = {
        "seed": args.seed,
        "replications": args.replications,
        "checkpoint": args.checkpoint,
        "deterministic": args.deterministic,
        "full_replay": args.replay,
        "render_mode": args.render_mode,
        "render_case": args.render_case,
        "render_replication": args.render_replication,
        "verbose": None if args.verbose is None else bool(args.verbose),
        "record": args.record,
    }
    if args.baseline:
        values["baselines"] = tuple(args.baseline)
    if args.scenario:
        values["scenarios"] = tuple(args.scenario)
    return type(defaults).model_validate(
        {**defaults.model_dump(), **{k: v for k, v in values.items() if v is not None}}
    )


def main(argv=None) -> int:
    parser = _parser()
    args = None
    try:
        args = parser.parse_args(argv)
        return _dispatch(args, parser)
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


def _execute_args(args, parser):
    try:
        if args.command == "batch-run":
            from smartsom.experiments.author_batch import compile_directory, run

            attach_after_launch = (
                args.background is None and sys.stdin.isatty() and sys.stdout.isatty()
            )
            interactive = sys.stderr.isatty() and attach_after_launch
            if interactive:
                from rich.console import Console

                preparation = Console(stderr=True).status(
                    "正在检查并冻结批次输入…", spinner="dots"
                )
            else:
                preparation = nullcontext()
            with preparation as indicator:
                prepared = compile_directory(
                    args.directory,
                    require_dependencies=True,
                    calibration_seconds=args.calibration_timeout,
                    calibration_level=args.calibration_level,
                    calibration_candidate=args.calibration_candidate,
                )
                if indicator is not None:
                    indicator.update("输入已检查；正在创建批次并启动监控…")
                payload = run(
                    prepared,
                    background=(args.background is True or attach_after_launch),
                )
            if attach_after_launch and payload.get("background"):
                from smartsom.telemetry.monitor import monitor

                monitor(payload["directory"], controlling=True)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return (
                1
                if payload.get("status") in {"failed", "stopped", "force_stopped"}
                else 0
            )
        if (
            args.command == "check"
            and (args.config or args.run_config)
            and Path(args.config or args.run_config).is_dir()
        ):
            from smartsom.experiments.author_batch import compile_directory

            semantic = (
                "preset",
                "set",
                "seed",
                "steps",
                "num_envs",
                "steps_per_update",
                "sampling_processes",
                "max_concurrent",
                "threads",
                "device",
                "learning_rate",
                "name",
                "output_root",
                "task",
                "factory",
                "workload",
                "algorithm",
                "data_seed",
                "background",
                "tune",
                "extension_module",
                "source",
                "study",
                "initialize_from",
                "checkpoint",
                "preview",
                "replications",
                "baseline",
                "scenario",
                "deterministic",
                "replay",
                "render_case",
                "render_replication",
                "mode",
                "calibration_timeout",
                "calibration_level",
                "calibration_candidate",
                "execution",
                "retry_failed",
                "preflight",
                "preflight_coverage",
                "record",
            )
            nullable_booleans = {"background", "deterministic", "replay", "record"}
            selected = [
                key
                for key in semantic
                if (
                    getattr(args, key, None) is not None
                    if key in nullable_booleans
                    else bool(getattr(args, key, None))
                )
            ]
            if selected:
                raise ConfigurationError(
                    "directory check does not accept per-Experiment overrides: "
                    + ", ".join(selected)
                )
            payload = compile_directory(args.config or args.run_config).summary()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"show-config", "validate"}:
            from smartsom.experiments.author_commands import execute as execute_author
            from smartsom.experiments.author_commands import is_author_input

            if is_author_input(args):
                translated = _parser().parse_args(
                    ["check", str(args.config or args.run_config)]
                )
                for key, value in vars(args).items():
                    if hasattr(translated, key) and key != "command":
                        setattr(translated, key, value)
                payload = execute_author(translated)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
        if args.command in {"check", "run"}:
            from smartsom.experiments.author_commands import execute as execute_author
            from smartsom.experiments.author_commands import is_author_input

            if is_author_input(args):
                attach_after_launch = (
                    args.command == "run"
                    and args.background is None
                    and sys.stdin.isatty()
                    and sys.stdout.isatty()
                    and args.render_mode is None
                )
                payload = execute_author(args)
                if attach_after_launch and payload.get("background"):
                    from smartsom.telemetry.monitor import monitor

                    monitor(payload["directory"], controlling=True)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return (
                    1
                    if payload.get("status") in {"failed", "stopped", "force_stopped"}
                    else 0
                )
            if any(
                getattr(args, key, None) is not None
                for key in (
                    "factory",
                    "workload",
                    "algorithm",
                    "data_seed",
                    "background",
                    "tune",
                    "calibration_timeout",
                    "calibration_level",
                    "calibration_candidate",
                )
            ) or getattr(args, "extension_module", None):
                raise ValueError(
                    "four-file and background settings require experiment-config/v4"
                )
        if args.command == "stop":
            from smartsom.experiments.control import stop

            payload = stop(args.source, timeout=args.timeout, force=args.force)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1 if payload["remaining"] else 0
        if args.command == "preflight":
            from smartsom.experiments.control import set_preflight_coverage

            payload = set_preflight_coverage(args.source, args.coverage)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "check" or (args.command == "run" and args.task):
            from smartsom.experiments.commands import execute

            payload = execute(args)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.command == "check":
                return 0
            outcomes = [
                payload,
                payload.get("training") or {},
                payload.get("evaluation") or {},
            ]
            return (
                1
                if any(
                    result.get("status") in {"failed", "interrupted", "stopped"}
                    for result in outcomes
                )
                else 0
            )
        if (
            args.command in {"train", "train-evaluate", "evaluate", "study", "tune"}
            or args.command == "run"
        ):
            message = "Compatibility entry: prefer check/run --task with --config, --source or --study."
            print(
                json.dumps({"type": "migration", "message": message})
                if getattr(args, "log_format", None) == "json"
                else message,
                file=sys.stderr,
            )
        if (
            args.command == "run"
            and not args.task
            and (
                args.preview
                or args.source
                or args.study
                or args.mode
                or args.execution
                or args.checkpoint
                or args.initialize_from
                or args.replications is not None
                or args.retry_failed
                or args.baseline
                or args.scenario
                or args.replay is not None
                or args.render_case
                or args.render_replication is not None
                or args.deterministic is not None
            )
        ):
            raise ValueError(
                "new task options require --task; legacy run semantics are unchanged"
            )
        if args.command == "monitor":
            from smartsom.telemetry.monitor import monitor
            from smartsom.telemetry.runtime import OVERRIDES, DisplayOptions

            return monitor(
                args.source,
                once=args.once,
                options=DisplayOptions.from_value(OVERRIDES.get()),
            )
        if args.command == "attach":
            from smartsom.telemetry.monitor import monitor
            from smartsom.telemetry.runtime import OVERRIDES, DisplayOptions

            return monitor(
                args.source,
                options=DisplayOptions.from_value(OVERRIDES.get()),
                controlling=True,
            )
        if args.command == "studio":
            try:
                from smartsom.studio.app import main as studio_main

                return studio_main(args.paths)
            except ModuleNotFoundError as exc:
                if exc.name == "PySide6" or (exc.name or "").startswith("PySide6."):
                    parser.error(
                        "Studio requires the studio extra: uv sync --extra studio"
                    )
                raise
        from smartsom import api

        if args.command == "tune":
            if args.tune_action == "resume":
                payload = api.resume_tune_batch(
                    args.directory, retry_failed=args.retry_failed
                )
            elif args.tune_action in {"run", "recommend"} and (
                args.background is True
                or args.background is None
                and sys.stdin.isatty()
                and sys.stdout.isatty()
            ):
                from dataclasses import replace

                from smartsom.experiments.background import launch
                from smartsom.experiments.tuning_batch import (
                    allocate_batch,
                    load_batch,
                    preflight,
                )

                inputs = load_batch(args.batch, args.study)
                inputs = replace(
                    inputs,
                    **{
                        key: value
                        for key, value in (
                            ("mode", args.mode),
                            ("execution", args.execution),
                            ("active_limit", args.calibration_timeout),
                            ("preflight", args.preflight),
                            ("preflight_coverage", args.preflight_coverage),
                        )
                        if value is not None
                    },
                )
                preflight(inputs)
                directory, _, _ = allocate_batch(inputs)
                payload = launch(
                    directory, recommend_only=args.tune_action == "recommend"
                )
                if args.background is None:
                    from smartsom.telemetry.monitor import monitor

                    monitor(directory, controlling=True)
            else:
                payload = api.tune_batch(
                    batch=args.batch,
                    study=args.study,
                    action=args.tune_action,
                    mode=args.mode,
                    execution=args.execution,
                    calibration_timeout=getattr(args, "calibration_timeout", None),
                    preflight=getattr(args, "preflight", None),
                    preflight_coverage=getattr(args, "preflight_coverage", None),
                )
        elif args.command == "study":
            from smartsom.experiments.composable_study import (
                prepare_study,
                run_study,
                show_study,
            )

            if args.study_action == "prepare":
                payload = prepare_study(args.config, args.output)
            elif args.study_action == "show":
                payload = show_study(args.directory)
            else:
                payload = run_study(args.directory, retry_failed=args.retry_failed)
        elif args.command == "presets":
            payload = (
                {name: description for name, (_, description) in PRESETS.items()}
                if args.action == "list"
                else preview(load_preset(args.name))
            )
        elif args.command == "doctor":
            payload = _doctor(args)
        elif args.command in {
            "show-config",
            "validate",
            "run",
            "train",
            "train-evaluate",
        }:
            config = _recipe(args)
            if args.command in {"show-config", "validate"}:
                payload = preview(config)
                if args.command == "validate":
                    prefix = (
                        "valid training"
                        if payload["provider"].startswith(("rllib.", "sb3."))
                        else "valid"
                    )
                    print(
                        f"{prefix} workload_sha256={payload['inputs']['workload_sha256']}"
                    )
                    return 0
            elif args.command == "run":
                result = api.run(
                    config,
                    render_mode=args.render_mode,
                    record=True if args.record is None else args.record,
                )
                print(
                    f"{result.status} makespan={result.simulation_result.makespan} run_dir={result.run_dir}"
                )
                return 0 if result.status == "completed" else 1
            else:
                if args.command == "train-evaluate":
                    config.evaluation.render_mode = args.render_mode
                    config.evaluation.record = args.record
                result = (api.train if args.command == "train" else api.train_evaluate)(
                    config, initialize_from=args.initialize_from
                )
                payload = primitive(result)
        elif args.command in {"init", "import-fjs"}:
            from smartsom.config.authoring import create_template, import_fjs_project

            if args.command == "import-fjs":
                output = import_fjs_project(
                    args.input,
                    args.output_dir,
                    instance_id=args.instance_id,
                    factory=args.factory,
                    machine_map=json.loads(args.machine_map.read_text())
                    if args.machine_map
                    else None,
                )
            else:
                if args.template == "composable":
                    from smartsom.config.composable_authoring import scaffold

                    payload = scaffold(args.destination, args.name)
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                    return 0
                output = create_template(args.template, args.destination)
            payload = {
                "project": str(output),
                "next": f"smartsom run --config {output / 'run.yaml'}",
            }
        elif args.command == "migrate":
            if args.to == "v3":
                from smartsom.config.composable_authoring import migrate

                payload = migrate(
                    args.input,
                    args.output,
                    preview=args.preview,
                    total_ticks=args.total_ticks,
                    ticks_per_update=args.ticks_per_update,
                )
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
            config = load_config(args.input)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(config) + "\n")
            payload = {
                "migrated": str(args.output),
                "original_preserved": str(args.input),
            }
        elif args.command == "evaluate":
            from smartsom.config.experiment import EvaluationOptions

            if args.config is not None:
                if args.source is not None:
                    raise ValueError("choose a run source or --config")
                config = load_config(args.config)
                config.evaluation = _evaluation_options(args, config.evaluation)
                payload = primitive(
                    api.evaluate(config=config, output_root=args.output_root)
                )
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
            if args.source is None:
                raise ValueError("evaluate requires RUN_DIRECTORY or --config")
            manifest = args.source / "run.json"
            v3 = manifest.is_file() and json.loads(manifest.read_text()).get(
                "schema"
            ) in {"smartsom.experiment/v3", "smartsom.experiment/v4"}
            if v3:
                from smartsom.experiments.composable import prepared_from_run

                defaults = prepared_from_run(args.source).config.evaluation
            else:
                defaults = EvaluationOptions()
            options = _evaluation_options(args, defaults)
            payload = primitive(
                api.evaluate(args.source, options, output_root=args.output_root)
            )
        elif args.command == "resume":
            from smartsom.experiments.commands import resume

            payload = resume(args)
        elif args.command == "playback":
            from smartsom.studio.playback import playback_window

            playback_window(args.source)
            return 0
        elif args.command == "audit":
            if args.training:
                from smartsom.experiments.catalog import training_locator
                from smartsom.experiments.training_audit import audit_training

                payload = audit_training(training_locator(args.source))
            else:
                from smartsom.trace.production import audit as audit_run

                payload = audit_run(args.source)
            if args.output:
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(canonical_json(payload) + "\n")
        elif args.command == "report":
            from smartsom.experiments.catalog import CURRENT_SCHEMAS
            from smartsom.experiments.report import build_report, export_figure

            current = args.source / "run.json"
            is_current = (
                current.is_file()
                and json.loads(current.read_text()).get("schema") in CURRENT_SCHEMAS
            )
            output = args.output or (
                (args.source / "reports" / f"report.{args.format}")
                if is_current
                else Path("reports") / f"{args.source.name}.{args.format}"
            )
            payload = {
                "report": str(
                    (build_report if args.format == "html" else export_figure)(
                        args.source, output
                    )
                )
            }
        elif args.command == "export":
            from smartsom.experiments.packaging import export_experiment, export_model

            output = (
                args.output or Path("exports") / f"{args.source.name}-{args.kind}.zip"
            )
            manifest = args.source / "run.json"
            if manifest.is_file() and json.loads(manifest.read_text()).get(
                "schema"
            ) in {"smartsom.experiment/v3", "smartsom.experiment/v4"}:
                from smartsom.experiments.composable import export

                payload = {
                    "package": str(
                        export(
                            args.source,
                            output,
                            group=args.group,
                            selection=args.checkpoint,
                            kind=args.kind,
                        )
                    )
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
            payload = {
                "package": str(
                    (export_model if args.kind == "model" else export_experiment)(
                        args.source, output
                    )
                )
            }
        elif args.command in {"runs", "index"}:
            from smartsom.experiments.catalog import (
                list_runs,
                read_run,
                rebuild_views,
                resolve_run,
            )

            roots = [Path(r) for r in args.root] or [Path("runs")]
            if args.command == "index":
                payload = {"index": str(rebuild_views(roots, args.views_dir))}
            elif args.action == "list":
                payload = [
                    primitive(entry)
                    for entry in list_runs(roots)
                    if (not args.name or args.name in entry.name)
                    and (not args.status or entry.status == args.status)
                    and (not args.algorithm or entry.provider == args.algorithm)
                    and set(args.tag).issubset(entry.tags)
                ]
            else:
                if not args.query:
                    raise ConfigurationError("runs show requires an ID or directory")
                payload = primitive(read_run(resolve_run(args.query, roots)))
        elif args.command in {"batch-train", "search"}:
            if args.resume:
                if (
                    args.run_config
                    or args.config
                    or args.preset
                    or args.set
                    or any(
                        getattr(args, name.replace("-", "_")) is not None
                        for name in FLAGS
                        if name
                        not in {
                            "verbose",
                            "progress",
                            "progress-title",
                            "log-format",
                            "summary-interval",
                        }
                    )
                    or getattr(args, "recipe", None)
                    or getattr(args, "seeds", None)
                ):
                    raise ConfigurationError(
                        "resume uses the frozen study; do not also override its recipe"
                    )
                result = (api.search if args.command == "search" else api.batch_train)(
                    resume=args.resume, retry_failed=args.retry_failed
                )
            elif args.command == "search":
                if args.retry_failed:
                    raise ConfigurationError("retry-failed requires an existing search")
                result = api.search(_recipe(args))
            else:
                if args.retry_failed:
                    raise ConfigurationError("retry-failed requires an existing batch")
                if args.recipe and (args.run_config or args.config or args.preset):
                    raise ConfigurationError(
                        "choose a preset/config or a list of --recipe files"
                    )
                configs = (
                    [
                        _recipe(
                            argparse.Namespace(**(vars(args) | {"run_config": path}))
                        )
                        for path in args.recipe
                    ]
                    if args.recipe
                    else [_recipe(args)]
                )
                if args.seeds and args.seed is not None:
                    raise ConfigurationError("seed and seeds are overlapping overrides")
                if args.seeds:
                    if len(args.seeds) != len(set(args.seeds)):
                        raise ConfigurationError("duplicate independent training seeds")
                    configs = [
                        apply_overrides(config, [("seed", seed)])
                        for config in configs
                        for seed in args.seeds
                    ]
                result = api.batch_train(
                    configs, max_concurrent=configs[0].runtime.max_concurrent
                )
            payload = primitive(result)
        elif args.command in {"plan", "batch"}:
            from smartsom.config import resolve_study
            from smartsom.experiments.batch import run_batch

            if args.command == "plan":
                study = resolve_study(args.study)
                payload = {
                    "runs": len(study.entries),
                    "plan_sha256": study.plan_sha256,
                    "entries": [primitive(e) for e in study.entries],
                }
            else:
                if bool(args.study) == bool(args.resume):
                    raise ConfigurationError(
                        "provide exactly one STUDY or --resume STUDY_DIR"
                    )
                result = run_batch(
                    resolve_study(args.study) if args.study else None,
                    resume=args.resume,
                    workers=args.workers,
                    retry_failed=args.retry_failed,
                )
                print(canonical_json(result))
                return (
                    130
                    if result.interrupted
                    else int(bool(result.failed or result.pending))
                )
        print(json.dumps(primitive(payload), indent=2, ensure_ascii=False))
        if isinstance(payload, dict):
            states = [payload.get("status")]
            for stage in ("training", "evaluation"):
                if isinstance(payload.get(stage), dict):
                    states.append(payload[stage].get("status"))
            if "interrupted" in states:
                return 130
            if (
                any(
                    status
                    in {
                        "failed",
                        "finished_with_failures",
                        "completed_with_failures",
                        "not_completed",
                        "stopped",
                        "force_stopped",
                    }
                    for status in states
                )
                or payload.get("pending", 0)
                or payload.get("failed", 0)
            ):
                return 1
        return 0
    except (ConfigurationError, ValueError, TypeError, yaml.YAMLError) as exc:
        _error(args, f"configuration error: {exc}")
        return 2
    except (OSError, RuntimeError, ImportError) as exc:
        _error(args, str(exc))
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
