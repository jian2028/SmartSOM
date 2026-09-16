"""Discoverable CLI over the typed Python experiment API."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
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
    "progress": ("logging.progress", str),
    "verbose": ("logging.verbose", int),
    "log-format": ("logging.format", str),
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


def _recipe_arguments(parser):
    parser.add_argument("run_config", nargs="?", help="experiment recipe")
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
        parser.add_argument(
            f"--{name}", type=kind, action=Once, help=f"override {field}"
        )
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")


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
    ]
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


def _parser():
    parser = argparse.ArgumentParser(
        prog="smartsom",
        description="Configure, train, evaluate and replay scheduling experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    studio = commands.add_parser(
        "studio", help="Browse and edit v2 factory designs in SmartSOM Studio"
    )
    studio.add_argument("paths", nargs="*", type=Path, help="Factory design YAML files")
    for name in ("doctor", "show-config", "validate", "run", "train", "train-evaluate"):
        command = commands.add_parser(name)
        _recipe_arguments(command)
        if name == "doctor":
            command.add_argument("--probe", action="store_true")
        if name in {"train", "train-evaluate"}:
            command.add_argument("--initialize-from")
        if name in {"run", "train-evaluate"}:
            command.add_argument(
                "--record", action=argparse.BooleanOptionalAction, default=True
            )
    presets = commands.add_parser("presets").add_subparsers(
        dest="action", required=True
    )
    presets.add_parser("list")
    presets.add_parser("show").add_argument("name", choices=PRESETS)
    init = commands.add_parser("init")
    init.add_argument("template")
    init.add_argument("destination", type=Path)
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
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("--checkpoint", choices=("last", "best"), default="last")
    evaluate.add_argument("--seed", type=int, default=202)
    evaluate.add_argument("--replications", type=int, default=5)
    evaluate.add_argument("--baseline", action="append", default=[])
    evaluate.add_argument("--scenario", action="append", default=[])
    evaluate.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    evaluate.add_argument(
        "--replay", action=argparse.BooleanOptionalAction, default=True
    )
    evaluate.add_argument("--output-root", type=Path)
    evaluate.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    evaluate.add_argument(
        "--record", action=argparse.BooleanOptionalAction, default=True
    )
    commands.add_parser("resume").add_argument("source", type=Path)
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
    commands.add_parser("plan").add_argument("study")
    batch = commands.add_parser("batch")
    batch.add_argument("study", nargs="?")
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


def main(argv=None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
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

        if args.command == "presets":
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
                result = api.run(config, record=args.record)
                print(
                    f"{result.status} makespan={result.simulation_result.makespan} run_dir={result.run_dir}"
                )
                return 0 if result.status == "completed" else 1
            else:
                if args.command == "train-evaluate":
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
                output = create_template(args.template, args.destination)
            payload = {
                "project": str(output),
                "next": f"smartsom run --config {output / 'run.yaml'}",
            }
        elif args.command == "migrate":
            config = load_config(args.input)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(config) + "\n")
            payload = {
                "migrated": str(args.output),
                "original_preserved": str(args.input),
            }
        elif args.command == "evaluate":
            from smartsom.config.experiment import EvaluationOptions

            options = EvaluationOptions(
                seed=args.seed,
                replications=args.replications,
                checkpoint=args.checkpoint,
                deterministic=args.deterministic,
                full_replay=args.replay,
                baselines=tuple(args.baseline),
                scenarios=tuple(args.scenario),
                verbose=args.verbose,
                record=args.record,
            )
            payload = primitive(
                api.evaluate(args.source, options, output_root=args.output_root)
            )
        elif args.command == "resume":
            payload = primitive(api.resume(args.source))
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
                    }
                    for status in states
                )
                or payload.get("pending", 0)
                or payload.get("failed", 0)
            ):
                return 1
        return 0
    except (ConfigurationError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
