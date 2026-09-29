"""CLI adaptation for the four-file compiler and immutable execution plans."""

import sys
from pathlib import Path

from smartsom.config.codec import ConfigurationError
from smartsom.config.experiment_v3 import read_document
from smartsom.config.experiment_v4 import compile_experiment


def is_author_input(args):
    if args.run_config and args.config:
        raise ConfigurationError("provide only one experiment configuration")
    path = args.config or args.run_config
    return bool(
        path and read_document(path).get("schema") == "smartsom.experiment-config/v4"
    )


def execute(args):
    if args.source or args.study or args.preset:
        raise ConfigurationError("v4 author input cannot combine source/study/preset")
    if args.initialize_from or args.baseline or args.scenario:
        raise ConfigurationError(
            "v4 models belong in Algorithm; cases belong in Workload"
        )
    if args.retry_failed:
        raise ConfigurationError("new author plans have no failed attempts; use resume")
    path = args.config or args.run_config
    task = args.task or read_document(path)["task"]
    if task == "evaluate" and any(
        getattr(args, key, None) is not None
        for key in ("steps", "steps_per_update", "learning_rate")
    ):
        raise ConfigurationError(
            "training budget/optimizer flags do not apply to evaluate"
        )
    if args.preview and task != "evaluate":
        raise ConfigurationError("--preview requires evaluate")
    evaluation = {}
    for key, target in (
        ("checkpoint", "checkpoint"),
        ("replications", "replications"),
        ("deterministic", "deterministic"),
        ("replay", "full_replay"),
        ("render_mode", "render_mode"),
        ("render_case", "render_case"),
        ("render_replication", "render_replication"),
        ("record", "record"),
    ):
        value = getattr(args, key)
        if value is not None:
            if task == "train":
                raise ConfigurationError("evaluation flags do not apply to train")
            evaluation[target] = value
    if args.preview:
        if any(
            getattr(args, key) is not None
            for key in ("replications", "render_case", "render_replication", "replay")
        ):
            raise ConfigurationError(
                "preview conflicts with case/replication/replay flags"
            )
        if any(
            item.split("=", 1)[0]
            in {
                "experiment.evaluation.replications",
                "experiment.evaluation.full_replay",
                "experiment.evaluation.render_case",
                "experiment.evaluation.render_replication",
                "experiment.evaluation",
            }
            for item in args.set
        ):
            raise ConfigurationError(
                "preview conflicts with evaluation --set overrides"
            )
        evaluation.update(
            replications=1,
            full_replay=False,
            record=True if args.record is None else args.record,
        )
    display = {}
    for key, target in (
        ("verbose", "verbose"),
        ("progress", "progress"),
        ("debug", "debug"),
        ("log_format", "format"),
        ("summary_interval", "every_seconds"),
        ("progress_title", "title"),
    ):
        value = getattr(args, key, None)
        if value is not None:
            display[target] = bool(value) if key == "verbose" else value
    changes = list(args.set)
    for key, target in (
        ("steps", "experiment.training.total_ticks"),
        ("steps_per_update", "experiment.training.ticks_per_update"),
        ("num_envs", "experiment.runtime.num_envs"),
        ("sampling_processes", "experiment.runtime.sampling_processes"),
        ("max_concurrent", "experiment.execution.max_concurrent"),
        ("threads", "experiment.runtime.numerical_threads"),
        ("device", "experiment.runtime.device"),
        ("learning_rate", "algorithm.learner.parameters.learning_rate"),
        ("name", "experiment.output.name"),
        ("mode", "experiment.execution.mode"),
        ("execution", "experiment.execution.scheduling"),
        ("calibration_timeout", "experiment.execution.calibration_seconds"),
        ("calibration_level", "experiment.execution.calibration_level"),
        ("calibration_candidate", "experiment.execution.calibration_candidate"),
        ("preflight", "experiment.execution.preflight"),
        ("preflight_coverage", "experiment.execution.preflight_coverage"),
    ):
        value = getattr(args, key, None)
        if value is not None:
            import json

            changes.append(target + "=" + json.dumps(value))
    if args.output_root is not None:
        import json

        changes.append(
            "experiment.output.root="
            + json.dumps(str(Path(args.output_root).expanduser().resolve()))
        )
    interactive_background = (
        args.command == "run"
        and args.background is None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and args.render_mode is None
    )
    plan = compile_experiment(
        path,
        task=task,
        factory=args.factory,
        workload=args.workload,
        algorithm=args.algorithm,
        sets=changes,
        seed=args.seed,
        data_seed=args.data_seed,
        background=True if interactive_background else args.background,
        tuning=args.tune,
        extension_modules=args.extension_module,
        require_dependencies=args.command != "check",
        evaluation_overrides=evaluation,
        display_overrides=display,
        purpose="preview" if args.preview else None,
    )
    if (
        args.command == "run"
        and any(
            getattr(args, key, None) is not None
            for key in (
                "calibration_level",
                "calibration_candidate",
                "calibration_timeout",
            )
        )
        and plan.experiment.execution.tuning == "off"
    ):
        raise ConfigurationError("calibration options require V4 Tune to be enabled")
    if (
        (args.mode or args.execution)
        and plan.experiment.execution.tuning == "off"
        and plan.experiment.execution.executor == "native"
    ):
        raise ConfigurationError(
            "Tune mode/scheduling flags do not apply to native execution"
        )
    if args.preview and len(plan.entries) != 1:
        raise ConfigurationError("preview requires a single selected combination")
    if args.command == "check":
        return plan.summary()
    from smartsom.experiments.author_driver import run

    return run(plan, display_options=display)
