"""Explicit task resolution shared by CLI check and execution."""

import json
import os
from dataclasses import replace
from pathlib import Path

from smartsom.config.codec import canonical_json, primitive
from smartsom.config.experiment import load_config
from smartsom.config.experiment_v3 import (
    ComposableExperimentConfig,
    prepare_v3,
    read_document,
)


def _checkpoint_conditions(config):
    selection = config.evaluation.checkpoint
    if selection == "last" and not config.checkpointing.save_last:
        raise ValueError("train-evaluate last requires checkpointing.save_last")
    if selection == "best" and (
        not config.validation.enabled
        or not config.checkpointing.save_best
        or config.training.total_ticks
        < config.training.ticks_per_update * config.validation.every_updates
    ):
        raise ValueError(
            "train-evaluate best requires scheduled validation and best saving"
        )
    if selection not in {"last", "best"}:
        raise ValueError("train-evaluate checkpoint must be last or best")


def _options(args, defaults):
    from smartsom.experiments.cli import _evaluation_options

    if args.preview:
        if args.task != "evaluate":
            raise ValueError("--preview requires single evaluate")
        if (
            any(
                value is not None
                for value in (
                    args.replications,
                    args.render_case,
                    args.render_replication,
                    args.replay,
                )
            )
            or args.scenario
        ):
            raise ValueError(
                "--preview conflicts with case, replication or replay overrides"
            )
    options = _evaluation_options(args, defaults)
    if args.preview:
        options = options.model_copy(
            update={
                "replications": 1,
                "full_replay": False,
                "record": True if args.record is None else args.record,
            }
        )
    return options


def resolve(args):
    """Read and validate, never allocate a run or instantiate a learner."""
    if args.task is None:
        raise ValueError(
            "legacy check requires --task; v4 reads explicit task from file"
        )
    if any(
        getattr(args, name, None)
        for name in (
            "factory",
            "workload",
            "algorithm",
            "data_seed",
            "background",
            "performance",
            "extension_module",
        )
    ):
        raise ValueError(
            "four-file selectors/performance/background require experiment-config/v4"
        )
    if sum(bool(v) for v in (args.config, args.source, args.study)) != 1:
        raise ValueError("choose exactly one of --config, --source or --study")
    if args.run_config or args.preset or args.set:
        raise ValueError(
            "explicit tasks require --config, --source or --study; legacy overrides are unsupported"
        )
    if args.study:
        if args.task != "train-evaluate":
            raise ValueError("prepared Study supports train-evaluate only")
        _batch_flags(args)
        from smartsom.experiments.composable_study import show_study

        return {
            "task": args.task,
            "input_type": "study",
            "executor": "native-study",
            **show_study(args.study),
        }, None
    if args.config:
        document = read_document(Path(args.config))
        if document.get("schema") == "smartsom.tune-batch/v1":
            if args.task != "train-evaluate":
                raise ValueError("Tune batch supports train-evaluate only")
            _batch_flags(args, tuning=True)
            from smartsom import api

            return {
                "task": args.task,
                "input_type": "tune-batch",
                "executor": "ray-tune",
                **api.tune_batch(
                    batch=args.config,
                    action="check",
                    mode=args.mode,
                    execution=args.execution,
                ),
            }, None
        config = load_config(args.config)
        if not isinstance(config, ComposableExperimentConfig):
            raise ValueError(
                "explicit task entry requires experiment-config/v3; migrate or use a legacy entry"
            )
    else:
        if args.task != "evaluate":
            raise ValueError(
                "--source supports evaluate only; use resume to continue training"
            )
        from smartsom.experiments.composable import prepared_from_run

        author_plan = Path(args.source) / "plan.json"
        if (
            author_plan.is_file()
            and json.loads(author_plan.read_text()).get("schema")
            == "smartsom.author-plan/v1"
        ):
            from smartsom.experiments.author_driver import load

            root, plan, state = load(args.source)
            if len(plan["entries"]) != 1:
                raise ValueError(
                    "batch source evaluation requires selecting an individual training run"
                )
            ledger = root / "entries" / plan["entries"][0]["id"] / "stages.json"
            stage = (
                json.loads(ledger.read_text()).get("stages", {}).get("training", {})
                if ledger.is_file()
                else {}
            )
            if state.get("tune_directory"):
                # load() has verified the delegated complete adaptive commit.
                row = state["entries"][plan["entries"][0]["id"]]
                if row["status"] == "completed" and row.get("checkpoint"):
                    checkpoint = Path(row["checkpoint"]).resolve()
                    matching = [
                        Path(attempt["run_dir"]).resolve()
                        for attempt in row.get("attempts", ())
                        if checkpoint.is_relative_to(Path(attempt["run_dir"]).resolve())
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            "Tune source has ambiguous training attempt ownership"
                        )
                    stage = {"status": "completed", "run_dir": str(matching[0])}
            if stage.get("status") not in {
                "completed",
                "early_stopped",
            } or not stage.get("run_dir"):
                raise ValueError(
                    "author source has no completed training stage to evaluate"
                )
            args.source = Path(stage["run_dir"])

        prepared = prepared_from_run(args.source)
        config = prepared.config
    _single_flags(args)
    config.evaluation = _options(args, config.evaluation)
    if args.output_root is not None:
        config.output.root = str(args.output_root)
    training = args.task != "evaluate"
    if training and config.training is None:
        raise ValueError("training task requires training settings")
    if args.task == "train-evaluate":
        _checkpoint_conditions(config)
    if args.source:
        from smartsom.experiments.composable import prepare_evaluation

        prepared, _, _ = prepare_evaluation(
            source=args.source,
            selection=config.evaluation.checkpoint,
            options=config.evaluation,
            output_root=args.output_root,
        )
    else:
        prepared = prepare_v3(config, training=training, require_dependencies=True)
        if not training:
            from smartsom.experiments.composable import prepare_evaluation

            prepared, _, _ = prepare_evaluation(prepared)
    if args.initialize_from:
        raise ValueError("v3 initialization belongs in policy model selectors")
    if args.task != "train" and config.evaluation.render_mode:
        cases = json.loads(prepared.evaluation_json)
        selected = config.evaluation.render_case or cases[0]["case"]
        if not any(
            case["case"] == selected
            and case["replication"] + 1 == config.evaluation.render_replication
            for case in cases
        ):
            raise ValueError(
                "render case/replication is absent from the selected evaluation"
            )
    root = Path(config.output.root).expanduser().resolve()
    parent = root
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError("output root has no writable directory parent")
    declarations = json.loads(prepared.policies_json)
    plan = {
        "status": "checked",
        "task": args.task,
        "input_type": "saved-run" if args.source else "experiment-v3",
        "executor": "native-composable",
        "output_root": str(root),
        "training": primitive(config.training) if training else None,
        "groups": {
            group: {
                "role": policy["role"],
                "kind": policy["implementation"]["kind"],
                "trainable": training and group in config.training.groups,
                "model": policy.get("resolved_model"),
            }
            for group, policy in declarations.items()
        },
        "validation": primitive(config.validation) if training else None,
        "evaluation": primitive(config.evaluation) if args.task != "train" else None,
        "preview": args.preview,
        "scope": "input/dependency checks only; no sampling or capacity measurement",
    }
    return plan, config


def _batch_flags(args, tuning=False):
    if tuning and args.retry_failed:
        raise ValueError(
            "new Tune batch has no failed attempts; use resume --retry-failed"
        )
    if (
        args.preview
        or args.initialize_from
        or any(
            getattr(args, name) is not None
            for name in (
                "checkpoint",
                "replications",
                "seed",
                "deterministic",
                "replay",
                "render_mode",
                "render_case",
                "render_replication",
                "record",
                "output_root",
            )
        )
        or args.baseline
        or args.scenario
    ):
        raise ValueError(
            "batch scientific/evaluation overrides belong in its frozen inputs"
        )
    if not tuning and (args.mode or args.execution):
        raise ValueError("--mode and --execution apply only to Tune batch")
    _legacy_flags(args)


def _legacy_flags(args):
    from smartsom.experiments.cli import FLAGS

    allowed = {
        "verbose",
        "progress",
        "log-format",
        "summary-interval",
        "progress-title",
        "seed",
        "output-root",
    }
    if any(
        getattr(args, key.replace("-", "_"), None) is not None
        for key in FLAGS
        if key not in allowed
    ):
        raise ValueError("explicit task scientific settings belong in YAML")


def _single_flags(args):
    _legacy_flags(args)
    if args.baseline:
        raise ValueError(
            "v3 comparison partners are selected through composition files"
        )
    if args.mode or args.execution or args.retry_failed:
        raise ValueError(
            "batch scheduling/retry flags do not apply to single experiments"
        )
    if args.task == "train" and any(
        getattr(args, key) is not None
        for key in (
            "checkpoint",
            "replications",
            "seed",
            "deterministic",
            "replay",
            "render_mode",
            "record",
            "render_case",
            "render_replication",
        )
    ):
        raise ValueError("evaluation options do not apply to train")


def execute(args):
    from smartsom import api

    plan, config = resolve(args)
    if args.command == "check":
        return plan
    if plan["input_type"] == "study":
        return api.run_study(args.study, retry_failed=args.retry_failed)
    if plan["input_type"] == "tune-batch":
        return api.tune_batch(
            batch=args.config, mode=args.mode, execution=args.execution
        )
    if args.task == "train":
        return primitive(api.train(config))
    if args.task == "train-evaluate":
        return primitive(api.train_evaluate(config))
    if not args.preview:
        return (
            primitive(
                api.evaluate(
                    args.source, config.evaluation, output_root=args.output_root
                )
            )
            if args.source
            else primitive(api.evaluate(config=config))
        )
    from smartsom.experiments.composable import (
        evaluate,
        prepare_evaluation,
    )
    from smartsom.telemetry.runtime import operation

    @operation("evaluation")
    def preview_run():
        checkpoint = None
        origin = None
        if args.source:
            prepared, checkpoint, origin = prepare_evaluation(
                source=args.source,
                selection=config.evaluation.checkpoint,
                options=config.evaluation,
            )
        else:
            prepared = prepare_v3(config, training=False, require_dependencies=True)
        saved = json.loads(prepared.config_json)
        saved["evaluation"] = primitive(config.evaluation)
        cases = json.loads(prepared.evaluation_json)[:1]
        prepared = replace(
            prepared,
            config_json=canonical_json(saved),
            evaluation_json=canonical_json(cases),
        )
        result = evaluate(prepared, output_root=args.output_root, purpose="preview")
        manifest = result.run_dir / "run.json"
        data = json.loads(manifest.read_text())
        data["purpose"] = "preview"
        if args.source:
            data["source_run_directory"] = str(args.source.resolve())
            data["checkpoint"] = str(checkpoint)
            data["source_scientific_sha256"] = origin
        from smartsom.experiments.evidence import write_json

        write_json(manifest, data)
        return primitive(result)

    return preview_run()


def resume(args):
    from smartsom import api

    root = args.source
    plan_file = root / "plan.json"
    schema = (
        json.loads(plan_file.read_text()).get("schema") if plan_file.is_file() else None
    )
    if schema == "smartsom.author-plan/v1":
        from smartsom.experiments.author_driver import execute_saved, load

        load(root)  # validate frozen identities before starting a new process
        if args.retry_failed:
            raise ValueError(
                "author plan resume retries unfinished stages automatically"
            )
        if getattr(args, "extension_module", None):
            raise ValueError(
                "resume uses frozen extension modules; cannot change inputs"
            )
        if getattr(args, "background", None):
            from smartsom.experiments.background import launch
            from smartsom.telemetry.runtime import OVERRIDES

            return launch(
                root,
                resume=True,
                display_options=OVERRIDES.get(),
                max_concurrent=getattr(args, "max_concurrent", None),
            )
        return execute_saved(root, max_concurrent=getattr(args, "max_concurrent", None))
    if (
        getattr(args, "background", None) is not None
        or getattr(args, "max_concurrent", None) is not None
        or getattr(args, "extension_module", None)
    ):
        raise ValueError(
            "background/concurrency resume flags require a new author plan"
        )
    if schema == "smartsom.tune-batch/v1":
        return api.resume_tune_batch(root, retry_failed=args.retry_failed)
    if schema == "smartsom.composable-study-plan/v1":
        return api.run_study(root, retry_failed=args.retry_failed)
    manifest = root / "run.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("schema") not in {
        "smartsom.experiment/v2",
        "smartsom.experiment/v3",
    }:
        raise ValueError(
            "resume requires a saved single training run, prepared Study or Tune batch"
        )
    if args.retry_failed:
        raise ValueError("--retry-failed applies only to Study/Tune")
    return primitive(api.resume(root))
