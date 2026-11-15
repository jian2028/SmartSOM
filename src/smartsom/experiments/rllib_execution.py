"""Compile frozen public run inputs into the shared official RLlib runtime."""

import json
import os
from pathlib import Path

from smartsom.config.base_diagnostic import generate_scenario, seed_for
from smartsom.config.rllib_compatibility import smoke_learner, validate_runtime
from smartsom.config.rllib_resources import RLlibResources
from smartsom.config.trial_resources import TrialResources


def trial_config(root):
    # Local execution uses the already selected interpreter and frozen checkout.
    # Ray reads this setting on first import, before local_runtime() is reached.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    from smartsom.experiments.reservation import arguments, load
    from smartsom.experiments.reservation_runtime import freeze, frozen_drain
    from smartsom.experiments.rllib_panels import compile_panels
    from smartsom.learning.rllib_config import build_algorithm_config
    from smartsom.learning.rllib_evaluation import configure_panel
    from smartsom.learning.rllib_worlds import TrainingWorlds

    root = Path(root).resolve()
    load(root)
    args, run = arguments(root)
    ex, factory, workload, options, paths = args._prepared_inputs[0]
    operations = json.loads((root / "execution.json").read_text())["settings"]
    validate_runtime(ex, options, operations)
    if run.task in {"calibrate", "evaluate"} or (
        run.task == "smoke" and not smoke_learner(options, run.methods)
    ):
        return scientific_task_config(root, run, ex)
    policy = options.learning_policy
    resources = RLlibResources.model_validate(operations.get("rllib", {}))
    drain = ex.drain_ticks if run.task == "smoke" else frozen_drain(args, paths)
    freeze(
        root,
        paths,
        run.task,
        {
            "seed": run.seed,
            "drain_ticks": drain,
            "tick_budget": ex.train_ticks if ex.train_episodes is None else None,
            "episode_budget": ex.train_episodes,
            "maximum_seconds": run.seconds,
            "runtime": "rllib",
            "tracking_group": getattr(args, "tracking_group", None),
            "experiment_short_name": getattr(args, "experiment_short_name", None),
        },
    )
    condition = ex.training_conditions[0]
    prototype = generate_scenario(
        *ex.inputs_for(condition, factory, workload),
        seed_for(run.seed, "rllib-space-prototype") % (2**31),
        condition,
        drain,
    )
    config = build_algorithm_config(
        scenario=prototype,
        options=options,
        algorithm=options.method,
        seed=run.seed,
        training=policy.learner.params,
        training_worlds=TrainingWorlds(ex, factory, workload, run.seed, drain),
        **resources.arguments(),
    )
    if config.num_env_runners == 0 and resources.env_runners.memory_gib is not None:
        raise ValueError("local EnvRunner memory belongs in rllib.resources.memory_gib")
    evaluation_mode = (
        "sample" if config.get_evaluation_config_object().explore else "argmax"
    )
    if run.task == "train" and (
        ex.periodic_evaluation.milestones or ex.final_evaluation
    ):
        panels = compile_panels(
            ex,
            args.experiment.parent,
            factory,
            workload,
            drain=drain,
            mode=evaluation_mode,
        )
        config = configure_panel(
            config,
            panels["periodic_evaluation"],
            test_cases=panels["test"],
            interval=None,
            evaluation_resources=resources.evaluation_arguments(),
            num_env_runners=resources.evaluation_num_env_runners
            or min(
                ex.concurrency.evaluation_processes,
                ex.concurrency.evaluation_cpus or ex.concurrency.evaluation_processes,
            ),
        )
    request = config.algo_class.default_resource_request(config)
    memory_budget = operations["resources"].get("memory_gib")
    declared = [bundle.get("smartsom_memory_gib") for bundle in request.bundles]
    if memory_budget is not None and not all(declared):
        raise ValueError(
            "total memory budget requires memory_gib declarations for every resident component; local components belong to the driver"
        )
    if memory_budget is None and any(declared):
        raise ValueError(
            "component memory declarations require profile resources.memory_gib"
        )
    return {
        "root": str(root),
        "task": run.task,
        "algorithm_config": config,
        "execution": {
            "seconds": run.seconds,
            "episodes": ex.train_episodes if run.task == "train" else None,
            "physical_ticks": run.smoke_ticks
            if run.task == "smoke"
            else ex.train_ticks
            if ex.train_episodes is None
            else None,
            "periodic_evaluation": {
                "unit": ex.periodic_evaluation.unit,
                "milestones": list(ex.periodic_evaluation.milestones)
                if run.task == "train"
                else [],
                "select_initial": ex.periodic_evaluation.select_initial,
            },
            "final_evaluation": ex.final_evaluation and run.task == "train",
            "final_checkpoint_selection": ["best", "last"],
            "checkpoint_retention": ex.checkpoint_retention,
        },
    }


def scientific_task_config(root, run, experiment):
    """Non-learning scientific tasks need a Tune actor, not a Learner."""
    operations = json.loads((root / "execution.json").read_text())["settings"]
    if (
        TrialResources.model_validate(operations.get("trial_resources", {}))
        != TrialResources()
    ):
        raise ValueError(
            "phase trial_resources were retired; use profile rllib.resources"
        )
    resources = RLlibResources.model_validate(operations.get("rllib", {}))
    driver = resources.resources
    if (operations["resources"].get("memory_gib") is None) != (
        driver.memory_gib is None
    ):
        raise ValueError(
            "scientific task memory requires both total and driver memory_gib"
        )
    if experiment.evaluation.recovery.action != "disabled":
        raise ValueError("recovery calibration requires the RLlib evaluation adapter")
    return {
        "root": str(root),
        "task": run.task,
        "scientific_task": True,
        "resource_bundles": resources.scientific_task_bundles(
            evaluation_workers=resources.panel_workers(experiment.concurrency)
            if run.task == "evaluate"
            else 0
        ),
        "execution": {"checkpoint_retention": experiment.checkpoint_retention},
    }
