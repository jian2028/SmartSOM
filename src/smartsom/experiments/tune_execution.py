"""One stock Tune controller for a frozen cohort of scientific trials.

The caller owns stage dependencies and the frozen trial manifest. Ray owns
admission, rolling replacement, actor lifecycle and configured failure retries.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from smartsom.config.trial_resources import check_capacity


def trial_config(root, *, memory_budget=False):
    from smartsom.experiments.rllib_execution import trial_config as compile_trial

    return compile_trial(root)


@contextmanager
def local_runtime(resources):
    """Start and stop only this driver's own local Ray instance."""
    # Configure Ray before its first import; never patch private constants.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    import ray

    if ray.is_initialized():
        raise RuntimeError("Tune execution requires an owned local Ray runtime")
    options = {
        "address": "local",
        "include_dashboard": False,
        "runtime_env": {"py_executable": sys.executable},
    }
    for source, target in (("cpus", "num_cpus"), ("gpus", "num_gpus")):
        if resources.get(source) is not None:
            value = resources[source]
            if source == "cpus":
                if value != int(value):
                    raise ValueError(
                        "Ray local runtime requires a whole-number total CPU count"
                    )
                value = int(value)
            options[target] = value
    if resources.get("memory_gib") is not None:
        options["resources"] = {"smartsom_memory_gib": resources["memory_gib"]}
    try:
        ray.init(**options)
        yield ray
    finally:
        ray.shutdown()


def restore_required(storage, roots, *, checkpoint_available, requested=None):
    """A started experiment cannot silently become a new Tune search/cohort."""
    storage = Path(storage)
    started = (storage.exists() and any(storage.iterdir())) or any(
        (Path(root) / name).exists()
        for root in roots
        for name in (
            "episodes.jsonl",
            "progress.json",
            "continuation/current.json",
            "trials",
        )
    )
    if requested is True and not checkpoint_available:
        raise ValueError("Tune experiment checkpoint is missing; cannot restore")
    if not checkpoint_available and started:
        raise ValueError(
            "started run has no Tune experiment checkpoint; refusing a fresh run"
        )
    if requested is False and checkpoint_available:
        raise ValueError("Tune experiment already exists; restore the frozen run")
    return checkpoint_available


def fit_cohort(
    roots,
    *,
    storage,
    resources,
    max_concurrent=None,
    max_failures=0,
    resume=None,
    retry_failed=False,
    callbacks=(),
    search=None,
    task_ids=None,
):
    """Run or restore the same frozen cohort, never append to an active Tuner."""
    import threading

    if search is not None:
        from smartsom.config.search import reject_failed_retry

        reject_failed_retry(retry_failed)

    if (
        threading.current_thread() != threading.main_thread()
        or os.getenv("TUNE_DISABLE_SIGINT_HANDLER", "0") != "0"
    ):
        raise ValueError(
            "whole-run control requires Tune SIGINT handling on the main thread"
        )
    configs = [
        trial_config(root, memory_budget=resources.get("memory_gib") is not None)
        for root in roots
    ]
    if not configs:
        raise ValueError("a Tune cohort must contain at least one trial")
    if len({config["root"] for config in configs}) != len(configs):
        raise ValueError("each trial must own a distinct scientific output directory")
    retention = {config["execution"]["checkpoint_retention"] for config in configs}
    if len(retention) != 1:
        raise ValueError("one Tune cohort requires one checkpoint_retention policy")
    keep_all = retention.pop() == "all"
    storage = Path(storage).resolve()
    from smartsom.telemetry.runtime import backend_diagnostics

    with backend_diagnostics(), local_runtime(resources) as ray:
        from ray import tune
        from ray.tune.schedulers import FIFOScheduler

        from smartsom.experiments.rllib_trainable import (
            CheckpointReferences,
            RLlibTrainable,
            checkpoint_config,
        )
        from smartsom.experiments.tune_control import WholeRunControl
        from smartsom.experiments.tune_progress import ScientificProgress

        for config in configs:
            request = RLlibTrainable.default_resource_request(config)
            check_capacity(request.bundles, ray.cluster_resources())
        trainable = RLlibTrainable
        # Grid-search whole configurations, preserving heterogeneous trial
        # inputs without a second search algorithm or application admission loop.
        search_algorithm = None
        if search is not None:
            if len(configs) != 1:
                raise ValueError(
                    "a search requires one explicit training-seed template"
                )
            from smartsom.experiments.tune_search import build_search

            configs[0]["search"] = search.model_dump(mode="json")
            space = {"trial": configs[0]}
            search_algorithm = build_search(search)
        else:
            space = {"trial": tune.grid_search(configs)}
        resume = restore_required(
            storage,
            roots,
            checkpoint_available=tune.Tuner.can_restore(str(storage)),
            requested=resume,
        )
        if resume:
            tuner = tune.Tuner.restore(
                str(storage),
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=retry_failed,
                param_space=space,
            )
        else:
            from smartsom.experiments.evidence import write_json

            storage.mkdir(parents=True, exist_ok=True)
            write_json(
                storage / "smartsom-execution.json",
                {
                    "schema": "smartsom.tune-execution/v1",
                    "roots": [config["root"] for config in configs],
                },
            )
            tuner = tune.Tuner(
                trainable,
                param_space=space,
                tune_config=tune.TuneConfig(
                    scheduler=FIFOScheduler(),
                    search_alg=search_algorithm,
                    num_samples=search.num_samples if search else 1,
                    reuse_actors=False,
                    max_concurrent_trials=max_concurrent,
                ),
                run_config=tune.RunConfig(
                    storage_path=str(storage.parent),
                    name=storage.name,
                    callbacks=[
                        CheckpointReferences(),
                        WholeRunControl([config["root"] for config in configs]),
                        ScientificProgress(task_ids),
                        *callbacks,
                    ],
                    failure_config=tune.FailureConfig(max_failures=max_failures),
                    checkpoint_config=checkpoint_config(keep_all=keep_all),
                    verbose=0,
                ),
            )
        return tuner.fit()
