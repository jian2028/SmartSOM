"""Bounded Ray case execution; only the caller publishes ordered receipts."""

from pathlib import Path


def _case(settings, task):
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.experiments.reservation_runtime import evaluate_case, identity

    prepared = load_inputs(settings["experiment"])
    if identity(prepared[-1]) != settings["inputs"]:
        raise ValueError("evaluation inputs changed before worker startup")
    return evaluate_case(settings, task, prepared)


def case_results(settings, tasks, prepared, *, workers, on_progress, capacity=None):
    from smartsom.experiments import control
    from smartsom.experiments.reservation_runtime import evaluate_case

    if not tasks:
        return
    if capacity is not None:
        raise ValueError("live evaluation resizing is replaced by Tune phase resources")
    if workers == 1 and not settings.get("worker_resources"):
        for task in tasks:
            control.boundary(settings["output"])
            yield (
                task["index"],
                evaluate_case(
                    settings,
                    task,
                    prepared,
                    on_progress=lambda row, index=task["index"]: on_progress(
                        index, row
                    ),
                ),
            )
        return
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        raise RuntimeError("parallel evaluation requires the owned Tune runtime")
    group = ray.util.get_current_placement_group()
    if group is None:
        raise RuntimeError("evaluation workers require a trial placement group")
    remote = ray.remote(max_retries=0)(_case).options(
        **(settings.get("worker_resources") or {"num_cpus": 1}),
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=group),
    )
    remaining = iter(tasks)
    pending, completed = {}, {}

    def fill():
        while len(pending) < workers:
            task = next(remaining, None)
            if task is None:
                break
            pending[remote.remote(settings, task)] = task["index"]

    try:
        fill()
        for task in tasks:
            index = task["index"]
            while index not in completed:
                control.boundary(Path(settings["output"]))
                ready, _ = ray.wait(list(pending), timeout=0.1)
                for reference in ready:
                    completed[pending.pop(reference)] = ray.get(reference)
                fill()
            yield index, completed.pop(index)
            fill()
    finally:
        for reference in pending:
            ray.cancel(reference, force=True)
