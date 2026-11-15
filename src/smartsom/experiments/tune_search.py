"""Lazy official OptunaSearch construction and explicit panel objectives."""

import json
import math
from pathlib import Path


def build_search(search):
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    space = {}
    for path, spec in search.parameters.items():
        if spec.distribution == "choice":
            space[path] = tune.choice(spec.values)
        elif spec.distribution == "randint":
            space[path] = tune.randint(int(spec.lower), int(spec.upper))
        else:
            space[path] = getattr(tune, spec.distribution)(spec.lower, spec.upper)
    from smartsom.experiments.tune_search_adapter import CensoredEvaluationSearch

    return CensoredEvaluationSearch(
        OptunaSearch(
            space=space,
            metric=objective_names(search),
            mode=[objective.mode for objective in search.objectives],
            seed=search.seed,
        )
    )


def objective_names(search):
    """Stable report keys, ordered to match Optuna directions, not priorities."""
    return [f"search_objective_{index}" for index in range(len(search.objectives))]


def panel_objective(directory, objective):
    """Never substitute final-test results or an arbitrary penalty for nulls."""
    path = Path(directory) / objective.panel / "summary.json"
    value = json.loads(path.read_text())
    for key in objective.metric.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(
                f"periodic evaluation metric is absent: {objective.metric}"
            )
        value = value[key]
    if value is None:
        return None  # Censored scientific evidence is not a fabricated penalty.
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(
            f"periodic evaluation objective is not finite: {objective.metric}"
        )
    return value


def pareto_candidates(results, search):
    """Use Optuna's public Pareto selection on observed completed candidates."""
    import optuna

    study = optuna.create_study(
        directions=[
            "minimize" if o.mode == "min" else "maximize" for o in search.objectives
        ]
    )
    keys = objective_names(search)
    for result in results:
        if result.error or not result.metrics.get("done"):
            continue
        values = [result.metrics.get(key) for key in keys]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                values=values,
                user_attrs={"path": str(result.path)},
            )
        )
    return [trial.user_attrs["path"] for trial in study.best_trials]
