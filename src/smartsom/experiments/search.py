"""Declared search spaces, validation-only scores and optional Optuna state."""

from __future__ import annotations

import importlib
import itertools
import math
import random
from pathlib import Path
from statistics import mean

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.experiment import apply_overrides
from smartsom.experiments.catalog import read_json
from smartsom.experiments.evidence import _file_digest


def validate_search(config) -> None:
    options = config.search
    if not options.space:
        raise ConfigurationError("search.space must explicitly declare parameters")
    if options.objective is None or options.failure_policy != "all_complete":
        raise ConfigurationError(
            "search requires an explicit objective and failure_policy=all_complete"
        )
    if not options.seeds or len(set(options.seeds)) != len(options.seeds):
        raise ConfigurationError("search.seeds must contain distinct training seeds")
    if not config.validation.enabled or not config.checkpointing.save_last:
        raise ConfigurationError(
            "search requires validation.enabled and checkpointing.save_last"
        )
    updates = config.training.total_steps // config.training.steps_per_update
    if updates % config.validation.every_updates:
        raise ConfigurationError(
            "search requires a validation at the final update; change the recipe explicitly"
        )
    if options.pruning and options.method != "optuna":
        raise ConfigurationError("search.pruning requires method=optuna")
    if options.method in {"random", "optuna"} and options.trials is None:
        raise ConfigurationError(
            "random/optuna search requires an explicit trial budget"
        )
    first = []
    for name, space in options.space.items():
        if not name.startswith(("algorithm.", "training.")):
            raise ConfigurationError(
                f"search parameter must be under algorithm or training: {name}"
            )
        value = space.choices[0] if space.type == "categorical" else space.low
        # Resolve paths and dependent typed constraints before allocating output.
        first.append((name, value))
        if options.method == "grid" and space.type != "categorical":
            raise ConfigurationError(
                "grid search requires categorical choices for every parameter"
            )
        if space.type == "categorical" and len(
            {digest(v) for v in space.choices}
        ) != len(space.choices):
            raise ConfigurationError(f"duplicate search choices: {name}")
    apply_overrides(config, first)


def candidates(config) -> tuple[dict, ...]:
    """Freeze grid/random proposals; Optuna proposals depend on earlier evidence."""
    validate_search(config)
    options = config.search
    names = sorted(options.space)
    spaces = [options.space[name] for name in names]
    if options.method == "optuna":
        raise ConfigurationError(
            "Optuna candidates are generated through persisted ask/tell"
        )
    if options.method == "grid":
        values = tuple(itertools.product(*(space.choices for space in spaces)))
        if options.trials is not None and options.trials != len(values):
            raise ConfigurationError(
                "grid trial budget must equal the complete Cartesian product"
            )
    else:
        rng = random.Random(options.seed)

        def sample(space):
            if space.type == "categorical":
                return rng.choice(space.choices)
            if space.type == "float":
                return (
                    math.exp(rng.uniform(math.log(space.low), math.log(space.high)))
                    if space.log
                    else rng.uniform(space.low, space.high)
                )
            values = range(space.low, space.high + 1, space.step)
            if space.log:
                sampled = math.exp(
                    rng.uniform(math.log(space.low - 0.5), math.log(space.high + 0.5))
                )
                return min(space.high, max(space.low, math.floor(sampled + 0.5)))
            return rng.choice(values)

        values = tuple(
            tuple(sample(space) for space in spaces) for _ in range(options.trials)
        )
    return tuple(dict(zip(names, row, strict=True)) for row in values)


def trial_configs(config, parameters: dict) -> tuple:
    trial = apply_overrides(config, list(parameters.items()))
    validate_search(trial)
    return tuple(
        apply_overrides(trial, [("seed", seed)]) for seed in config.search.seeds
    )


def validation_score(report: dict, objective: str) -> float | None:
    """Only all-complete finite validation results have a comparison value."""
    if not report.get("episodes") or report.get("completed") != report["episodes"]:
        return None
    rows = report.get("results", [])
    successful = report.get("successful_inputs", [])
    if len(rows) != report["episodes"] or len(successful) != report["episodes"]:
        return None
    if len(set(successful)) != len(successful) or set(successful) != {
        row["input_id"] for row in rows
    }:
        return None
    if any(row.get("reason") != "completed" for row in rows):
        return None
    values = [row.get(objective) for row in rows]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        return None
    score = mean(values)
    if report.get("metrics", {}).get(objective) != score:
        raise ValueError("validation aggregate does not match its complete rows")
    return score


def final_validation(training_dir: Path, updates: int, objective: str) -> dict:
    path = training_dir / f"validation-{updates:06d}.json"
    report = read_json(path)
    if report.get("ppo_updates") != updates:
        raise ValueError("validation is not bound to the final training update")
    return {
        "value": validation_score(report, objective),
        "validation": str(path),
        "validation_sha256": _file_digest(path),
        "weights": report.get("weights"),
        "successful_inputs": report.get("successful_inputs", []),
        "completed": report.get("completed"),
        "episodes": report.get("episodes"),
    }


def require_optuna():
    try:
        return importlib.import_module("optuna")
    except ImportError as exc:
        raise RuntimeError(
            "Optuna search requires the optional search extra: uv sync --locked --extra search"
        ) from exc


def _seed(seed: int, index: int) -> int:
    return int(digest(["smartsom.optuna-sampler/v1", seed, index])[:8], 16)


class OptunaSession:
    """SQLite ask/tell state with a reproducible per-proposal TPE seed.

    A logical candidate can have multiple execution attempts. Failed attempts
    remain FAIL; explicit retries enqueue the exact candidate instead of sampling
    replacement parameters. Interrupted RUNNING attempts retain their trial id.
    """

    def __init__(self, root: Path, options):
        self.optuna = require_optuna()
        self.root, self.options = Path(root), options
        self.storage = f"sqlite:///{self.root / 'optuna.sqlite3'}"
        self.name = "smartsom-search"
        study = self._study(0, create=True)
        version = study.user_attrs.get("smartsom_optuna_version")
        if version is not None and version != self.optuna.__version__:
            raise ValueError("Optuna dependency changed; create a new search")
        if version is None:
            study.set_user_attr("smartsom_optuna_version", self.optuna.__version__)

    def _study(self, index, *, create=False):
        kwargs = {
            "study_name": self.name,
            "storage": self.storage,
            "sampler": self.optuna.samplers.TPESampler(
                seed=_seed(self.options.seed, index)
            ),
            "pruner": self.optuna.pruners.MedianPruner()
            if self.options.pruning
            else self.optuna.pruners.NopPruner(),
        }
        if create:
            return self.optuna.create_study(
                **kwargs,
                direction="minimize" if self.options.direction == "min" else "maximize",
                load_if_exists=True,
            )
        return self.optuna.load_study(**kwargs)

    def ask(self, index: int, *, parameters: dict | None = None) -> tuple[dict, dict]:
        study = self._study(index)
        distributions, choices = {}, {}
        for name, space in sorted(self.options.space.items()):
            data = primitive(space)
            if space.type == "categorical":
                # Indices support JSON-native structured choices such as hidden_sizes.
                choices[name] = list(space.choices)
                distributions[name] = self.optuna.distributions.CategoricalDistribution(
                    tuple(range(len(space.choices)))
                )
            elif space.type == "float":
                distributions[name] = self.optuna.distributions.FloatDistribution(
                    data["low"], data["high"], log=data["log"]
                )
            else:
                distributions[name] = self.optuna.distributions.IntDistribution(
                    data["low"], data["high"], step=data["step"], log=data["log"]
                )
        for saved in reversed(study.get_trials(deepcopy=False)):
            if saved.state.name != "RUNNING":
                continue
            proposal = saved.user_attrs.get("smartsom_proposal_index")
            if proposal not in (None, index):
                continue
            if set(saved.params) != set(distributions):
                raise ValueError(
                    "Optuna proposal was interrupted before all parameters were persisted"
                )
            values = {
                name: choices[name][value] if name in choices else value
                for name, value in saved.params.items()
            }
            if parameters is not None and digest(values) != digest(parameters):
                continue
            trial = self.optuna.trial.Trial(study, saved._trial_id)
            trial.set_user_attr("smartsom_proposal_index", index)
            return values, {
                "number": saved.number,
                "id": saved._trial_id,
                "optuna_version": self.optuna.__version__,
            }
        if parameters is not None:
            fixed = {
                name: next(
                    i
                    for i, value in enumerate(choices[name])
                    if digest(value) == digest(raw)
                )
                if name in choices
                else raw
                for name, raw in parameters.items()
            }
            study.enqueue_trial(fixed)
        trial = study.ask(fixed_distributions=distributions)
        trial.set_user_attr("smartsom_proposal_index", index)
        values = {
            name: choices[name][value] if name in choices else value
            for name, value in trial.params.items()
        }
        # Optuna's public Trial constructor requires its storage identity. Keep
        # this one internal-id access isolated; the locked version is recorded.
        return values, {
            "number": trial.number,
            "id": trial._trial_id,
            "optuna_version": self.optuna.__version__,
        }

    def trial(self, identity: dict):
        study = self._study(identity["number"])
        trial = self.optuna.trial.Trial(study, identity["id"])
        if trial.number != identity["number"]:
            raise ValueError("Optuna trial identity mismatch")
        return study, trial

    def report(self, identity: dict, value: float, step: int) -> bool:
        if not self.options.pruning:
            return False
        _, trial = self.trial(identity)
        trial.report(value, step)
        return trial.should_prune()

    def finish(self, identity: dict, status: str, value: float | None):
        study, trial = self.trial(identity)
        frozen = study.get_trials(deepcopy=False)[trial.number]
        if frozen.state.is_finished():
            return
        if status == "interrupted":
            return
        if status == "completed" and value is not None:
            study.tell(trial, value)
        else:
            state = (
                self.optuna.trial.TrialState.PRUNED
                if status == "pruned"
                else self.optuna.trial.TrialState.FAIL
            )
            study.tell(trial, state=state)
