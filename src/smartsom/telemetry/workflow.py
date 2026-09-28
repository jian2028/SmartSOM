"""Presentation metadata and planned-work estimates; no execution or learner state."""

import json
import math
import time
from collections import deque
from pathlib import Path
from string import Formatter

TITLE_FIELDS = {"id", "name", "algorithm", "H", "V", "travel", "seed"}


def validate_title_template(value):
    if value is not None:
        if not isinstance(value, str):
            raise ValueError("progress titles must be text")
        if not value.strip() or any(c in value for c in "\n\r\t"):
            raise ValueError("progress titles must be nonempty single-line text")
        for _, field, spec, conversion in Formatter().parse(value):
            if field is not None and (field not in TITLE_FIELDS or spec or conversion):
                raise ValueError(
                    "task title supports only " + ", ".join(sorted(TITLE_FIELDS))
                )
    return value


def task_title(template, fields, default):
    if not template:
        return default
    return template.format_map({key: fields.get(key, "N/A") for key in TITLE_FIELDS})


def describe_config(config, kind, *, scenario=None, validation=None, evaluation=None):
    training = config.get("training") or {}
    val, test = config.get("validation", {}), config.get("evaluation", {})
    v3 = config.get("schema") in {
        "smartsom.experiment-config/v3",
        "smartsom.execution-config/v1",
        "smartsom.execution-config/v2",
        "smartsom.experiment-config/v4",
    }
    total = training.get("total_ticks" if v3 else "total_steps", 0)
    interval = training.get("ticks_per_update" if v3 else "steps_per_update", 0)
    rounds = math.ceil(total / interval) if interval else 0
    factory = (scenario or {}).get("factory", {})
    grid = factory.get("grid", {})
    horizon = (
        (scenario or {}).get("tick_limit")
        or training.get("max_ticks")
        or training.get("max_decisions")
        or 1
    )

    def lengths(cases, options):
        if cases is not None:
            return [
                max(
                    1,
                    (item.get("scenario") or item.get("episode") or {}).get(
                        "tick_limit", horizon
                    ),
                )
                for item in cases
            ]
        return [horizon] * (
            max(1, len(options.get("scenarios", []))) * options.get("replications", 1)
        )

    val_horizons = (
        lengths(validation, val) if val.get("enabled", True) and total else []
    )
    eval_horizons = lengths(evaluation, test)
    matrix = (scenario or {}).get("transport_matrix")
    travel = "grid" if scenario else None
    if matrix:
        travel = matrix.get("source", "matrix")
        if travel == "manual" and all(item[2] == 0 for item in matrix.get("times", [])):
            travel = "zero"
    algorithm = (
        training.get("algorithm")
        or config.get("algorithm", {}).get("provider")
        or "composition"
    )
    return {
        "mode": kind,
        "name": config.get("output", {}).get("name", kind),
        "title": config.get("logging", {}).get("title"),
        "task_title": config.get("logging", {}).get("task_title"),
        "algorithm": algorithm.upper(),
        "training_seed": config.get("seed"),
        "training_seeds": [config["seed"]] if "seed" in config else [],
        "validation_seed": val.get("seed"),
        "evaluation_seed": test.get("seed"),
        "training_total": total,
        "training_unit": "physical ticks" if v3 else "adapter decisions",
        "round_size": interval,
        "round_total": rounds,
        "validation_every": val.get("every_updates", 0),
        "validation_rounds": rounds // val["every_updates"]
        if val_horizons and val.get("every_updates")
        else 0,
        "validation_cases": len(val_horizons),
        "validation_horizons": val_horizons,
        "evaluation_cases": len(eval_horizons),
        "evaluation_horizons": eval_horizons,
        "factory": factory.get("name") or factory.get("factory_id"),
        "factory_id": factory.get("factory_id"),
        "map": f"{grid['width']}×{grid['height']}"
        if "width" in grid and "height" in grid
        else None,
        "travel": travel,
        "run_total": (scenario or {}).get("tick_limit"),
        "runtime_mode": " · ".join(
            f"{key}={value}"
            for key, value in config.get("runtime", {}).items()
            if key in {"num_envs", "sampling_processes", "numerical_threads", "device"}
        ),
    }


def describe_prepared(prepared, kind):
    config = json.loads(prepared.config_json)
    if hasattr(prepared, "scenario_json"):
        scenario = json.loads(prepared.scenario_json)
    else:
        from smartsom.config.codec import primitive

        scenario = primitive(prepared.resolved.scenario)
    val = (
        json.loads(prepared.validation_json)
        if getattr(prepared, "validation_json", None)
        else None
    )
    test = (
        json.loads(prepared.evaluation_json)
        if getattr(prepared, "evaluation_json", None)
        else None
    )
    metadata = describe_config(
        config, kind, scenario=scenario, validation=val, evaluation=test
    )
    if hasattr(prepared, "resolved"):
        metadata["algorithm"] = prepared.resolved.algorithm.provider
        metadata["runtime_mode"] = " · ".join(
            f"{key}={value}"
            for key, value in config.get("runtime", {}).items()
            if key in {"num_envs", "sampling_processes", "numerical_threads", "device"}
        )
    return metadata


def describe_study(directory, plan):
    recipe = plan["recipe"]
    first = plan["entries"][0] if plan.get("entries") else {}
    metadata = {}
    try:
        saved = json.loads((Path(directory) / first["snapshot"]).read_text())
        metadata = describe_config(
            json.loads(saved["config_json"]),
            "study",
            scenario=json.loads(saved["scenario_json"]),
        )
    except (OSError, KeyError, ValueError):
        pass
    if metadata.get("factory_id"):
        # A batch includes heterogeneous variants. Do not label the whole batch
        # with the first variant's homogeneous display name.
        metadata["factory"] = metadata["factory_id"]
    total, interval = recipe.get("total_ticks", 0), recipe.get("ticks_per_update", 256)
    rounds = math.ceil(total / interval)
    seed = recipe.get("training_seed")
    logging = recipe.get("logging", {})
    metadata.update(
        mode="study",
        title=logging.get("title"),
        task_title=logging.get("task_title"),
        training_total=total,
        training_unit="physical ticks",
        round_size=interval,
        round_total=rounds,
        validation_every=recipe.get("validation_every_updates", 4),
        validation_rounds=rounds // recipe.get("validation_every_updates", 4),
        validation_cases=recipe.get("validation_cases", 5),
        evaluation_cases=recipe.get("evaluation_cases", 5),
        training_seed=seed,
        training_seeds=sorted(
            {
                entry.get("training_seed", seed)
                for entry in plan.get("entries", [])
                if entry.get("training_seed", seed) is not None
            }
        ),
        validation_seed=recipe.get("validation_seed"),
        evaluation_seed=recipe.get("evaluation_seed"),
        algorithms=sorted(
            {e.get("algorithm", "?").upper() for e in plan.get("entries", [])}
        ),
        H_cases=sorted({e.get("H_case", "?") for e in plan.get("entries", [])}),
        V_cases=sorted({e.get("V_case", "?") for e in plan.get("entries", [])}),
        transports=sorted({e.get("transport", "?") for e in plan.get("entries", [])}),
        max_concurrent=recipe.get("max_concurrent", 1),
        social_information="OFF",
        runtime_mode="每实验 envs=1 · CPU / 1 thread · 采样与验证顺序执行",
    )
    return metadata


def case_credit(values, prefix, horizons):
    ended = min(len(horizons), values.get(prefix + "_finished", 0) or 0)
    result = sum(horizons[:ended])
    if ended < len(horizons) and values.get(prefix + "_case_active", True):
        limit = values.get(prefix + "_tick_limit")
        if limit:
            result += horizons[ended] * min(
                1, max(0, (values.get(prefix + "_tick", 0) or 0) / limit)
            )
    return result


class WorkflowWork:
    """Count frozen budgets separately from physical ticks and optimization steps."""

    def __init__(self):
        self.started = time.monotonic()
        self.samples = deque()
        self.highwater = 0

    def overview(self, metadata, tasks, *, status="running", now=None):
        now = time.monotonic() if now is None else now
        mode = metadata.get("mode")
        training = mode in {"training", "train-evaluate", "study-entry"}
        testing = mode in {"evaluation", "train-evaluate", "study-entry"}
        val = metadata.get("validation_horizons", [])
        test = metadata.get("evaluation_horizons", [])
        total = (
            metadata.get("training_total", 0)
            + sum(val) * metadata.get("validation_rounds", 0)
            if training
            else 0
        ) + (sum(test) if testing else 0)
        rows = list(tasks.values())
        train_row = next(
            (
                r
                for r in rows
                if r.get("phase") == "training" or r.get("id") == "training"
            ),
            None,
        )
        if train_row is None and training:
            train_row = next(
                (
                    r
                    for r in rows
                    if r.get("unit")
                    in {
                        "physical ticks",
                        "environment steps",
                        "adapter decisions",
                        "sampling decisions",
                    }
                    and r.get("id") != "evaluation"
                ),
                None,
            )
        done = 0
        if training and train_row:
            values = train_row.get("values", {})
            done += min(
                metadata.get("training_total", 0), train_row.get("completed", 0) or 0
            )
            batches = values.get("validation_batches_finished", 0) or 0
            done += min(metadata.get("validation_rounds", 0), batches) * sum(val)
            if train_row.get("stage") == "validation":
                done += case_credit(values, "validation", val)
        if testing:
            evaluation = next((r for r in rows if r.get("id") == "evaluation"), None)
            if evaluation:
                done += case_credit(evaluation.get("values", {}), "evaluation", test)
        if mode == "run":
            total = metadata.get("run_total") or 0
            done = max(
                (
                    max(
                        r.get("completed", 0) or 0,
                        r.get("values", {}).get("evaluation_tick", 0) or 0,
                    )
                    for r in rows
                ),
                default=0,
            )
        if status == "completed":
            done = total
        done = min(total, max(self.highwater, done))
        self.highwater = done
        elapsed = max(0, now - self.started)
        if not self.samples or now - self.samples[-1][0] >= 5:
            self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > 300:
            self.samples.popleft()
        eta = None
        if len(self.samples) > 1:
            span, advance = now - self.samples[0][0], done - self.samples[0][1]
            if span >= 30 and advance > 0 and status == "running":
                eta = (total - done) * span / advance
        if total and done >= total:
            eta = 0
        return {
            "work_total": total,
            "work_completed": done,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        }


def validate_workflow(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("invalid runtime workflow")
    validate_title_template(metadata.get("task_title"))
    for key in ("title", "factory", "map", "algorithm", "mode", "runtime_mode"):
        value = metadata.get(key)
        if value is not None and (
            not isinstance(value, str) or any(c in value for c in "\n\r\t")
        ):
            raise ValueError("invalid runtime workflow text")
    for key in (
        "training_total",
        "round_size",
        "round_total",
        "validation_every",
        "validation_rounds",
        "validation_cases",
        "evaluation_cases",
        "run_total",
    ):
        value = metadata.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid runtime workflow count")
    for key in ("training_seeds", "validation_horizons", "evaluation_horizons"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("invalid runtime workflow list")
    for key in ("algorithms", "H_cases", "V_cases", "transports"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError("invalid runtime workflow labels")
