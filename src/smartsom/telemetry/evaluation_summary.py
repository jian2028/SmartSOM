"""Hierarchical summaries with explicit coverage and no success-only cost means."""

from collections import defaultdict
from statistics import mean, stdev

from smartsom.telemetry.research_metrics import aggregate

METRICS = (
    "delivery_fraction",
    "drained",
    "joint_cost",
    "qualified_coverage",
    "mean_tardiness",
    "final_passing_rate",
    "tardiness_p95",
    "on_time_coverage",
    "mean_flow_time",
    "throughput_per_1000_ticks",
    "common_throughput",
)


def case_values(row):
    result = {**aggregate(row), **row.get("research_metrics", {}).get("summary", {})}
    result["common_throughput"] = row.get("common_window", {}).get("throughput")
    return result


def hierarchical(rows, *, planned_cases=None):
    """Repeats -> worlds -> conditions, all equal within each nested level."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["scenario_seed"])].append(case_values(row))
    metrics = {}
    for metric in METRICS:
        conditions = defaultdict(list)
        missing = False
        for (condition, _), repeats in groups.items():
            values = [r.get(metric) for r in repeats]
            if any(v is None for v in values):
                missing = True
                continue
            conditions[condition].append(mean(values))
        metrics[metric] = (
            None
            if missing or not conditions
            else mean(mean(v) for v in conditions.values())
        )
    return {
        "metrics": metrics,
        "actual_cases": len(rows),
        "planned_cases": planned_cases if planned_cases is not None else len(rows),
        "complete_inventory": planned_cases is None or len(rows) == planned_cases,
        "weighting": "repeat_mean/world_equal/condition_equal",
    }


def across_seeds(values, planned_seeds):
    available = [v for v in values if v is not None]
    return {
        "mean": mean(available) if available else None,
        "sample_sd": stdev(available) if len(available) > 1 else None,
        "actual_seeds": len(available),
        "planned_seeds": planned_seeds,
        "status": "complete" if len(available) == planned_seeds else "provisional",
    }


def paired_robustness(rows):
    by_key = {
        (r["method"], r["scenario_seed"], r["action_seed"], r["condition"]): r
        for r in rows
    }
    result = []
    for (method, world, repeat, condition), row in by_key.items():
        if condition == "normal":
            continue
        nominal = by_key.get((method, world, repeat, "normal"))
        a = nominal.get("common_window", {}) if nominal else {}
        b = row.get("common_window", {})
        valid = (
            a.get("status") == b.get("status") == "observed"
            and a.get("end_tick") == b.get("end_tick")
            and a.get("throughput", 0) > 0
        )
        result.append(
            {
                "method": method,
                "scenario_seed": world,
                "action_seed": repeat,
                "condition": condition,
                "ratio": b["throughput"] / a["throughput"] if valid else None,
                "status": "observed" if valid else "missing_or_zero_reference",
            }
        )
    return result


def paired_sig(base_rows, social_rows):
    def key(r):
        i = r["evaluation_identity"]
        return (
            r["scenario_seed"],
            r["condition"],
            r["action_seed"],
            r["selection"],
            i.get("split"),
        )

    if {key(r) for r in base_rows} != {key(r) for r in social_rows}:
        raise ValueError("SIG requires the same world/condition/repeat/selection panel")
    if len({key(r) for r in base_rows}) != len(base_rows) or len(base_rows) != len(
        social_rows
    ):
        raise ValueError("SIG panel contains duplicate or unequal cases")
    paired = {key(r): r for r in social_rows}
    for base in base_rows:
        social = paired[key(base)]
        left, right = base["evaluation_identity"], social["evaluation_identity"]
        first, second = left.get("sl_comparison", {}), right.get("sl_comparison", {})
        if first.get("status") != "supported" or second.get("status") != "supported":
            return {
                "sig": None,
                "absolute_sl_minus_base": None,
                "status": "comparison_identity_missing",
            }
        if (
            first["non_sl_sha256"] != second["non_sl_sha256"]
            or first["arm"] != "base"
            or second["arm"] == "base"
        ):
            raise ValueError("SIG requires Base/SL with equal non-SL scientific inputs")
        for field in ("world_sha256", "source_sha256", "runtime_sha256"):
            if not left.get(field) or left[field] != right.get(field):
                raise ValueError(
                    "SIG world/source/runtime identities differ or are missing"
                )
        x, y = left.get("training_progress") or {}, right.get("training_progress") or {}
        if any(
            x.get(field) is None or x[field] != y.get(field)
            for field in ("train_seed", "episodes")
        ):
            raise ValueError("SIG requires paired training seed and episode budget")
    a, b = (
        hierarchical(rows)["metrics"]["joint_cost"] for rows in (base_rows, social_rows)
    )
    return {
        "absolute_sl_minus_base": b - a if a is not None and b is not None else None,
        "sig": (a - b) / a if a is not None and a > 0 and b is not None else None,
        "status": "observed"
        if a is not None and a > 0 and b is not None
        else "missing_or_zero_reference",
        "definition": "cost improvement after within-seed aggregation",
    }


def sl_comparison_identity(algorithm, experiment, paths):
    """Only the known Base/Social observation and pool changes are exempted.

    Actions, rewards, bindings, mover, learner settings and encoder width remain
    in the common signature. Custom networks are not guessed to be comparable.
    """
    import hashlib

    from smartsom.telemetry.research_metrics import fingerprint

    document = algorithm.model_dump(mode="json", by_alias=True)
    learning = [
        p
        for p in document["production"]["policies"].values()
        if p["kind"] == "learning"
    ]
    if len(learning) != 1:
        return {"status": "unsupported", "reason": "requires_single_learning_policy"}
    policy = learning[0]
    observation, encoder = policy["observation"], policy["encoder"]
    prefix = "smartsom.representations."
    if (
        policy["network"]
        or not encoder
        or observation["target"]
        not in {
            prefix + arm + ".observation.create_observation"
            for arm in ("base", "social")
        }
        or encoder["target"]
        not in {prefix + arm + ".encoder.create_encoder" for arm in ("base", "social")}
    ):
        return {
            "status": "unsupported",
            "reason": "custom_representation_requires_explicit_comparison",
        }
    arm = (
        observation["params"].get("arm", "all")
        if ".social." in observation["target"]
        else "base"
    )
    policy["observation"] = {
        "family": "builtin_base_social",
        "params": {k: v for k, v in observation["params"].items() if k != "arm"},
    }
    encoder_params = {
        k: v for k, v in encoder["params"].items() if k != "pool" and v is not None
    }
    encoder_params.setdefault("width", policy["learner"]["params"].get("width"))
    policy["encoder"] = {"family": "builtin_base_social", "params": encoder_params}
    return {
        "status": "supported",
        "arm": arm,
        "non_sl_sha256": fingerprint(
            {
                "algorithm": document,
                "factory": hashlib.sha256(paths["factory"].read_bytes()).hexdigest(),
                "workload": hashlib.sha256(paths["workload"].read_bytes()).hexdigest(),
                "training": {
                    key: getattr(experiment, key)
                    for key in (
                        "train_ticks",
                        "train_episodes",
                        "budget_boundary",
                        "condition_sampling",
                        "training_conditions",
                        "training_world_domain",
                        "training_world_prefix",
                        "training_seed_modulus",
                        "drain_ticks",
                        "drain_frozen",
                    )
                },
                "validation": experiment.validation.model_dump(mode="json"),
                "evaluation": experiment.evaluation.model_dump(
                    mode="json", exclude={"learning_target"}
                ),
            }
        ),
        "allowed_differences": [
            "builtin observation base/social and arm",
            "builtin encoder base/social and pool",
        ],
    }


def reference_definition(experiment, factory, workload, algorithm):
    """Shared deterministic Rule target, independent of learning representation."""
    from smartsom.algorithms.base_policy import evaluation_recipe
    from smartsom.telemetry.research_metrics import fingerprint, primitive

    recipe = evaluation_recipe(algorithm, "rule").model_dump(mode="json", by_alias=True)
    recipe["traffic"]["mover"].pop("sha256", None)
    definition = {
        "factory": primitive(factory),
        "workload": workload.model_dump(mode="json", by_alias=True),
        "rule_recipe": recipe,
        "validation_cases": experiment.validation_cases,
        "fixed_cases_sha256": experiment.case_set_sha256.get("validation"),
        "world_seed_namespace": 80000,
        "conditions": ["normal", "low", "medium", "high"],
        "protocol": experiment.evaluation.model_dump(
            mode="json", by_alias=True, exclude={"learning_target"}
        ),
    }
    return {"definition": definition, "sha256": fingerprint(definition)}


def reference_worlds(rows):
    """Unique frozen worlds: deterministic Rule ignores decoder/repetition."""
    return sorted(
        {
            (
                r["scenario_seed"],
                r["condition"],
                r["evaluation_identity"]["world_sha256"],
            )
            for r in rows
        }
    )
