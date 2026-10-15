"""Read composition evidence without loading models or reconstructing clocks."""

import json
import math
from numbers import Real
from statistics import mean

METRICS = ("return", "delivered", "flow_time", "waiting", "makespan", "throughput")


def aggregate_cases(rows, requested=None):
    usable = [
        row
        for row in rows
        if not row.get("engineering_failure")
        and row.get("status") in {"completed", "truncated"}
    ]
    result = {
        "requested": requested,
        "observed": len(rows),
        "missing": max(0, requested - len(rows)) if requested is not None else None,
        "completed": sum(row.get("status") == "completed" for row in usable),
        "truncated": sum(row.get("status") == "truncated" for row in usable),
        "exceptions": len(rows) - len(usable),
    }
    for key in METRICS:
        values = [
            row[key]
            for row in usable
            if isinstance(row.get(key), Real)
            and not isinstance(row[key], bool)
            and math.isfinite(row[key])
            and (key != "makespan" or row["status"] == "completed")
        ]
        result[f"mean_{key}"] = mean(values) if values else None
        result[f"{key}_samples"] = len(values)
    return result


def read_composition_results(root, metadata, *, max_rows):
    def read(path, default):
        if not path.is_file():
            return default
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError(f"report JSON exceeds size limit: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    prepared = read(root / "config/prepared.json", {})
    config = json.loads(prepared.get("config_json", "{}"))
    validation_cases = json.loads(prepared.get("validation_json", "null"))
    evaluation_cases = json.loads(prepared.get("evaluation_json", "null"))
    history = read(root / "reports/training.json", [])
    episodes = read(root / "reports/training-episodes.json", [])
    evaluations = metadata.get("results", [])
    count = len(history) + len(episodes) + len(evaluations)
    if count > max_rows:
        raise ValueError("report row limit exceeded; select a smaller run")
    series = []

    def curves(rows, prefix, x_key, label):
        for key in METRICS:
            points = []
            for row in rows:
                value = row.get(key)
                if (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or row.get("engineering_failure")
                    or (
                        "status" in row
                        and row["status"] not in {"completed", "truncated"}
                    )
                    or (
                        key == "makespan"
                        and "status" in row
                        and row["status"] != "completed"
                    )
                ):
                    value = None
                points.append([row.get(x_key), value])
            points = [p for p in points if isinstance(p[0], Real)]
            if points and any(isinstance(p[1], Real) for p in points):
                series.append(
                    {"label": f"{prefix} {key}", "x_label": label, "points": points}
                )

    # Old ledgers lack cumulative ticks. Keep their actual episode axis separate.
    physical = [r for r in episodes if "training_physical_ticks" in r]
    legacy = [r for r in episodes if "training_physical_ticks" not in r]
    curves(physical, "Training", "training_physical_ticks", "Training physical ticks")
    for env in sorted({str(r.get("env", r.get("case_id", "unknown"))) for r in legacy}):
        curves(
            [
                r
                for r in legacy
                if str(r.get("env", r.get("case_id", "unknown"))) == env
            ],
            f"Training env {env} (legacy)",
            "replication",
            "Episode (recorded)",
        )
    for group in sorted({g for r in history for g in r.get("optimizations", {})}):
        for physical_axis in (True, False):
            rows = [r for r in history if ("physical_ticks" in r) == physical_axis]
            if rows:
                series.append(
                    {
                        "label": f"Training optimizations/{group}",
                        "x_label": "Training physical ticks"
                        if physical_axis
                        else "Update (recorded)",
                        "points": [
                            [
                                r["physical_ticks"] if physical_axis else r["update"],
                                r.get("optimizations", {}).get(group),
                            ]
                            for r in rows
                        ],
                    }
                )
    ticks = {r["update"]: r["physical_ticks"] for r in history if "physical_ticks" in r}
    updates = {r["update"] for r in history}
    updates.update(
        int(p.stem.split("-")[-1]) for p in root.glob("logs/validation-*.json")
    )
    latest = max(updates | {metadata.get("updates", 0)})
    val = config.get("validation", {})
    if val.get("enabled") and val.get("every_updates", 0) > 0:
        updates.update(range(val["every_updates"], latest + 1, val["every_updates"]))
    validation = []
    for update in sorted(updates):
        path = root / "logs" / f"validation-{update:06d}.json"
        scheduled = (
            val.get("enabled")
            and val.get("every_updates", 0) > 0
            and update % val["every_updates"] == 0
        )
        if not scheduled and not path.is_file():
            continue
        rows = read(path, [])
        count += len(rows)
        if count > max_rows:
            raise ValueError("report row limit exceeded; select a smaller run")
        statistics = aggregate_cases(
            rows, len(validation_cases) if validation_cases is not None else None
        )
        validation.append(
            {
                "update": update,
                "physical_ticks": ticks.get(update),
                "saved": path.is_file(),
                "cases": rows,
                **statistics,
            }
        )
    # A partially missing clock cannot form a physical-time curve with honest
    # gaps. Use the saved update axis for that legacy curve, keeping known ticks
    # in the coverage table rather than connecting across unknown timestamps.
    physical_axis = all(r["physical_ticks"] is not None for r in validation)
    axis = "physical_ticks" if physical_axis else "update"
    label = (
        "Training physical ticks"
        if physical_axis
        else "Update (recorded; physical ticks unavailable)"
    )
    curves(
        [{**r, **{key: r[f"mean_{key}"] for key in METRICS}} for r in validation],
        "Validation",
        axis,
        label,
    )
    summary = {
        "status": metadata.get("status", "unknown"),
        "ended_episodes": len(episodes),
        "validation_saved": sum(r["saved"] for r in validation),
        "validation_scheduled": len(validation),
    }
    if metadata.get("kind") == "evaluation":
        summary.update(
            aggregate_cases(
                evaluations,
                len(evaluation_cases) if evaluation_cases is not None else None,
            )
        )
        identities = sorted(
            {
                (
                    str(r.get("case_id", "unknown")),
                    str(r.get("algorithm_id", "composition")),
                )
                for r in evaluations
            }
        )
        for case, algorithm in identities:
            curves(
                [
                    r
                    for r in evaluations
                    if (
                        str(r.get("case_id", "unknown")),
                        str(r.get("algorithm_id", "composition")),
                    )
                    == (case, algorithm)
                ],
                f"Independent evaluation {case}/{algorithm}",
                "replication",
                "Evaluation replication (recorded)",
            )
    return {
        "summary": summary,
        "series": series,
        "training": {
            "episodes": episodes,
            "updates": history,
            "validation": validation,
        },
        "evaluation": evaluations if metadata.get("kind") == "evaluation" else None,
        "note": "Only recorded clocks; missing validation is a gap. Metric sample counts are explicit; makespan uses complete non-error cases.",
    }
