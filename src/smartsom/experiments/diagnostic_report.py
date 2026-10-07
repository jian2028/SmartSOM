"""Reusable analysis of saved native results; never trains or rewrites inputs.

Run ``python -m smartsom.experiments.diagnostic_report --run ROOT --output NEW``.
Supply --baseline for paired worlds; old missing diagnostics stay unavailable.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from smartsom._filesystem import native_path

REPORT_CAP = 16 * 1024 * 1024


def reward_components(row):
    contract = row.get("task_reward_contract")
    if not contract:
        return {
            "status": "unavailable",
            "reason": "no declared shipment reward contract",
        }
    quality = row.get("privileged_output_quality", {})
    values = [
        quality.get("shipped"),
        quality.get("passing_rate"),
        row.get("accumulated_overdue_time"),
    ]
    # The reward contract explicitly defines empty-output passing contribution as
    # zero. The observed passing-rate metric remains None.
    if values[0] == 0 and values[1] is None:
        values[1] = 0.0
    keys = (
        "shipment_weight",
        "passing_weight",
        "tardiness_weight",
        "reference_jobs",
        "reference_ticks",
    )
    if any(k not in contract for k in keys) or any(v is None for v in values):
        return {
            "status": "unavailable",
            "reason": "missing recorded component or denominator",
        }
    parts = {
        "shipment": contract["shipment_weight"]
        * values[0]
        / contract["reference_jobs"],
        "passing": contract["passing_weight"] * values[1],
        "tardiness": -contract["tardiness_weight"]
        * values[2]
        / (contract["reference_jobs"] * contract["reference_ticks"]),
    }
    return {
        "status": "available",
        "components": parts,
        "sum": sum(parts.values()),
        "reference_jobs": contract["reference_jobs"],
        "reference_ticks": contract["reference_ticks"],
        "observed_return": row.get("return"),
        "scope": "whole observed episode task objective, excluding learner shaping",
    }


def cohort_metrics(sim, provenance=None):
    # Require explicit immutable demand metadata; never guess from IDs or routes.
    pool = (provenance or {}).get("pool", [])
    mapping = {
        r["id"]: r["novel"] for r in pool if "id" in r and type(r.get("novel")) is bool
    }
    result = {}
    for label, flag in (("common", False), ("novel", True)):
        ids = {d for d in sim.demands if mapping.get(d) is flag}
        known = bool(mapping)
        shipped = ids & set(sim.shipped)
        good = ids & set(sim._qualified_shipments)
        result[label] = {
            "declared_demands": len(ids) if known else None,
            "released_demands": len(ids & sim.released) if known else None,
            "shipped": len(shipped) if known else None,
            "good_shipped": len(good) if known else None,
            "shipment_fraction": len(shipped) / len(ids) if ids else None,
            "passing_fraction": len(good) / len(shipped) if shipped else None,
        }
    result["unknown_cohort_demands"] = len(set(sim.demands) - mapping.keys())
    result["assignment_complete"] = set(sim.demands) <= mapping.keys()
    result["scope"] = (
        "fixed demand denominator, no unreleased exclusion; passing conditioned on shipment"
    )
    return result


def interval_diagnostics(history, interval_updates=1):
    from smartsom.config.diagnostics import DiagnosticReportOptions

    interval_updates = DiagnosticReportOptions(
        interval_updates=interval_updates
    ).interval_updates
    boundaries, pending = [], []
    for row in history:
        if pending and row.get("diagnostic_capture") != pending[-1].get(
            "diagnostic_capture"
        ):
            boundaries.append(pending)
            pending = []
        pending.append(row)
        if len(pending) == interval_updates:
            boundaries.append(pending)
            pending = []
    if pending:
        boundaries.append(pending)
    previous, result = {}, []
    for window in boundaries:
        row = window[-1]
        groups = {}
        current = row.get("learner_diagnostics", {}).get("groups", {})
        for group, state in current.items():
            metrics = {}
            old = previous.get(group, {}).get("metrics", {})
            for name, metric in state.get("metrics", {}).items():
                before = old.get(name, {"weight": 0, "mean": None})
                weight = metric.get("weight", 0) - before.get("weight", 0)
                total = metric.get(
                    "weighted_sum", (metric.get("mean") or 0) * metric.get("weight", 0)
                )
                old_total = before.get(
                    "weighted_sum", (before.get("mean") or 0) * before.get("weight", 0)
                )
                metrics[name] = {
                    "mean": (total - old_total) / weight if weight > 0 else None,
                    "weight": max(weight, 0),
                    "status": "available" if weight > 0 else "unavailable",
                }
            from smartsom.learning.production_diagnostics import explained_variance

            current_moments = state.get("value_moments", {})
            old_moments = previous.get(group, {}).get("value_moments", {})
            moments = {k: v - old_moments.get(k, 0) for k, v in current_moments.items()}
            groups[group] = {
                "metrics": metrics,
                "value_explained_variance": explained_variance(moments),
                "value_moments": moments,
            }
        result.append(
            {
                "update": row.get("update"),
                "update_start": window[0].get("update"),
                "updates_observed": len(window),
                "diagnostic_capture": row.get("diagnostic_capture"),
                "physical_ticks": row.get("physical_ticks"),
                "groups": groups,
            }
        )
        previous = current
    return result


def paired_differences(rows, baseline):
    def key(row):
        return row.get("case_id"), row.get("replication"), row.get("seed")

    indexed = {}
    for row in baseline:
        k = key(row)
        if k in indexed:
            raise ValueError(
                "duplicate baseline world key; provide one policy/checkpoint suite"
            )
        indexed[k] = row
    seen, pairs = set(), []
    for row in rows:
        k = key(row)
        if k in seen:
            raise ValueError("duplicate candidate world key; do not pool checkpoints")
        seen.add(k)
        other = indexed.get(k)
        verified = bool(
            other
            and row.get("world_sha256")
            and row.get("world_sha256") == other.get("world_sha256")
        )
        mismatch = bool(
            other
            and row.get("world_sha256")
            and other.get("world_sha256")
            and row["world_sha256"] != other["world_sha256"]
        )
        match = (
            "world_mismatch"
            if mismatch
            else "verified_world"
            if verified
            else "identity_only_unverified"
            if other
            else "missing_baseline"
        )
        valid = bool(
            other
            and not mismatch
            and not row.get("engineering_failure")
            and not other.get("engineering_failure")
        )
        delta = {}
        for name in (
            "return",
            "delivered",
            "physical_ticks",
            "fixed_job_makespan",
            "throughput",
            "total_tardiness",
        ):
            a, b = row.get(name), other.get(name) if other else None
            delta[name] = (
                a - b
                if valid
                and isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and math.isfinite(a)
                and math.isfinite(b)
                else None
            )
        pairs.append(
            {
                "world": list(k),
                "matching": match,
                "candidate_minus_baseline": delta,
                "candidate_status": row.get("status"),
                "baseline_status": other.get("status") if other else None,
            }
        )
    return {
        "pairs": pairs,
        "requested": len(rows),
        "matched": sum(
            p["matching"] not in {"missing_baseline", "world_mismatch"} for p in pairs
        ),
        "verified": sum(p["matching"] == "verified_world" for p in pairs),
        "unused_baseline_worlds": len(indexed.keys() - seen),
    }


def build_report(
    *,
    evaluation=(),
    baseline=(),
    episodes=None,
    partial=(),
    development=(),
    history=(),
    interval_updates=1,
):
    return {
        "schema": "smartsom.diagnostic-report/v1",
        "interpretation": "coverage and descriptive conditions are not evidence of policy improvement or causal blame",
        "evaluation": {
            "cases": len(evaluation),
            "zero_cases_failure": not bool(evaluation),
            "rows": [
                {
                    "world": [r.get("case_id"), r.get("replication"), r.get("seed")],
                    "reward": reward_components(r),
                    "cohorts": r.get("cohorts"),
                    "dispatcher_diagnostics": r.get("dispatcher_diagnostics"),
                    "decision_diagnostics": r.get("decision_diagnostics"),
                }
                for r in evaluation
            ],
        },
        "paired_baseline": paired_differences(evaluation, baseline)
        if baseline
        else {"status": "unavailable"},
        "training_whole_episodes": {
            "status": "available" if episodes is not None else "unavailable",
            "rows": episodes,
        },
        "training_partial_episodes": list(partial),
        "development": list(development),
        "interval_learner_diagnostics": interval_diagnostics(history, interval_updates),
    }


def write_report(path, report):
    data = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode(
        "utf-8"
    )
    if len(data) > REPORT_CAP:
        # Retain the analytical evidence, explicitly omit bulky bounded examples.
        import copy

        report = copy.deepcopy(report)
        report["omissions"] = [
            "event and decision examples removed to meet report byte budget"
        ]

        def trim(value):
            if isinstance(value, dict):
                for key in ("snippets", "recent", "examples"):
                    value.pop(key, None)
                for child in value.values():
                    trim(child)
            elif isinstance(value, list):
                for child in value:
                    trim(child)

        trim(report)
        data = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode(
            "utf-8"
        )
    if len(data) > REPORT_CAP:
        raise ValueError(
            "diagnostic report exceeds 16 MiB; select a smaller explicit suite"
        )
    path = native_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Standalone consumers cannot overwrite source results or previous analyses.
    with path.open("xb") as stream:
        stream.write(data)
    return path


def read_results(root):
    root = native_path(root)
    paths = sorted(root.glob("case-*/result.json"))
    if not paths:
        paths = sorted(root.glob("evaluation/case-*/result.json"))
    if not paths:
        paths = sorted(root.glob("evidence/case-*/result.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths], paths


def read_committed_suite(root):
    """Thin adapter for existing Batch04 committed JSON, never checkpoint pickle."""
    rows, sources = [], []
    for path in sorted(native_path(root).glob("*/committed.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("identity", {}).get("schema") != "batch04.evaluation-case/v2":
            raise ValueError("unsupported committed evaluation schema")
        sources.append(path)
        for relative, expected in data.get("files", {}).items():
            file = (path.parent / relative.replace("\\", "/")).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError("committed source path escapes its case")
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError("committed source hash mismatch")
            sources.append(file)
        row = dict(data["row"])
        row["world_sha256"] = data["identity"].get("world_sha256")
        rows.append(row)
    return rows, sources


def batch04_report(root, baseline_root=None):
    root = native_path(root)
    rows, sources = read_committed_suite(root / "final-primary")
    baseline, baseline_sources = (
        read_committed_suite(baseline_root) if baseline_root else ([], [])
    )
    sources += baseline_sources
    path = root / "training-episodes.json"
    episodes = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if path.exists():
        sources.append(path)
    history = []
    path = root / "updates.jsonl"
    if path.exists():
        sources.append(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                history.append(
                    {
                        "update": row["update"],
                        "physical_ticks": row.get("ticks"),
                        "learner_diagnostics": row.get("learning_diagnostics", {}).get(
                            "native_diagnostics", {}
                        ),
                    }
                )
    development = []
    for stage in sorted((root / "dev").glob("*")):
        if stage.is_dir():
            cases, paths = read_committed_suite(stage)
            sources += paths
            development.append({"checkpoint": stage.name, "rows": cases})
    report = build_report(
        evaluation=rows,
        baseline=baseline,
        episodes=episodes.get("completed_objective_windows"),
        partial=episodes.get("partial_budget_episodes", []),
        development=development,
        history=history,
    )
    report["historical_limitations"] = [
        "Old reservation/missed-pickup fields are retained as legacy raw observations, not valid V3 error rates.",
        "Missing decision scores, replay ages, raw advantages and cohort outcomes cannot be reconstructed from aggregate metrics.",
        "Optimizer coverage in original harness files does not prove actor decision learning.",
    ]
    return report, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--layout", choices=("native", "batch04"), default="native")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.run = native_path(args.run)
    if args.baseline is not None:
        args.baseline = native_path(args.baseline)
    if args.layout == "batch04":
        report, sources = batch04_report(args.run, args.baseline)
        report["sources"] = [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sources
        ]
        write_report(args.output, report)
        return
    rows, sources = read_results(args.run)
    baseline, other_sources = read_results(args.baseline) if args.baseline else ([], [])

    def load(relative, default):
        path = args.run / relative
        if not path.exists():
            return default
        sources.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    episode_data = load("reports/episodes.json", {})
    dev = []
    for path in sorted((args.run / "logs").glob("validation-*.json")):
        sources.append(path)
        dev.append(
            {
                "checkpoint": path.stem,
                "rows": json.loads(path.read_text(encoding="utf-8")),
            }
        )
    from smartsom.config.diagnostics import DiagnosticsOptions

    options = DiagnosticsOptions.model_validate(
        load("config/experiment.json", {}).get("diagnostics", {})
    )
    report = build_report(
        evaluation=rows,
        baseline=baseline,
        episodes=episode_data.get("whole"),
        partial=episode_data.get("partial", []),
        development=dev,
        history=load("reports/training.json", []),
        interval_updates=options.reports.interval_updates,
    )
    report["sources"] = [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sources + other_sources
    ]
    write_report(args.output, report)


if __name__ == "__main__":
    main()
