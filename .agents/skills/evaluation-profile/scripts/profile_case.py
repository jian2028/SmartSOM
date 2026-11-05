"""Development-only fixed-case timing and equivalence evidence; no training."""

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import pstats
import subprocess
import sys
import time
from pathlib import Path


def main():
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument(
        "--rebind-builtins",
        action="store_true",
        help="Explicit development comparison: validate archived checkpoint, then bind builtins to --source in memory",
    )
    p.add_argument(
        "--source",
        required=True,
        help="Checkout root to execute (not the archived source)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Local evidence directory, outside source control",
    )
    p.add_argument("--label", required=True)
    p.add_argument("--checkpoint", default="checkpoints/episode-00001000.pt")
    p.add_argument("--case", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=["baseline", "cprofile", "phases", "trace"],
        default="baseline",
    )
    p.add_argument("--metrics", default=None)
    a = p.parse_args()
    root = Path(a.run).resolve()
    out = Path(a.output).resolve()
    if Path(a.label).name != a.label:
        p.error("label must be a file basename")
    if out == root or out.is_relative_to(root):
        p.error("output must be outside the archived trial")
    if (out / (a.label + ".json")).exists():
        p.error("refusing to overwrite existing evidence")
    out.mkdir(parents=True, exist_ok=True)
    source = Path(a.source).resolve()
    if not (source / "src/smartsom/__init__.py").is_file():
        p.error("source must be a SmartSOM checkout root")
    manifest = json.loads((root / "manifest.json").read_text())
    package = root / "source/smartsom"
    actual = hashlib.sha256(
        b"".join(
            str(x.relative_to(package)).encode() + b"\0" + x.read_bytes()
            for x in sorted(package.rglob("*.py"))
        )
    ).hexdigest()
    assert actual == manifest["source"]["implementation_sha256"], (
        actual,
        manifest["source"]["implementation_sha256"],
    )
    sys.path.insert(0, str(source / "src"))
    source_hash = hashlib.sha256(
        b"".join(
            str(x.relative_to(source / "src/smartsom")).encode()
            + b"\0"
            + x.read_bytes()
            for x in sorted((source / "src/smartsom").rglob("*.py"))
        )
    ).hexdigest()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    )
    import smartsom
    from smartsom.algorithms.base_policy import build_policy, evaluation_recipe
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.evaluation_cases import load_cases
    from smartsom.config.scenario_snapshot import scenario_from_snapshot
    from smartsom.experiments.reservation_runtime import episode

    print(
        json.dumps({"stage": "loading", "label": a.label, "module": smartsom.__file__}),
        flush=True,
    )
    assert Path(smartsom.__file__).resolve().parent == source / "src/smartsom"
    setup_start = time.perf_counter()
    ex, f, w, options, paths = load_inputs(root / "inputs/experiment.yaml")
    cases = load_cases(
        root / "inputs" / ex.case_sets["test"],
        split="test",
        factory=f,
        workload=w,
        condition_inputs=ex._condition_inputs,
    )
    case = cases[a.case]
    scenario = scenario_from_snapshot(case["scenario"])
    options = evaluation_recipe(options, "configured")
    checkpoint = root / a.checkpoint
    from smartsom.learning.state import load as load_checkpoint

    checkpoint_state = load_checkpoint(checkpoint, options)
    rebound = []
    if a.rebind_builtins:
        from smartsom.config.components import builtin_sources, digest_files

        document = options.model_dump(mode="json", by_alias=True)
        current_digest = digest_files(builtin_sources())
        for key, spec in options.component_items():
            if spec.target:
                rebound.append(
                    {
                        "component": key,
                        "archived_sha256": spec.sha256,
                        "executed_sha256": current_digest,
                    }
                )
                node = document
                for part in key.split("."):
                    node = node[part]
                node["sha256"] = current_digest
        rebound_options = type(options).model_validate(document)
        for key, value in options.__pydantic_private__.items():
            setattr(rebound_options, key, value)
        options = rebound_options
    policy = build_policy(
        options,
        case["action_seeds"][0],
        training=False,
        selection="argmax",
        numerical_threads=1,
    )
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy.set_state(checkpoint_state)
    policy.seed_sampling(case["action_seeds"][0])
    assert not policy.training
    setup_seconds = time.perf_counter() - setup_start
    originals = [
        json.loads(s)
        for s in (root / "evaluation" / checkpoint.stem / "argmax/cases.jsonl")
        .read_text()
        .splitlines()
    ]
    reference = next(
        r
        for r in originals
        if r["condition"] == case["condition"]
        and r["scenario_seed"] == case["seed"]
        and r["action_seed"] == case["action_seeds"][0]
    )
    phase_clock = None
    if a.mode == "phases":
        from phase_clock import PhaseClock

        phase_clock = PhaseClock()
        phase_clock.install(policy)
    trace = None
    if a.mode == "trace":
        from trace_clock import TraceDigest

        trace = TraceDigest()
        trace.install(policy)
    prof = cProfile.Profile() if a.mode == "cprofile" else None
    print(
        json.dumps(
            {
                "stage": "running",
                "label": a.label,
                "mode": a.mode,
                "condition": case["condition"],
                "original_seconds": reference["wall_seconds"],
                "setup_seconds": setup_seconds,
            }
        ),
        flush=True,
    )
    wall = time.perf_counter()
    cpu = time.process_time()
    if prof:
        prof.enable()
    try:
        report, rewards, critic = episode(
            *ex.inputs_for(case["condition"], f, w),
            options,
            policy,
            case["seed"],
            case["condition"],
            ex.drain_ticks,
            scenario=scenario,
            diagnostics=ex.run.diagnostics,
            metrics_level=a.metrics or ex.run.metrics,
            metric_interval=ex.run.metric_interval,
            partial_on_timeout=True,
            common_window_end=ex.evaluation.common_window_end_tick
            or scenario.tick_limit,
        )
    finally:
        if prof:
            prof.disable()
            prof.dump_stats(str(out / (a.label + ".pstats")))
    wall = time.perf_counter() - wall
    cpu = time.process_time() - cpu
    final_hash = hashlib.sha256(
        b"".join(
            str(x.relative_to(source / "src/smartsom")).encode()
            + b"\0"
            + x.read_bytes()
            for x in sorted((source / "src/smartsom").rglob("*.py"))
        )
    ).hexdigest()
    if final_hash != source_hash:
        raise RuntimeError("source changed during measurement; discard this timing")
    keys = [
        "condition",
        "scenario_seed",
        "status",
        "termination_reason",
        "tick",
        "completed",
        "total_orders",
        "mean_tardiness",
        "passing_rate",
        "joint_cost",
        "observed_overdue",
        "failed",
        "decisions",
        "protocol",
        "physics",
        "rule_actions",
    ]
    mismatches = {
        k: {"reference": reference.get(k), "actual": report.get(k)}
        for k in keys
        if reference.get(k) != report.get(k)
    }
    result = {
        "label": a.label,
        "mode": a.mode,
        "source_commit": source_commit,
        "source_status": source_status,
        "implementation_sha256": source_hash,
        "archived_commit": manifest["source"]["commit"],
        "archived_implementation_sha256": actual,
        "checkpoint_sha256": checkpoint_digest,
        "source_root": str(source / "src"),
        "case_id": case["id"],
        "scenario_sha256": hashlib.sha256(
            json.dumps(case["scenario"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_index": a.case,
        "scenario_seed": case["seed"],
        "action_seed": case["action_seeds"][0],
        "metrics_level": a.metrics or ex.run.metrics,
        "numerical_threads": 1,
        "training": False,
        "development_source_rebinding": rebound,
        "tracking_started": False,
        "progress_callback": False,
        "setup_seconds": setup_seconds,
        "episode_wall_seconds": wall,
        "episode_cpu_seconds": cpu,
        "original_wall_seconds": reference["wall_seconds"],
        "semantic_mismatches": mismatches,
        "report": report,
        "environment": {
            "python": sys.version,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ["torch", "numpy", "pydantic", "PyYAML"]
            },
        },
    }
    if trace:
        result["trace"] = trace.result()
    if phase_clock:
        result["phases"] = dict(phase_clock.rows)
        result["phase_distribution"] = phase_clock.distribution(wall)
    (out / (a.label + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if prof:
        stats = pstats.Stats(prof)
        rows = [
            {
                "file": key[0],
                "line": key[1],
                "function": key[2],
                "primitive_calls": v[0],
                "calls": v[1],
                "self_seconds": v[2],
                "cumulative_seconds": v[3],
            }
            for key, v in stats.stats.items()
        ]
        rows.sort(key=lambda r: r["self_seconds"], reverse=True)
        (out / (a.label + ".functions.json")).write_text(
            json.dumps(
                {
                    "total_seconds": stats.total_tt,
                    "total_calls": stats.total_calls,
                    "functions": rows,
                },
                indent=2,
            )
            + "\n"
        )
        with (out / (a.label + ".top.txt")).open("w") as stream:
            s = pstats.Stats(prof, stream=stream)
            s.sort_stats("cumulative").print_stats(45)
            s.sort_stats("tottime").print_stats(45)
    print(
        json.dumps(
            {
                "stage": "done",
                "label": a.label,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "reported_wall_seconds": report["wall_seconds"],
                "semantic_mismatches": list(mismatches),
                "tick": report["tick"],
                "completed": report["completed"],
                "decisions": report["decisions"],
            }
        ),
        flush=True,
    )
    # Historical results are context only: compare baseline and candidate at the same main.
    # Upstream changes may intentionally differ from the archived run.


if __name__ == "__main__":
    main()
