"""Run serial paired development profiles; fail on outcome or trace differences."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def semantic(report):
    # Exclude only measured wall-clock fields, never scientific time/metrics.
    result = {key: value for key, value in report.items() if key != "wall_seconds"}
    if isinstance(result.get("research_metrics"), dict):
        result["research_metrics"] = {
            key: value
            for key, value in result["research_metrics"].items()
            if key != "observer_seconds"
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--case", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "phases", "trace", "cprofile"],
        default=["baseline", "phases", "trace"],
    )
    parser.add_argument("--checkpoint", default="checkpoints/episode-00001000.pt")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_identities = {}
    for index, run in enumerate(args.run):
        for case in args.case or [0]:
            for mode in args.modes:
                times = {"baseline": [], "candidate": []}
                files = []
                for rep in range(args.repetitions if mode == "baseline" else 1):
                    pair = {}
                    order = (
                        ("baseline", "candidate")
                        if rep % 2 == 0
                        else ("candidate", "baseline")
                    )
                    for side in order:
                        label = f"run{index}-case{case}-{mode}-{rep}-{side}"
                        path = args.output / (label + ".json")
                        if path.exists():
                            raise ValueError(f"Refusing to overwrite evidence: {path}")
                        command = [
                            sys.executable,
                            str(Path(__file__).with_name("profile_case.py")),
                            "--source",
                            getattr(args, side),
                            "--run",
                            run,
                            "--checkpoint",
                            args.checkpoint,
                            "--case",
                            str(case),
                            "--output",
                            str(args.output),
                            "--label",
                            label,
                            "--mode",
                            mode,
                            "--rebind-builtins",
                        ]
                        subprocess.run(command, check=True)
                        pair[side] = json.loads(path.read_text())
                        identity = tuple(
                            pair[side][k]
                            for k in ("source_commit", "implementation_sha256")
                        )
                        if source_identities.setdefault(side, identity) != identity:
                            raise RuntimeError(
                                f"{side} source changed between paired runs"
                            )
                        times[side].append(pair[side]["episode_wall_seconds"])
                        files.append(str(path.resolve()))
                    a, b = pair["baseline"], pair["candidate"]
                    identity_keys = (
                        "case_id",
                        "scenario_seed",
                        "action_seed",
                        "checkpoint_sha256",
                        "scenario_sha256",
                        "metrics_level",
                        "numerical_threads",
                    )
                    assert all(a[k] == b[k] for k in identity_keys), (
                        "comparison inputs differ"
                    )
                    assert semantic(a["report"]) == semantic(b["report"]), (
                        f"outcome mismatch: {files[-2:]}"
                    )
                    if mode == "trace":
                        assert a["trace"] == b["trace"], (
                            f"trajectory mismatch: {files[-2:]}"
                        )
                row = {
                    "run": run,
                    "case": case,
                    "mode": mode,
                    "files": files,
                    "exact_report_match": True,
                    "trace_match": True if mode == "trace" else None,
                    "times": times,
                    "timing_is_instrumented": mode != "baseline",
                }
                if mode == "baseline":
                    base, candidate = (
                        statistics.median(times[k]) for k in ("baseline", "candidate")
                    )
                    row.update(
                        baseline_median=base,
                        candidate_median=candidate,
                        speedup=base / candidate,
                        wall_reduction_percent=100 * (1 - candidate / base),
                        performance_check={
                            "wall_increase_percent": 100 * (candidate / base - 1),
                            "attention_threshold_percent": 5.0,
                            "needs_attention": candidate > base * 1.05,
                            "repetitions": args.repetitions,
                            "preliminary": args.repetitions < 3,
                        },
                    )
                rows.append(row)
                (args.output / "comparison.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                print(json.dumps({"comparison": row}), flush=True)
                if row.get("performance_check", {}).get("needs_attention"):
                    print(
                        json.dumps(
                            {
                                "performance_attention": "wall-time increase exceeds 5%",
                                "run": run,
                                "case": case,
                                **row["performance_check"],
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
