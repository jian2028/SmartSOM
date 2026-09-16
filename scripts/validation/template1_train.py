"""Bounded local curriculum; no rule actions enter training or evaluation."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

from template1_audit import check
from template1_evidence import capture_source

from smartsom import api
from smartsom.config.production import AlgorithmConfig
from smartsom.experiments.production import execute
from smartsom.trace.production import atomic_json, audit

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=21600)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Evaluate an existing immutable selection without training",
    )
    args = parser.parse_args()
    if not 0 < args.seconds <= 21600:
        parser.error("budget must be between 1 and 21600 seconds")
    output = (
        ROOT
        / "artifacts/template1"
        / (
            ("evaluation-" if args.selection else "curriculum-")
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    capture_source(output / "source")
    if args.selection:
        return evaluate_selected(args.selection, output)
    start = time.monotonic()
    deadline = start + args.seconds
    report = {
        "status": "running",
        "stages": [],
        "budget_seconds": args.seconds,
        "selection": "best stage endpoint on development seed 401, ranked by delivered count then completed operations",
        "execution": "stochastic, no rule takeover",
        "final_seeds": [501, 502, 503, 504, 505],
    }
    previous = None
    for index, name in enumerate(("warmup", "static", "disturbed")):
        if time.monotonic() >= deadline:
            break
        config = api.load_config(ROOT / f"configs/runs/template1_train_{name}.yaml")
        config.validation.enabled = False
        config.checkpointing.every_updates = 16
        config.checkpointing.keep_last = 2
        config.checkpointing.save_best = False
        config.logging.progress = "off"
        stage_deadline = min(
            deadline, time.monotonic() + (deadline - time.monotonic()) / (3 - index)
        )

        def progress(event):
            if event.get("stage") == "learning_metrics":
                atomic_json(
                    output / "progress.json",
                    {
                        **event,
                        "curriculum_stage": name,
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
                if time.monotonic() >= stage_deadline:
                    return {"stop": "interrupted"}

        result = api.train(config, initialize_from=previous, on_progress=progress)
        previous = result.last_checkpoint
        report["stages"].append(
            {
                "name": name,
                "run_dir": str(result.run_dir),
                "checkpoint": str(previous),
                "steps": result.environment_steps,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        from smartsom.learning.production import LearnedProductionDriver

        development = []
        for case in ("static", "disturbed"):
            scenario = api.prepare(
                api.load_config(ROOT / f"configs/runs/template1_{case}.yaml"),
                training=False,
            ).resolved.scenario
            driver = LearnedProductionDriver(
                previous, scenario, deterministic=False, seed=401
            )
            try:
                while not driver.env.finished:
                    driver.next_tick()
                snap = driver.sim.snapshot()
                development.append(
                    {
                        "case": case,
                        "qualified": len(snap["completed"]),
                        "operations": sum(j["step"] for j in snap["jobs"].values()),
                    }
                )
            finally:
                driver.env.close()
        report["stages"][-1]["development"] = development
        atomic_json(output / "report.json", report)
        print(json.dumps(report["stages"][-1]), flush=True)
    if previous is None:
        raise RuntimeError("budget ended without a checkpoint")
    # Freeze selection before touching held-out execution seeds.
    selected = max(
        report["stages"],
        key=lambda stage: (
            sum(v["qualified"] for v in stage["development"]),
            sum(v["operations"] for v in stage["development"]),
        ),
    )
    previous = Path(selected["checkpoint"])
    report["selected_checkpoint"] = str(previous)
    atomic_json(output / "selection.json", report)
    return evaluate_selected(output / "selection.json", output)


def evaluate_selected(selection, output):
    """Evaluate an already frozen selection; never choose weights from test scores."""
    start = time.monotonic()
    report = json.loads(Path(selection).read_text())
    report["selection_file"] = str(Path(selection).resolve())
    previous = Path(report["selected_checkpoint"])
    report["evaluations"] = []
    jobs = [
        (name, seed, str(previous), str(output))
        for name in ("static", "disturbed")
        for seed in report["final_seeds"]
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as workers:
        for row in workers.map(evaluate_case, jobs):
            report["evaluations"].append(row)
            atomic_json(output / "report.json", report)
            print(
                json.dumps({k: v for k, v in row.items() if k != "replay"}), flush=True
            )
    report["passed"] = all(e["ledger"]["passed"] for e in report["evaluations"])
    report["status"] = "passed" if report["passed"] else "model_not_accepted"
    report["evaluation_elapsed_seconds"] = time.monotonic() - start
    atomic_json(output / "report.json", report)
    return 0 if report["passed"] else 1


def evaluate_case(job):
    import torch

    torch.set_num_threads(1)
    name, seed, checkpoint, output = job
    previous = Path(checkpoint)
    metadata = json.loads((previous / "checkpoint.json").read_text())
    algorithm = AlgorithmConfig.model_validate_json(json.dumps(metadata["algorithm"]))
    algorithm = algorithm.model_copy(update={"checkpoint": str(previous)})
    scenario = api.prepare(
        api.load_config(ROOT / f"configs/runs/template1_{name}.yaml"), training=False
    ).resolved.scenario
    directory = execute(
        scenario,
        algorithm,
        output_root=Path(output) / "evaluation",
        name=f"{name}-{seed}",
        verbose=False,
        deterministic=False,
        policy_seed=seed,
    )
    record = json.loads((directory / "run.json").read_text())
    try:
        ledger = check(directory)
    except AssertionError as exc:
        ledger = {"passed": False, "reason": str(exc)}
    return {
        "case": name,
        "seed": seed,
        "run_dir": str(directory),
        "ledger": ledger,
        "qualified": len(record["result"]["completed"]),
        "replay": audit(directory),
    }


if __name__ == "__main__":
    raise SystemExit(main())
