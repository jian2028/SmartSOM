"""Episode milestone panels using the shared case-recoverable evaluator."""

import json
import shutil
from types import SimpleNamespace

from smartsom.experiments import reservation_runtime as runtime


def candidate_key(rows, episode, rule):
    """Incomplete panels retain censored costs; no complete-case-only average."""
    incomplete = sum(row["joint_cost"] is None for row in rows)
    cost = sum(row["joint_cost"] for row in rows) / len(rows) if not incomplete else 0
    if rule == "sample_delivery_cost":
        delivered = sum(row["completed"] / row["total_orders"] for row in rows) / len(
            rows
        )
        return incomplete, -delivered, cost, episode
    return incomplete, cost, episode


def validate(session):
    settings = session.ex.validation
    episode = len(session.rows)
    if episode not in settings.milestones:
        return
    import torch

    directory = session.root / "validation" / f"episode-{episode:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.pt"
    if not model.exists():
        temporary = model.with_suffix(".tmp")
        shutil.copyfile(session.root / "last.pt", temporary)
        temporary.replace(model)
    # Evaluator initialization/loading may touch Torch's global generator; it
    # must not perturb the learner. Its policies use separate action generators.
    with torch.random.fork_rng(devices=[]):
        for mode in settings.modes:
            args = SimpleNamespace(
                experiment=session.args.experiment,
                tracking=session.tracking,
                output=directory / mode,
                native=True,
                calibration=session.args.calibration,
                checkpoint=str(model),
                method=["configured"],
                selection=mode,
                split="validation",
                cases=session.ex.validation_cases,
                action_seeds=settings.sample_action_seeds if mode == "sample" else 1,
                seconds=session.args.seconds,
                deadline_unix=session.deadline,
                numerical_threads=torch.get_num_threads(),
                training_progress={
                    "train_seed": session.args.seed,
                    "episodes": episode,
                    "physical_ticks": session.ticks,
                    "rollout_and_update_seconds": sum(
                        r["wall_seconds"] + r.get("learner_update_seconds", 0)
                        for r in session.rows
                    ),
                    "timing_scope": "active rollouts and learner updates; excludes validation, checkpoint IO and downtime",
                },
            )
            runtime.evaluate(args)
    candidates = []
    for milestone in settings.milestones:
        panel = session.root / "validation" / f"episode-{milestone:08d}"
        if milestone > episode or (milestone == 0 and not settings.select_initial):
            continue
        # Every configured panel must complete before this model is eligible.
        if not all(
            (panel / mode / "summary.json").is_file() for mode in settings.modes
        ):
            continue
        sample = panel / "sample/cases.jsonl"
        if not sample.exists():
            continue
        rows = [json.loads(line) for line in sample.read_text().splitlines()]
        candidates.append(
            (candidate_key(rows, milestone, settings.selection), milestone, panel)
        )
    selected = min(candidates) if candidates and settings.selection != "last" else None
    if selected:
        temporary = session.root / "best.pt.tmp"
        shutil.copyfile(selected[2] / "model.pt", temporary)
        temporary.replace(session.root / "best.pt")
    runtime.write_json(
        session.root / "selection.json",
        {
            "rule": settings.selection,
            "eligibility": "all_configured_panels_complete",
            "selected_episode": selected[1] if selected else None,
            "status": "selected"
            if selected
            else "last"
            if settings.selection == "last"
            else "unselected",
            "candidates": [{"episode": item[1], "key": item[0]} for item in candidates],
        },
    )
