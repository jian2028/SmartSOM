"""Episode milestone panels using the shared case-recoverable evaluator."""

import json
import os
import shutil
from types import SimpleNamespace

from smartsom.experiments import reservation_runtime as runtime


def copy_model(source, destination, *, link=False):
    """Atomically publish immutable weights, sharing storage when supported."""
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    if link:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
    else:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def copy_checkpoint(source, destination, *, link=False):
    """Copy provenance only when it proves the bytes being copied."""
    import hashlib

    evidence = source.with_suffix(source.suffix + ".json")
    metadata = None
    if evidence.exists():
        metadata = json.loads(evidence.read_text())
        if (
            metadata.get("schema") != "smartsom.checkpoint-evidence/v1"
            or metadata.get("checkpoint_sha256")
            != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            raise ValueError("checkpoint sidecar does not identify its model")
    copy_model(source, destination, link=link)
    target = destination.with_suffix(destination.suffix + ".json")
    if metadata is not None:
        runtime.write_json(target, metadata)
    elif target.exists():
        target.unlink()  # Never retain provenance for a replaced model.
    if link:
        # Persist aliases before any owned source generation can be collected.
        from smartsom.experiments.checkpoint_retention import _sync_directory

        for path in (destination, target) if metadata is not None else (destination,):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(destination.parent)


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


def evaluate_periodically(session):
    settings = session.ex.periodic_evaluation
    episode = session.periodic_evaluation_node
    if episode not in settings.milestones:
        return
    import torch

    prefix = "tick" if settings.unit == "physical_ticks" else "episode"
    directory = session.root / "periodic_evaluation" / f"{prefix}-{episode:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.pt"
    bounded = session.ex.checkpoint_retention == "best_last"
    complete = all(
        (directory / mode / "summary.json").is_file() for mode in settings.modes
    )
    if not model.exists() and not (bounded and complete):
        copy_checkpoint(session.root / "last.pt", model, link=bounded)
    # Evaluator initialization/loading may touch Torch's global generator; it
    # must not perturb the learner. Its policies use separate action generators.
    with torch.random.fork_rng(devices=[]):
        for mode in settings.modes:
            if bounded and complete:
                continue
            args = SimpleNamespace(
                experiment=session.args.experiment,
                tracking=session.tracking,
                output=directory / mode,
                native=True,
                calibration=session.args.calibration,
                checkpoint=str(model),
                method=["configured"],
                selection=mode,
                split="periodic_evaluation",
                cases=session.ex.periodic_evaluation_cases,
                action_seeds=settings.sample_action_seeds if mode == "sample" else 1,
                seconds=session.args.seconds,
                deadline_unix=None
                if session.deadline == float("inf")
                else session.deadline,
                numerical_threads=torch.get_num_threads(),
                training_progress={
                    "train_seed": session.args.seed,
                    "episodes": len(session.rows),
                    "physical_ticks": session.ticks,
                    "rollout_and_update_seconds": sum(
                        r["wall_seconds"] + r.get("learner_update_seconds", 0)
                        for r in session.rows
                    ),
                    "timing_scope": "active rollouts and learner updates; excludes periodic_evaluation, checkpoint IO and downtime",
                },
            )
            runtime.evaluate(args)
    if session.ex.evaluation.learning_target is not None:
        record_learning_target(session, episode)
    candidates = []
    for milestone in settings.milestones:
        panel = session.root / "periodic_evaluation" / f"{prefix}-{milestone:08d}"
        if milestone > episode or (milestone == 0 and not settings.select_initial):
            continue
        # Every configured panel must complete before this model is eligible.
        if not all(
            (panel / mode / "summary.json").is_file() for mode in settings.modes
        ):
            continue
        decoder = "argmax" if settings.selection == "argmax_complete_cost" else "sample"
        sample = panel / decoder / "cases.jsonl"
        if not sample.exists():
            continue
        rows = [json.loads(line) for line in sample.read_text().splitlines()]
        candidates.append(
            (candidate_key(rows, milestone, settings.selection), milestone, panel)
        )
    selected = min(candidates) if candidates and settings.selection != "last" else None
    if selected:
        source = selected[2] / "model.pt"
        best = session.root / "best.pt"
        if source.exists():
            copy_checkpoint(source, best, link=bounded)
        else:
            import hashlib

            expected = json.loads(source.with_suffix(".pt.json").read_text())
            if (
                not best.is_file()
                or hashlib.sha256(best.read_bytes()).hexdigest()
                != expected["checkpoint_sha256"]
                or json.loads(best.with_suffix(".pt.json").read_text()) != expected
            ):
                raise ValueError(
                    "retained best checkpoint does not match periodic_evaluation selection"
                )
    runtime.write_json(
        session.root / "selection.json",
        {
            "rule": settings.selection,
            "eligibility": "all_configured_panels_complete",
            "unit": settings.unit,
            "selected_node": selected[1] if selected else None,
            "selected_episode": selected[1]
            if selected and settings.unit == "episodes"
            else None,
            "status": "selected"
            if selected
            else "last"
            if settings.selection == "last"
            else "unselected",
            "candidates": [{"episode": item[1], "key": item[0]} for item in candidates],
        },
    )
    if bounded:
        # The score/provenance remain; best and last hold the only durable models.
        for milestone in settings.milestones:
            if milestone > episode:
                continue
            panel = session.root / "periodic_evaluation" / f"{prefix}-{milestone:08d}"
            candidate = panel / "model.pt"
            if candidate.is_symlink() or not candidate.is_file():
                continue
            evidence = candidate.with_suffix(".pt.json")
            if not evidence.is_file():
                continue
            metadata = json.loads(evidence.read_text())
            if metadata.get("run_id") == session.run_id and all(
                (panel / mode / "summary.json").is_file() for mode in settings.modes
            ):
                candidate.unlink()


def record_learning_target(session, episode):
    """Actual periodic_evaluation receipts, never training losses, establish target progress."""
    from smartsom.telemetry.evaluation_summary import hierarchical
    from smartsom.telemetry.research_comparison import LearningTarget, time_to_target
    from smartsom.telemetry.research_metrics import fingerprint

    unit = session.ex.periodic_evaluation.unit
    prefix = "tick" if unit == "physical_ticks" else "episode"
    config = session.ex.evaluation.learning_target
    if config.rule_cost is None or not config.reference_sha256:
        raise ValueError("learning target requires a compiled frozen Rule reference")
    target = LearningTarget(
        "smartsom.learning-target/v2",
        config.reference_sha256,
        config.rule_cost * config.cost_ratio,
        0,
        config.consecutive,
        unit,
    )
    outcomes = {}
    for mode in session.ex.periodic_evaluation.modes:
        points = []
        for milestone in session.ex.periodic_evaluation.milestones:
            if milestone > episode:
                continue
            directory = (
                session.root
                / "periodic_evaluation"
                / f"{prefix}-{milestone:08d}"
                / mode
            )
            file = directory / "cases.jsonl"
            if not file.exists():
                continue
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            if not rows:
                continue
            values = hierarchical(rows)["metrics"]
            identity = rows[0]["evaluation_identity"]
            panel = fingerprint(
                {
                    "cases": sorted(
                        (
                            r["scenario_seed"],
                            r["condition"],
                            r["action_seed"],
                            r["evaluation_identity"]["world_sha256"],
                        )
                        for r in rows
                    ),
                    "evaluation": session.ex.evaluation.model_dump(
                        mode="json", exclude={"learning_target"}
                    ),
                }
            )
            expected = session.ex.periodic_evaluation_cases * (
                session.ex.periodic_evaluation.sample_action_seeds
                if mode == "sample"
                else 1
            )
            complete = len(rows) == expected and values["drained"] == 1
            points.append(
                {
                    "budget": milestone,
                    "train_seed": session.args.seed,
                    "decoder": mode,
                    "panel_sha256": panel,
                    "model_sha256": identity["checkpoint_sha256"],
                    "cost": values["joint_cost"] if complete else None,
                    "passing_rate": values["final_passing_rate"] if complete else None,
                    "all_drained": complete,
                }
            )
        if points:
            by_budget = {p["budget"]: p for p in points}
            exemplar = points[0]
            points = [
                by_budget.get(
                    m,
                    {
                        "budget": m,
                        "train_seed": session.args.seed,
                        "decoder": mode,
                        "panel_sha256": exemplar["panel_sha256"],
                        "model_sha256": f"missing-{m}",
                        "cost": None,
                        "passing_rate": None,
                        "all_drained": None,
                    },
                )
                for m in session.ex.periodic_evaluation.milestones
                if m <= episode
                and (m != 0 or session.ex.periodic_evaluation.select_initial)
            ]
        outcomes[mode] = time_to_target(points, target)
    runtime.write_json(
        session.root / "learning-target.json",
        {
            "schema": "smartsom.learning-target/v2",
            "target": config.model_dump(mode="json"),
            "modes": outcomes,
        },
    )
    session.tracking.log(
        {
            "phase": "periodic_evaluation",
            "kind": "learning_target",
            "training_progress": {
                "episodes": len(session.rows),
                "physical_ticks": session.ticks,
                "train_seed": session.args.seed,
            },
            "modes": outcomes,
        }
    )
