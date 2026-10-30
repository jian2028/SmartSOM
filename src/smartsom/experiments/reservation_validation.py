"""Episode milestone panels using the shared case-recoverable evaluator."""

import json
import shutil
from types import SimpleNamespace

from smartsom.experiments import reservation_runtime as runtime


def copy_checkpoint(source, destination):
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
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    target = destination.with_suffix(destination.suffix + ".json")
    if metadata is not None:
        runtime.write_json(target, metadata)
    elif target.exists():
        target.unlink()  # Never retain provenance for a replaced model.


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
    episode = session.validation_node
    if episode not in settings.milestones:
        return
    import torch

    prefix = "tick" if settings.unit == "physical_ticks" else "episode"
    directory = session.root / "validation" / f"{prefix}-{episode:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.pt"
    if not model.exists():
        copy_checkpoint(session.root / "last.pt", model)
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
                    "episodes": len(session.rows),
                    "physical_ticks": session.ticks,
                    "rollout_and_update_seconds": sum(
                        r["wall_seconds"] + r.get("learner_update_seconds", 0)
                        for r in session.rows
                    ),
                    "timing_scope": "active rollouts and learner updates; excludes validation, checkpoint IO and downtime",
                },
            )
            runtime.evaluate(args)
    if session.ex.evaluation.learning_target is not None:
        record_learning_target(session, episode)
    candidates = []
    for milestone in settings.milestones:
        panel = session.root / "validation" / f"{prefix}-{milestone:08d}"
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
        copy_checkpoint(selected[2] / "model.pt", session.root / "best.pt")
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


def record_learning_target(session, episode):
    """Actual validation receipts, never training losses, establish target progress."""
    from smartsom.telemetry.evaluation_summary import hierarchical
    from smartsom.telemetry.research_comparison import LearningTarget, time_to_target
    from smartsom.telemetry.research_metrics import fingerprint

    unit = session.ex.validation.unit
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
    for mode in session.ex.validation.modes:
        points = []
        for milestone in session.ex.validation.milestones:
            if milestone > episode:
                continue
            directory = session.root / "validation" / f"{prefix}-{milestone:08d}" / mode
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
            expected = session.ex.validation_cases * (
                session.ex.validation.sample_action_seeds if mode == "sample" else 1
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
                for m in session.ex.validation.milestones
                if m <= episode and (m != 0 or session.ex.validation.select_initial)
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
            "phase": "validation",
            "kind": "learning_target",
            "training_progress": {
                "episodes": len(session.rows),
                "physical_ticks": session.ticks,
                "train_seed": session.args.seed,
            },
            "modes": outcomes,
        }
    )
