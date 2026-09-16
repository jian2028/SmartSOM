"""Grid fixed-budget learning audit and fifteen paired evaluations.

Historical matrix acceptance remains attached to its original source. This gate
checks the migrated grid recipes and never calls a truncated episode complete.
"""

import argparse
import json
from pathlib import Path

import yaml
from validation.idetc_audit import audit_run
from validation.resource_acceptance import (
    require_integrated_source,
    require_matching_source,
    run_identity,
)
from validation.usability_acceptance import (
    require_central_coverage,
    require_central_recipe,
    require_frozen_training,
)

from smartsom.config import load_resolved_run, resolve_study
from smartsom.config.codec import canonical_json, digest
from smartsom.config.experiment import ExperimentConfig, PreparedExperiment
from smartsom.config.production import (
    ProductionRecipe,
    checkpoint_algorithm_document,
    recipe_identity,
)
from smartsom.experiments import run_batch
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.packaging import model_locator
from smartsom.experiments.production_training import verify_checkpoint
from smartsom.experiments.training_audit import audit_training
from smartsom.learning.checkpoint import file_hash
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[1]


def validate_fixed_training(name, directory):
    checkpoint = model_locator(directory)
    verify_checkpoint(checkpoint)
    payload = json.loads((checkpoint / "recipe.json").read_text())
    config = ExperimentConfig.model_validate_json(canonical_json(payload["config"]))
    recipe = ProductionRecipe(**payload["recipe"])
    actual = PreparedExperiment(
        canonical_json(config), "{}", recipe, digest(recipe_identity(recipe, config))
    )
    try:
        require_frozen_training(name, actual)
    except ValueError as exc:
        raise ValueError(
            f"{name} training differs from the fixed acceptance recipe"
        ) from exc
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rllib-training-dir", type=Path, required=True)
    parser.add_argument("--sb3-training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow engineering checks on unintegrated source; never formal acceptance.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    source = source_identity()
    report = {
        "status": "running",
        "source": source,
        "training": {},
        "evaluation": [],
        "recipe_version": "smartsom.grid-central-acceptance/v1",
        "evidence_kind": "development" if args.development else "formal",
        "accepted_source_sha": None,
    }
    try:
        commit = None if args.development else require_integrated_source(ROOT, source)
        references = {}
        for name, directory, label in (
            ("rllib", args.rllib_training_dir, "RLlib-PPO"),
            ("sb3", args.sb3_training_dir, "SB3-MaskablePPO"),
        ):
            checkpoint = validate_fixed_training(name, directory)
            metadata = json.loads((checkpoint / "checkpoint.json").read_text())
            if not args.development:
                require_matching_source(metadata.get("source"), commit, "training")
            training = audit_training(checkpoint)
            if (
                not training["budget_completed"]
                or training["learner_updates"] <= 0
                or metadata["initial_weights_sha256"]
                == metadata["final_weights_sha256"]
            ):
                raise ValueError(
                    "fixed learning budget or actual parameter updates are missing"
                )
            report["training"][name] = training
            path = output / f"{label}.json"
            write_json(
                path,
                checkpoint_algorithm_document(
                    metadata, checkpoint, file_hash(checkpoint / "checkpoint.json")
                ),
            )
            references[label] = path
            write_json(output / "report.json", report)
        template = ROOT / "configs/studies/learning_evaluation.yaml"
        spec = yaml.safe_load(template.read_text())
        spec["cases"][0]["scenario"] = str(
            (template.parent / spec["cases"][0]["scenario"]).resolve()
        )
        for algorithm in spec["algorithms"]:
            algorithm["config"] = str(
                references.get(
                    algorithm["id"], template.parent / algorithm["config"]
                ).resolve()
            )
        spec["output_root"] = str(output / "studies")
        path = output / "study.yaml"
        path.write_text(yaml.safe_dump(spec))
        resolved = resolve_study(path)
        require_central_recipe([entry.resolved for entry in resolved.entries])
        expected = {entry.entry_id: entry for entry in resolved.entries}
        batch = run_batch(
            resolved,
            workers=args.workers,
            on_progress=lambda row: print(json.dumps(row), flush=True),
        )
        report.update(
            study_dir=str(batch.study_dir),
            completed=batch.completed,
            failed=batch.failed,
            pending=batch.pending,
        )
        summary = json.loads((batch.study_dir / "summary.json").read_text())
        if len(summary["runs"]) != 15 or {
            row["entry_id"] for row in summary["runs"]
        } != set(expected):
            raise ValueError("evaluation child coverage differs from frozen plan")
        for child in summary["runs"]:
            entry = expected[child["entry_id"]]
            row = {
                "provider": entry.resolved.resolved.algorithm.provider,
                "replication": entry.replication,
                "world_sha256": digest(entry.resolved.resolved.scenario),
                "run_dir": child.get("run_dir"),
                "audit_status": "failed",
                "makespan": None,
            }
            try:
                if not child.get("run_dir"):
                    raise ValueError("child run evidence unavailable")
                run_dir = Path(child["run_dir"])
                record = json.loads((run_dir / "run.json").read_text())
                if not args.development:
                    require_matching_source(record.get("source"), commit, "evaluation")
                actual = load_resolved_run(run_dir / "run.json")
                if digest(run_identity(actual)) != digest(run_identity(entry.resolved)):
                    raise ValueError("evaluation differs from frozen study inputs")
                row["run_status"] = record["status"]
                row["execution_replay"] = audit(run_dir)
                audited = audit_run(run_dir, expected_jobs=4, expected_operations=9)
                if record.get("observations") not in ("hash", "full"):
                    raise ValueError("evaluation observation evidence is missing")
                if (
                    row["provider"] != "builtin.spt"
                    and audited.get("learning", {}).get("reason") != "completed"
                ):
                    raise ValueError("learning decision replay did not complete")
                row.update(audited, audit_status="passed")
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["evaluation"].append(row)
            write_json(output / "report.json", report)
        require_central_coverage(
            {"status": "passed", "completed": batch.completed, "failed": batch.failed},
            report["evaluation"],
        )
        report["final_source"] = source_identity()
        if not args.development:
            require_matching_source(
                report["final_source"], commit, "acceptance completion"
            )
            require_integrated_source(ROOT, report["final_source"])
            report["accepted_source_sha"] = commit
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(output / "report.json", report)
    (output / "report.md").write_text(
        f"# Grid learning acceptance\n\nStatus: **{report['status']}**\n\n"
        f"Evidence: {report['evidence_kind']}. Completed evaluations: {report.get('completed', 0)}/15; "
        f"failed: {report.get('failed', 0)}.\n\n"
        "Fixed seeds and budgets. All qualified demands and physical/observation replays must pass. "
        "Historical matrix results do not establish grid acceptance; no performance superiority claim.\n"
        + (f"\nError: {report['error']}\n" if "error" in report else "")
    )
    print(
        json.dumps({"status": report["status"], "report": str(output / "report.json")}),
        flush=True,
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
