"""Audit 4096 grid adapter steps and ten paired resource-PPO/SPT evaluations."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import yaml
from validation.idetc_audit import audit_run
from validation.resource_acceptance import (
    RECIPE_VERSION,
    require_evaluation_recipe,
    require_evaluation_result,
    require_integrated_source,
    require_matching_source,
    require_training_recipe,
    require_training_result,
    run_identity,
)

import smartsom
from smartsom.config import load_resolved_run, resolve_study
from smartsom.config.codec import canonical_json, digest
from smartsom.config.experiment import ExperimentConfig, PreparedExperiment
from smartsom.config.production import (
    ProductionRecipe,
    checkpoint_algorithm_document,
    recipe_identity,
)
from smartsom.experiments import run_batch
from smartsom.experiments.catalog import training_locator
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.packaging import model_locator
from smartsom.experiments.production_training import verify_checkpoint
from smartsom.experiments.training_audit import audit_training
from smartsom.learning.checkpoint import file_hash
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow unintegrated source; never records formal acceptance.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    source = source_identity()
    report = {
        "status": "running",
        "source": source,
        "evaluation": [],
        "evidence_kind": "development" if args.development else "formal",
        "recipe_version": RECIPE_VERSION,
        "implementation_sha": source.get("git", {}).get("commit"),
        "accepted_source_sha": None,
        "platform": source.get("platform"),
    }
    try:
        if not Path(smartsom.__file__).resolve().is_relative_to(ROOT / "src"):
            raise ValueError("acceptance must import this checkout's smartsom source")
        commit = None if args.development else require_integrated_source(ROOT, source)
        directory = training_locator(args.training_dir)
        checkpoint = model_locator(args.training_dir)
        verify_checkpoint(checkpoint)
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        if not args.development:
            require_matching_source(metadata.get("source"), commit, "training")
        payload = json.loads((checkpoint / "recipe.json").read_text())
        config = ExperimentConfig.model_validate_json(canonical_json(payload["config"]))
        recipe = ProductionRecipe(**payload["recipe"])
        prepared = PreparedExperiment(
            canonical_json(config),
            "{}",
            recipe,
            digest(recipe_identity(recipe, config)),
        )
        require_training_recipe(prepared)
        report["training"] = audit_training(checkpoint)
        report["training"]["source"] = metadata.get("source")
        try:
            require_training_result(report["training"], directory)
            report["training_eligibility"] = {"status": "passed"}
        except ValueError as exc:
            report["training_eligibility"] = {"status": "failed", "error": str(exc)}
        write_json(output / "report.json", report)
        algorithm_path = output / "resource_checkpoint.json"
        write_json(
            algorithm_path,
            checkpoint_algorithm_document(
                metadata, checkpoint, file_hash(checkpoint / "checkpoint.json")
            ),
        )
        template = ROOT / "configs/studies/resource_evaluation.yaml"
        spec = yaml.safe_load(template.read_text())
        spec["cases"][0]["scenario"] = str(
            (template.parent / spec["cases"][0]["scenario"]).resolve()
        )
        for algorithm in spec["algorithms"]:
            algorithm["config"] = str(
                algorithm_path
                if algorithm["id"] == "Resource-PPO"
                else (template.parent / algorithm["config"]).resolve()
            )
        spec["output_root"] = str(output / "studies")
        path = output / "study.yaml"
        path.write_text(yaml.safe_dump(spec))
        resolved = resolve_study(path)
        require_evaluation_recipe(resolved)
        expected_runs = {e.entry_id: e for e in resolved.entries}
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
        if len(summary["runs"]) != len(expected_runs) or {
            r["entry_id"] for r in summary["runs"]
        } != set(expected_runs):
            raise ValueError("evaluation child coverage differs from frozen plan")
        for child in summary["runs"]:
            entry = expected_runs[child["entry_id"]]
            row = {
                "algorithm": entry.resolved.resolved.algorithm.provider,
                "replication": entry.replication,
                "world_sha256": digest(entry.resolved.resolved.scenario),
                "run_dir": child.get("run_dir"),
            }
            try:
                if not child.get("run_dir"):
                    raise ValueError("child run evidence unavailable")
                run_dir = Path(child["run_dir"])
                manifest = json.loads((run_dir / "run.json").read_text())
                if not args.development:
                    require_matching_source(
                        manifest.get("source"), commit, "evaluation"
                    )
                run = load_resolved_run(run_dir / "run.json")
                if digest(run_identity(run)) != digest(run_identity(entry.resolved)):
                    raise ValueError("evaluation run differs from frozen study inputs")
                row["run_status"] = manifest["status"]
                row["execution_replay"] = audit(run_dir)
                row.update(audit_run(run_dir, expected_jobs=4, expected_operations=9))
                if manifest.get("observations") not in ("hash", "full"):
                    raise ValueError(
                        "resource evaluation requires observation evidence"
                    )
                row["checks"].append("observation_replay")
                if row["algorithm"] == "rllib.resource_ppo":
                    learning = row.get("learning", {})
                    if learning.get("reason") != "completed" or not learning.get(
                        "decisions"
                    ):
                        raise ValueError(
                            "resource evaluation learning replay is incomplete"
                        )
                    row["learning_replay"] = {"status": "passed", **learning}
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["evaluation"].append(row)
            write_json(output / "report.json", report)
        report["aggregates"] = {}
        for provider in ("builtin.spt", "rllib.resource_ppo"):
            rows = [r for r in report["evaluation"] if r["algorithm"] == provider]
            successful = [r for r in rows if r["status"] == "passed"]
            values = [r["makespan"] for r in successful]
            report["aggregates"][provider] = {
                "successful": len(successful),
                "failed": len(rows) - len(successful),
                "makespan_mean": mean(values) if values else None,
                "makespan_sample_std": stdev(values) if len(values) > 1 else None,
            }
        require_training_result(report["training"], directory)
        require_evaluation_result(report)
        report["final_source"] = source_identity()
        if not args.development:
            require_matching_source(
                report["training"]["source"], commit, "training audit"
            )
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
        f"# Resource PPO acceptance\n\nStatus: **{report['status']}**\n\n"
        + f"Evidence: {report['evidence_kind']}; platform: {report['platform']}; implementation: {report['implementation_sha']}.\n\n"
        + f"Completed evaluations: {report.get('completed', 0)}/10; failed: {report.get('failed', 0)}.\n\n"
        + "4096 grid adapter decisions, seed 101; paired evaluation root 202, replications 0–4. "
        + "Adapter decisions may take zero physical time; they are not simulator ticks or historical joint rounds. "
        + "No performance superiority requirement; see report.json for all identities, role updates, individual results and unified physical/observation replay audits. Historical matrix acceptance is not grid acceptance.\n"
        + (f"\nError: {report['error']}\n" if "error" in report else "")
    )
    print(
        json.dumps({"status": report["status"], "report": str(output / "report.json")}),
        flush=True,
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
