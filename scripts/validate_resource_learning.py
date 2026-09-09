"""Fixed 4096-joint-step training and ten paired resource-PPO/SPT evaluations."""

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
from smartsom.config.codec import digest, read_model
from smartsom.config.models import AlgorithmFile, ResourceLearningAlgorithm
from smartsom.config.training import episode_input
from smartsom.experiments import run_batch
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.resource_audit import audit_joint_run
from smartsom.experiments.training_audit import audit_training, load_training_snapshot
from smartsom.learning.checkpoint import ResourceCheckpointManifest, file_hash

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
        directory = args.training_dir.resolve()
        if not args.development:
            training_manifest = json.loads((directory / "manifest.json").read_text())
            require_matching_source(training_manifest.get("source"), commit, "training")
        require_training_recipe(
            load_training_snapshot(directory / "resolved_training.json")
        )
        report["training"] = audit_training(directory)
        require_training_result(report["training"], directory)
        write_json(output / "report.json", report)
        checkpoint = directory / "checkpoint"
        metadata = read_model(
            checkpoint / "checkpoint.json", ResourceCheckpointManifest
        )[0]
        algorithm_path = output / "resource_checkpoint.json"
        write_json(
            algorithm_path,
            AlgorithmFile(
                schema="smartsom.algorithm/v1",
                algorithm=ResourceLearningAlgorithm(
                    provider=metadata.provider,
                    projection=metadata.projection,
                    parameters=metadata.parameters,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=file_hash(checkpoint / "checkpoint.json"),
                ),
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
        expected_runs = {digest(run_identity(e.resolved)) for e in resolved.entries}
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
        for manifest_path in sorted(batch.study_dir.rglob("manifest.json")):
            run_dir = manifest_path.parent
            if not (run_dir / "resolved_run.yaml").exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if not args.development:
                require_matching_source(manifest.get("source"), commit, "evaluation")
            run = load_resolved_run(run_dir / "resolved_run.yaml")
            if digest(run_identity(run)) not in expected_runs:
                raise ValueError("evaluation run differs from frozen study inputs")
            row = {
                "algorithm": run.algorithm.algorithm.provider,
                "replication": run.study_seed_origin.replication,
                "run_dir": str(run_dir),
                "world_sha256": digest(episode_input(run)),
            }
            if manifest["status"] != "completed":
                row.update(
                    status="failed",
                    failure=json.loads((run_dir / "failure.json").read_text()),
                )
            else:
                row.update(audit_run(run_dir, expected_jobs=4, expected_operations=9))
                if row["algorithm"] == "rllib.resource_ppo":
                    row["joint_replay"] = audit_joint_run(run_dir)
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
        + "4096 joint rounds, seed 101; paired evaluation root 202, replications 0–4. "
        + "Joint rounds are not equivalent to centralized decision steps. "
        + "No performance superiority requirement; see report.json for all identities, role updates, individual results and three replay audits.\n"
        + (f"\nError: {report['error']}\n" if "error" in report else "")
    )
    print(
        json.dumps({"status": report["status"], "report": str(output / "report.json")}),
        flush=True,
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
