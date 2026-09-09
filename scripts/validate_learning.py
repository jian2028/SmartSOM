"""Fixed-budget training audits and the preregistered 15-run paired evaluation."""

import argparse
import json
from pathlib import Path

import yaml
from validation.idetc_audit import audit_run

from smartsom.config import load_resolved_run, resolve_study, resolve_training_run
from smartsom.config.codec import digest, read_model
from smartsom.config.models import AlgorithmFile, LearningAlgorithm
from smartsom.experiments import run_batch
from smartsom.experiments.catalog import training_locator
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.packaging import model_locator
from smartsom.experiments.training_audit import audit_training, load_training_snapshot
from smartsom.learning.checkpoint import CheckpointManifest, file_hash

ROOT = Path(__file__).resolve().parents[1]


def validate_fixed_training(name, directory):
    actual = load_training_snapshot(directory / "resolved_training.json")
    expected = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")

    def scientific_identity(resolved):
        return digest(
            {
                "seed": resolved.run.seed,
                "budget": resolved.run.budget,
                "objective": resolved.run.objective,
                "algorithm": resolved.algorithm,
                "factory": resolved.base.factory,
                "workload": resolved.base.workload,
                "scenario": resolved.base.scenario.model_dump(
                    exclude={"factory", "workload"}
                ),
            }
        )

    if scientific_identity(actual) != scientific_identity(expected):
        raise ValueError(f"{name} training differs from the fixed acceptance recipe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rllib-training-dir", type=Path, required=True)
    parser.add_argument("--sb3-training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.rllib_training_dir = training_locator(args.rllib_training_dir)
    args.sb3_training_dir = training_locator(args.sb3_training_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)  # Never overwrite acceptance evidence.
    report = {
        "status": "running",
        "source": source_identity(),
        "training": {},
        "evaluation": [],
    }
    try:
        for name, directory in (
            ("rllib", args.rllib_training_dir),
            ("sb3", args.sb3_training_dir),
        ):
            validate_fixed_training(name, directory)
            report["training"][name] = audit_training(directory)
            write_json(output / "report.json", report)
        template = ROOT / "configs/studies/learning_evaluation.yaml"
        spec = yaml.safe_load(template.read_text())
        spec["cases"][0]["scenario"] = str(
            (template.parent / spec["cases"][0]["scenario"]).resolve()
        )
        references = {}
        for name, directory in (
            ("RLlib-PPO", args.rllib_training_dir),
            ("SB3-MaskablePPO", args.sb3_training_dir),
        ):
            checkpoint = model_locator(directory)
            metadata = read_model(checkpoint / "checkpoint.json", CheckpointManifest)[0]
            algorithm_path = output / f"{name}.json"
            write_json(
                algorithm_path,
                AlgorithmFile(
                    schema="smartsom.algorithm/v1",
                    algorithm=LearningAlgorithm(
                        provider=metadata.provider,
                        projection=metadata.projection,
                        parameters=metadata.parameters,
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=file_hash(checkpoint / "checkpoint.json"),
                    ),
                ),
            )
            references[name] = algorithm_path
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
        if len(resolved.entries) != 15:
            raise ValueError("acceptance must contain exactly 15 paired evaluations")
        batch = run_batch(
            resolved,
            workers=args.workers,
            on_progress=lambda row: print(json.dumps(row), flush=True),
        )
        report["study_dir"] = str(batch.study_dir)
        report["completed"] = batch.completed
        report["failed"] = batch.failed
        # The batch attempt index is authoritative; failed children remain retained.
        for manifest_path in sorted(batch.study_dir.rglob("manifest.json")):
            run_dir = manifest_path.parent
            if not (run_dir / "resolved_run.yaml").exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] != "completed":
                report["evaluation"].append(
                    {
                        "status": "failed",
                        "run_dir": str(run_dir),
                        "failure": json.loads((run_dir / "failure.json").read_text()),
                    }
                )
                continue
            row = audit_run(run_dir, expected_jobs=4, expected_operations=9)
            run = load_resolved_run(run_dir / "resolved_run.yaml")
            row.update(
                algorithm=run.algorithm.algorithm.provider,
                replication=run.study_seed_origin.replication,
                world_sha256=digest(
                    {
                        "factory": run.factory,
                        "workload": run.workload,
                        "arrivals": run.arrivals,
                        "quality": run.quality,
                    }
                ),
            )
            report["evaluation"].append(row)
            write_json(output / "report.json", report)
        for replication in range(5):
            rows = [
                r for r in report["evaluation"] if r.get("replication") == replication
            ]
            if len(rows) != 3 or len({r["world_sha256"] for r in rows}) != 1:
                raise ValueError("paired evaluation input/coverage mismatch")
        report["status"] = (
            "passed"
            if batch.completed == 15 and not batch.failed and not batch.pending
            else "failed"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(output / "report.json", report)
    (output / "report.md").write_text(
        f"# Learning acceptance\n\nStatus: **{report['status']}**\n\n"
        + f"Completed evaluations: {report.get('completed', 0)}/15; failed: {report.get('failed', 0)}.\n\n"
        + "Fixed seeds and budgets; no performance superiority requirement. See report.json for source, training audits and individual replay evidence.\n"
        + (f"\nError: {report['error']}\n" if "error" in report else "")
    )
    print(
        json.dumps({"status": report["status"], "report": str(output / "report.json")}),
        flush=True,
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
