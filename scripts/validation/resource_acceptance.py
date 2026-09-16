"""Grid resource recipe and evidence gates; historical item-13 results stay source-bound."""

import json
import math
import re
import subprocess
from collections import Counter

from smartsom.config.codec import digest
from smartsom.config.experiment import ExperimentConfig
from smartsom.config.production import ProductionRecipe
from smartsom.config.study import semantic_run

# Supersedes the matrix recipe for new runs; this is not a new acceptance claim.
RECIPE_VERSION = "smartsom.grid-resource-acceptance/v1"
TRAINING_RECIPE_SHA256 = (
    "74eb89151e4e9ac35f2b5411cbd21b4899f7b4ebbf194b64c952820bb23cf87f"
)
EVALUATION_RECIPE_SHA256 = (
    "4bc24d0620c52adb20b7a0fb9df8f12b4a924abd46af07f996236e548a95918b"
)
PROVIDERS = ("builtin.spt", "rllib.resource_ppo")
ROLES = ("agv_policy", "buffer_policy", "machine_policy", "quality_policy")


def run_identity(prepared):
    if not isinstance(prepared.resolved, ProductionRecipe):
        raise ValueError(
            "historical resource recipes require their recorded source checkout"
        )
    row = semantic_run(prepared)
    row["algorithm"].pop("checkpoint", None)
    row["world_sha256"] = digest(prepared.resolved.scenario)
    return row


def training_identity(prepared):
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    return digest(
        {
            "version": RECIPE_VERSION,
            "seed": config.seed,
            "recipe": run_identity(prepared),
        }
    )


def evaluation_identity(study):
    rows = [
        {
            "case": e.case_id,
            "algorithm": e.algorithm_id,
            "replication": e.replication,
            "variant": e.variant_id,
            "recipe": run_identity(e.resolved),
        }
        for e in study.entries
    ]
    return digest({"version": RECIPE_VERSION, "runs": sorted(rows, key=digest)})


def require_training_recipe(prepared):
    if training_identity(prepared) != TRAINING_RECIPE_SHA256:
        raise ValueError("training differs from frozen resource acceptance recipe")


def require_evaluation_recipe(study):
    if evaluation_identity(study) != EVALUATION_RECIPE_SHA256:
        raise ValueError("evaluation differs from frozen resource acceptance recipe")


def require_matching_source(source, commit, label):
    git = source.get("git", {}) if isinstance(source, dict) else {}
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit) is None
        or git.get("commit") != commit
        or git.get("dirty") is not False
        or git.get("status") != ""
    ):
        raise ValueError(f"{label} requires clean source at implementation {commit}")


def require_integrated_source(root, source):
    commit = source.get("git", {}).get("commit")
    require_matching_source(source, commit, "formal acceptance")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    integrated = subprocess.run(
        ["git", "merge-base", "--is-ancestor", actual, "refs/heads/main"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if actual != commit or integrated.returncode != 0:
        raise ValueError(
            "formal acceptance requires a commit integrated into local main"
        )
    return commit


def require_training_result(audit, directory):
    """Check the audited checkpoint, not an unrelated attempt's latest metrics."""
    if (
        audit.get("status") != "passed"
        or not audit.get("budget_completed")
        or audit.get("environment_steps") != 4096
        or audit.get("ppo_updates") != 16
        or audit.get("provider") != "rllib.resource_ppo"
    ):
        raise ValueError("fixed resource training audit/counts did not pass")
    from pathlib import Path

    checkpoint = Path(audit["checkpoint"])
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    changes = metadata.get("component_changes", {})
    if (
        set(metadata.get("modules", [])) != set(ROLES)
        or set(changes) != set(ROLES)
        or any(
            changes[role].get(part) is not True
            for role in ROLES
            for part in ("actor", "critic")
        )
    ):
        raise ValueError(
            "all four resource actors and critics require actual parameter updates"
        )
    if (
        metadata.get("environment_steps") != audit["environment_steps"]
        or metadata.get("learner_updates") != audit["ppo_updates"]
    ):
        raise ValueError("checkpoint and audited training counts disagree")
    metrics = [
        json.loads(line)
        for line in (checkpoint / "learner_metrics.jsonl").read_text().splitlines()
    ]
    if [row["sampled_steps"] for row in metrics] != list(range(256, 4097, 256)):
        raise ValueError("fixed resource training requires all 16 optimizer records")
    for row in metrics:
        for role in ROLES:
            clipped = row["metrics"].get(f"{role}/vf_loss")
            unclipped = row["metrics"].get(f"{role}/vf_loss_unclipped")
            if not all(
                type(x) in (int, float) and math.isfinite(x)
                for x in (clipped, unclipped)
            ) or not math.isclose(clipped, unclipped, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(
                    "fixed resource critic value loss is saturated/nonfinite"
                )


def require_evaluation_result(report):
    rows = report["evaluation"]
    expected = Counter((r, p) for r in range(5) for p in PROVIDERS)
    if Counter((r["replication"], r["algorithm"]) for r in rows) != expected:
        raise ValueError(
            "resource evaluation requires one MARL/SPT pair per replication"
        )
    for replication in range(5):
        pair = [r for r in rows if r["replication"] == replication]
        if len({r["world_sha256"] for r in pair}) != 1:
            raise ValueError("resource evaluation paired input mismatch")
    required_checks = {
        "frozen_inputs",
        "state_hashes",
        "semantic_commands",
        "events",
        "qualified_demand_coverage",
        "observation_replay",
    }
    for row in rows:
        if row.get("status") != "passed" or not required_checks <= set(
            row.get("checks", [])
        ):
            raise ValueError(
                "resource evaluation physical/observation audit did not pass"
            )
        if (
            row["algorithm"] == "rllib.resource_ppo"
            and row.get("learning_replay", {}).get("status") != "passed"
        ):
            raise ValueError(
                "resource evaluation learning decision replay did not pass"
            )
    if (report.get("completed"), report.get("failed"), report.get("pending")) != (
        10,
        0,
        0,
    ):
        raise ValueError("resource evaluation requires all ten runs completed")
