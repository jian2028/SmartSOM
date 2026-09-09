"""Fixed item-13 recipe, source eligibility and complete paired-run coverage."""

import json
import math
import re
import subprocess
from collections import Counter

from smartsom.config.codec import digest
from smartsom.config.study import semantic_run
from smartsom.config.training import episode_input

RECIPE_VERSION = "smartsom.resource-acceptance/v1"
# Frozen from the authorized micro/seed101/4096 recipe and seed202 five pairs.
# Output locations and checkpoint bytes are intentionally outside recipe identity.
TRAINING_RECIPE_SHA256 = (
    "351abb17df774a9ee0bd0381082d62840b8db895d9f69aa33e0871124b99deda"
)
EVALUATION_RECIPE_SHA256 = (
    "9c28dd2a73a4e07cf5ace40692b17112f8891f679bef06c0d5a07b0d2faa9be3"
)
PROVIDERS = ("builtin.spt", "rllib.resource_ppo")


def training_identity(resolved):
    return digest(
        {
            "version": RECIPE_VERSION,
            "seed": resolved.run.seed,
            "budget": resolved.run.budget,
            "objective": resolved.run.objective,
            "algorithm": resolved.algorithm,
            "framework_seed": resolved.framework_seed,
            "episode_seed_version": resolved.episode_seed_version,
            "scenario": semantic_run(resolved.base)["scenario"],
            "input": episode_input(resolved.base),
        }
    )


def run_identity(resolved):
    row = semantic_run(resolved)
    for key in ("checkpoint", "checkpoint_sha256"):
        row["algorithm"]["algorithm"].pop(key, None)
    row["world_sha256"] = digest(episode_input(resolved))
    return row


def evaluation_identity(study):
    rows = [run_identity(e.resolved) for e in study.entries]
    return digest({"version": RECIPE_VERSION, "runs": sorted(rows, key=digest)})


def require_training_recipe(resolved):
    if training_identity(resolved) != TRAINING_RECIPE_SHA256:
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
    if (
        audit.get("status") != "passed"
        or audit.get("environment_steps") != 4096
        or audit.get("agent_steps") != 49152
        or audit.get("learner_updates") != 16
    ):
        raise ValueError("fixed resource training audit/counts did not pass")
    weights = audit.get("role_weights", [])
    if Counter(w["role"] for w in weights) != Counter(
        ("machine_policy", "agv_policy")
    ) or any(w["initial_sha256"] == w["final_sha256"] for w in weights):
        raise ValueError("both resource roles require actual parameter updates")
    metrics = [
        json.loads(line)
        for line in (directory / "learner_metrics.jsonl").read_text().splitlines()
    ]
    if len(metrics) != 16:
        raise ValueError("fixed resource training requires 16 optimizer records")
    for row in metrics:
        for role in ("machine_policy", "agv_policy"):
            clipped = row["metrics"][f"{role}/vf_loss"]
            unclipped = row["metrics"][f"{role}/vf_loss_unclipped"]
            if not all(
                math.isfinite(x) for x in (clipped, unclipped)
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
        "action_replay",
        "schedule_replay",
        "action_observations",
        "schedule_observations",
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
            and row.get("joint_replay", {}).get("status") != "passed"
        ):
            raise ValueError("resource evaluation joint replay did not pass")
    if (report.get("completed"), report.get("failed"), report.get("pending")) != (
        10,
        0,
        0,
    ):
        raise ValueError("resource evaluation requires all ten runs completed")
