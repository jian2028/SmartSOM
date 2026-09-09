"""Additional fixed-source coverage gates for the combined macOS acceptance."""

import math
from collections import Counter
from xml.etree import ElementTree

from validation.resource_acceptance import run_identity, training_identity

from smartsom.config.codec import digest

TRAINING_IDENTITIES = {
    "rllib": "66c08a842f78db755ad76288484705d794c723881784623d3f3556cd3695f6ce",
    "sb3": "1b064fa97349b1d354d3e8c843b5ceec702bea5065d5fa77665b5144e6a9ea2c",
    "marl": "351abb17df774a9ee0bd0381082d62840b8db895d9f69aa33e0871124b99deda",
}
CENTRAL_PROVIDERS = {"builtin.spt", "rllib.ppo", "sb3.maskable_ppo"}
CENTRAL_RECIPE_SHA256 = (
    "2afb03bc98abe885b32f09deae6356a6c0039fcd0b7cd4efb2e95ca6f7405926"
)


def _junit_identity(nodeid):
    """Match pytest's classname/name split without splitting parameter contents."""
    if not isinstance(nodeid, str) or not nodeid:
        raise ValueError("required feature nodeids must be nonempty strings")
    address, bracket, parameters = nodeid.partition("[")
    names = address.split("::")
    if len(names) < 2 or any(not name for name in names):
        raise ValueError(f"invalid required feature nodeid: {nodeid!r}")
    names[0] = names[0].replace("/", ".").removesuffix(".py")
    names[-1] += bracket + parameters
    return ".".join(names[:-1]), names[-1]


def require_feature_results(path, expected_nodeids):
    if isinstance(expected_nodeids, (str, bytes)):
        raise ValueError("required feature nodeids must be a collection")
    expected = Counter(_junit_identity(nodeid) for nodeid in expected_nodeids)
    if not expected:
        raise ValueError("required feature collection must not be empty")
    suites = ElementTree.parse(path).getroot()
    cases = list(suites.iter("testcase"))
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("skipped", "failure", "error")
    ):
        raise ValueError("required feature tests must all execute and pass")
    actual = Counter((case.get("classname"), case.get("name")) for case in cases)
    if actual != expected:
        raise ValueError(
            "required feature XML coverage differs from collection: "
            f"missing={dict(expected - actual)}, unexpected={dict(actual - expected)}"
        )
    return {"tests": len(cases), "passed": len(cases), "skipped": 0}


def require_frozen_training(name, resolved):
    if training_identity(resolved) != TRAINING_IDENTITIES[name]:
        raise ValueError(f"{name} differs from the frozen pre-refactor recipe")


def require_central_recipe(resolved_runs):
    rows = [run_identity(resolved) for resolved in resolved_runs]
    identity = digest(
        {
            "version": "smartsom.central-acceptance/v1",
            "runs": sorted(rows, key=digest),
        }
    )
    if identity != CENTRAL_RECIPE_SHA256:
        raise ValueError("evaluation differs from frozen centralized acceptance recipe")


def require_central_coverage(report, rows):
    if (
        report.get("status") != "passed"
        or report.get("completed") != 15
        or report.get("failed") != 0
    ):
        raise ValueError(
            "centralized acceptance requires fifteen successful evaluations"
        )
    if len(rows) != 15 or {r["replication"] for r in rows} != set(range(5)):
        raise ValueError("centralized replication coverage differs")
    for replication in range(5):
        paired = [r for r in rows if r["replication"] == replication]
        if Counter(r["provider"] for r in paired) != Counter(CENTRAL_PROVIDERS):
            raise ValueError(
                "each replication requires exactly one run of each provider"
            )
        if len({r["world_sha256"] for r in paired}) != 1:
            raise ValueError("paired full environment inputs differ")
        if any(
            r["audit_status"] != "passed"
            or not isinstance(r["makespan"], (int, float))
            or not math.isfinite(r["makespan"])
            for r in paired
        ):
            raise ValueError("evaluation audit or finite completion result is missing")
