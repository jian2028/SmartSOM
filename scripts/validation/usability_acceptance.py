"""Grid coverage gates; historical pre-grid acceptance remains source-bound.

These recipe identities describe the explicit grid input migration, not a renewed
acceptance of historical matrix results. Passing gates still requires every run.
"""

import math
from collections import Counter
from xml.etree import ElementTree

from validation.resource_acceptance import run_identity, training_identity

from smartsom.config.codec import digest

TRAINING_IDENTITIES = {
    "rllib": "f5ba3745a8c3180cf5042ce9fefbbbf5abd1b4842e5c1bf3f9357d56c79af1e5",
    "sb3": "4ddcc3754315e44c3d8dc84267c67cb14bab00e034d6a621ce9ba86486b3e4e3",
    "marl": "74eb89151e4e9ac35f2b5411cbd21b4899f7b4ebbf194b64c952820bb23cf87f",
}
CENTRAL_PROVIDERS = {"builtin.spt", "rllib.ppo", "sb3.maskable_ppo"}
CENTRAL_RECIPE_SHA256 = (
    "3ab063b9ff4639614d2184ae24a040168096d243a43f2d922c9ed854eb92c9e1"
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
        raise ValueError(f"{name} differs from the frozen grid recipe")


def require_central_recipe(resolved_runs):
    rows = [run_identity(resolved) for resolved in resolved_runs]
    identity = digest(
        {
            "version": "smartsom.grid-central-acceptance/v1",
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
