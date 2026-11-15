"""Read-only views of historical append-only scientific inventories."""

import copy
import json
from pathlib import Path

from smartsom.config.codec import digest


def extend_view(root, plan, state, runtime_state):
    plan, state = copy.deepcopy(plan), copy.deepcopy(state)
    previous = None
    for identity in runtime_state.get("inventory_extensions", []):
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("invalid batch inventory extension identity")
        extension = json.loads((root / "inventory" / (identity + ".json")).read_text())
        if (
            digest(extension) != identity
            or extension["base_plan_sha256"] != state["plan_sha256"]
            or extension["previous"] != previous
        ):
            raise ValueError("frozen batch inventory extension changed")
        if set(extension["inventory"]) & set(plan["inventory"]):
            raise ValueError("batch inventory extension replaces an existing member")
        for key, revision in extension.get("revisions", {}).items():
            if (
                key not in plan["inventory"]
                or digest(plan["inventory"][key]) != revision["previous"]
            ):
                raise ValueError("batch inventory revision predecessor changed")
            plan["inventory"][key] = revision["inventory"]
        known = {row["id"] for row in plan["files"]}
        for row in extension["files"]:
            if row["id"] in known:
                raise ValueError("duplicate extended task identity")
            known.add(row["id"])
            state["files"].setdefault(row["id"], {"status": "deferred"})
        plan["inventory"].update(extension["inventory"])
        plan["files"].extend(extension["files"])
        previous = identity
    for job in runtime_state.get("additions", {}).values():
        if (
            job.get("owner_root") == str(root)
            and job.get("owner_task") in state["files"]
        ):
            state["files"][job["owner_task"]]["status"] = job["status"]
    # A newly published extension can precede the next driver state write.
    # Never expose an old complete aggregate over new deferred membership.
    if state.get("status") == "completed":
        from smartsom.experiments.incremental_batch import completion

        state["status"] = completion(state)
    return plan, state


def read_view(root):
    """Read frozen scientific membership without requiring the active code version."""
    from smartsom.experiments.runtime_control import snapshot

    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if digest(plan) != state.get("plan_sha256"):
        raise ValueError("frozen directory batch changed")
    return extend_view(root, plan, state, snapshot(root))
