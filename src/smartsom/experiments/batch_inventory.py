"""Append-only scientific membership, committed by the queue's atomic journal.

Original plan/manifest and existing trial archives never change. A prepared
extension is invisible until its digest is published in runtime.json. Interrupted
preparation can only reuse its own verified, unstarted child snapshots.
"""

import copy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest
from smartsom.experiments.evidence import write_json


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


def validate(compiled, plan, root):
    """Only monotonic membership growth is allowed; old science stays exact."""
    if (
        not plan.get("incremental")
        or plan["directory"] != str(compiled.directory)
        or any(key not in compiled.inventory for key in plan["inventory"])
    ):
        raise ConfigurationError(
            "batch directory or scientific inventory changed; choose a new output root"
        )
    from smartsom.config.codec import read_document, safe_yaml_load
    from smartsom.config.launch import scientific_identity

    for key, old in plan["inventory"].items():
        new = compiled.inventory[key]
        if digest(old) == digest(new):
            continue
        old_sources = {name: (path, sha) for name, path, sha in old["source_files"]}
        new_sources = {name: (path, sha) for name, path, sha in new["source_files"]}
        if (
            old["entry"] != new["entry"]
            or not old.get("seed_independent_sha256")
            or old["seed_independent_sha256"] != new.get("seed_independent_sha256")
            or digest(old["components"]) != digest(new["components"])
            or old["batch"] != new["batch"]
            or old_sources.keys() != new_sources.keys()
            or any(
                old_sources[name] != new_sources[name]
                for name in old_sources
                if name != "experiment"
            )
            or old_sources.get("experiment", (None,))[0]
            != new_sources.get("experiment", (None,))[0]
        ):
            raise ConfigurationError(
                "batch scientific inventory changed; choose a new output root"
            )
        old_rows = [
            row for row in plan["files"] if row["id"].startswith(key + "--seed-")
        ]
        new_rows = [
            row for row in compiled.files if row["id"].startswith(key + "--seed-")
        ]
        if not old_rows or not new_rows:
            raise ConfigurationError(
                "batch task identities changed; choose a new output root"
            )
        anchor, candidate = old_rows[0], new_rows[0]
        if any(
            item.startswith(
                ("experiment.train_seeds=", "experiment.run.seed=", "seed=")
            )
            for item in candidate["compiled"].overrides
        ):
            raise ConfigurationError(
                "batch scientific inventory changed; choose a new output root"
            )
        if anchor["task"] != "train" or any(
            anchor[k] != candidate[k] for k in ("stage", "parallel_files")
        ):
            raise ConfigurationError(
                "batch task identities changed; choose a new output root"
            )
        old_ex = read_document(Path(root) / anchor["child"] / "inputs/experiment.yaml")
        documents = {
            name: safe_yaml_load(value)
            for name, value in candidate["compiled"].documents
        }
        documents["experiment"]["train_seeds"] = old_ex["train_seeds"]
        documents["experiment"]["run"]["seed"] = old_ex["run"]["seed"]
        documents["experiment"]["run"]["output"] = old_ex["run"]["output"]
        if scientific_identity(documents) != anchor["scientific_sha256"][0]:
            raise ConfigurationError(
                "batch scientific inventory changed; choose a new output root"
            )


def prepare(root, compiled, plan, state, queue_state):
    """Called under the owning queue transaction (and driver claim if inactive)."""
    from smartsom.experiments.reservation import allocate, load
    from smartsom.experiments.reservation_batch import frozen_row

    # Reject stale views before preparing files; a concurrent append may have
    # committed after the caller read the plan but before it acquired this lock.
    base = json.loads((root / "plan.json").read_text())
    saved = json.loads((root / "batch.json").read_text())
    current, _ = extend_view(root, base, saved, queue_state)
    if digest(current["inventory"]) != digest(plan["inventory"]) or [
        row["id"] for row in current["files"]
    ] != [row["id"] for row in plan["files"]]:
        raise ValueError("batch inventory changed during submission; retry")
    validate(compiled, plan, root)
    new_inventory = {
        k: v for k, v in compiled.inventory.items() if k not in plan["inventory"]
    }
    revisions = {
        k: {"previous": digest(v), "inventory": compiled.inventory[k]}
        for k, v in plan["inventory"].items()
        if digest(v) != digest(compiled.inventory[k])
    }
    if not new_inventory and not revisions:
        return plan, state
    known = {row["id"] for row in plan["files"]}
    members = [row for row in compiled.files if row["id"] not in known]
    previous = queue_state.get("inventory_extensions", [])
    request = {
        "base_plan_sha256": state["plan_sha256"],
        "previous": previous[-1] if previous else None,
        "inventory": new_inventory,
    }
    if revisions:
        request["revisions"] = revisions
    directory = root / "inventory"
    directory.mkdir(exist_ok=True)
    intent = directory / ("pending-" + digest(request) + ".json")
    for pending in directory.glob("pending-*.json"):
        record = json.loads(pending.read_text())
        if (
            pending != intent
            and digest(record["extension"]) not in previous
            and record["request"].get("base_plan_sha256") == request["base_plan_sha256"]
            and record["request"].get("previous") == request["previous"]
        ):
            raise ValueError(
                f"unfinished batch submission; restore the prior Experiment YAML "
                f"from authored_sources in {pending} and retry the original submission"
            )
    if intent.exists():
        prepared = json.loads(intent.read_text())
        if digest(prepared["request"]) != digest(request):
            raise ValueError("pending batch inventory identity changed")
        extension = prepared["extension"]
    else:
        staging = directory / ("staging-" + uuid4().hex)
        rows = []
        authored_sources = {}
        for member in list(new_inventory.values()) + [
            item["inventory"] for item in revisions.values()
        ]:
            for name, source_path, expected in member["source_files"]:
                if name == "experiment":
                    raw = Path(source_path).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError(
                            "Experiment YAML changed during submission; retry"
                        )
                    authored_sources[source_path] = raw.decode("utf-8")
        for row in members:
            final = root / row["child"]
            if final.exists():
                for pending in directory.glob("pending-*.json"):
                    record = json.loads(pending.read_text())
                    if any(
                        item["child"] == row["child"]
                        for item in record["extension"]["files"]
                    ):
                        raise ValueError(
                            f"uncommitted child {final}; restore the prior Experiment YAML "
                            f"from authored_sources in {pending} and retry the original submission"
                        )
                raise ValueError(
                    f"unowned batch child already exists: {final}; if a prior "
                    "submission was interrupted, restore its original Experiment "
                    "YAML and retry it before editing the seed list again"
                )
            child = staging / row["child"]
            allocate(row["compiled"], root=child, tracking_group=plan["tracking_group"])
            child_plan = load(child)
            rows.append(frozen_row(row, child_plan, row["child"]))
        extension = {**request, "files": rows}
        prepared = {
            "request": request,
            "extension": extension,
            "staging": str(staging),
            "authored_sources": authored_sources,
        }
        write_json(intent, prepared)
    # Validate every prepared snapshot before moving any. A restart never adopts
    # arbitrary existing results, a changed request, or a started trial.
    lookup = {row["id"]: row for row in members}
    if set(lookup) != {row["id"] for row in extension["files"]}:
        raise ValueError("pending batch task identities changed")
    for row in extension["files"]:
        expected = lookup[row["id"]]
        if row["child"] != expected["child"] or row["scientific_sha256"] != [
            expected["scientific_sha256"]
        ]:
            raise ValueError("pending batch task science changed")
        final = root / row["child"]
        staged = Path(prepared["staging"]) / row["child"]
        child = final if final.exists() else staged
        if not child.resolve().is_relative_to(root.resolve()):
            raise ValueError("pending batch child escapes its run")
        if digest(load(child)) != row["child_plan_sha256"]:
            raise ValueError("pending batch child changed")
        if json.loads((child / "batch.json").read_text())[
            "status"
        ] != "allocated" or any(
            (child / name).exists()
            for name in ("progress.json", "continuation", "evaluation", "last.pt")
        ):
            raise ValueError("pending inventory contains a started trial")
    for row in extension["files"]:
        final = root / row["child"]
        if not final.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            (Path(prepared["staging"]) / row["child"]).rename(final)
    identity = digest(extension)
    path = directory / (identity + ".json")
    if path.exists() and digest(json.loads(path.read_text())) != identity:
        raise ValueError("frozen batch inventory extension changed")
    if not path.exists():
        write_json(path, extension)
    queue_state.setdefault("inventory_extensions", []).append(identity)
    plan = copy.deepcopy(plan)
    state = copy.deepcopy(state)
    plan["inventory"].update(new_inventory)
    for key, revision in revisions.items():
        plan["inventory"][key] = revision["inventory"]
    plan["files"].extend(extension["files"])
    for row in extension["files"]:
        state["files"][row["id"]] = {"status": "deferred"}
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
