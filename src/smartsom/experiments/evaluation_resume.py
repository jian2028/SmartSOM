"""Verified, stage-local case commits. Scientific outcomes are never retried to pass."""

import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from smartsom._filesystem import atomic_replace, native_path
from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.experiments import locking

SCHEMA = "smartsom.evaluation-cases/v1"


@contextmanager
def stage_lock(root):
    """Reject another writer for the whole evaluation, including final summaries."""
    root = native_path(Path(root))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".evaluation.lock").open("a+b") as stream:
        try:
            locking.acquire(stream)
        except BlockingIOError as exc:
            raise RuntimeError(
                "evaluation stage is already running: " + str(root)
            ) from exc
        try:
            yield
        finally:
            locking.release(stream)


def durable_json(path, value):
    """A receipt becomes visible only after its contents have reached the OS."""
    path = native_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(primitive(value)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(Path(temporary), path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def selected_cases(prepared, purpose=None):
    """Select a prefix per frozen scenario; never renumber or regenerate seeds."""
    cases = json.loads(prepared.evaluation_json)
    options = prepared.config.evaluation
    count = getattr(options, "replications", None)
    if count is None:
        return cases  # Older prepared interfaces already freeze the entire target.
    if purpose == "initial_control":
        count = getattr(options, "initial_control_replications", None)
        if count is None:
            count = options.replications
    # Explicit frozen-input suites have distinct case IDs with replication=0.
    recipe = json.loads(prepared.training_inputs_json)
    if recipe.get("frozen_input_contract"):
        # Historically explicit suites use their entire frozen list, regardless
        # of the convenience replications default. Only an explicit control
        # target selects a prefix of that list.
        if (
            purpose == "initial_control"
            and getattr(options, "initial_control_replications", None) is not None
        ):
            if count > len(cases):
                raise ValueError("initial control target exceeds the frozen suite")
            return cases[:count]
        return [] if count == 0 else cases
    return [case for case in cases if case["replication"] < count]


def _sha(path):
    checksum = hashlib.sha256()
    with native_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


class CaseStore:
    """Used under stage_lock; one receipt is the sole completion authority."""

    def __init__(self, root, prepared, cases, phase, *, annotations=None, context=None):
        self.attempt = uuid4().hex
        self.root = Path(root)
        self.cases = cases
        self.phase = phase
        self.annotations = annotations or {}
        if context is None:
            raise ValueError(
                "current case recovery requires an explicit frozen identity"
            )
        context = copy.deepcopy(context)
        self.ids = [digest({"identity": context, "case": case}) for case in cases]
        keys = [(case["case"], case["replication"]) for case in cases]
        if len(set(self.ids)) != len(cases) or len(set(keys)) != len(keys):
            raise ValueError("duplicate frozen evaluation case identity")
        self.plan = {
            "schema": SCHEMA,
            "identity": context,
            "annotations": self.annotations,
            "cases": [
                {"id": key, "case": case}
                for key, case in zip(self.ids, cases, strict=True)
            ],
        }
        plan_path = self.root / "plan.json"
        if plan_path.exists():
            if self.read(plan_path) != self.plan:
                raise ValueError(
                    "evaluation case plan/identity changed (model, inputs, seeds, phase "
                    "or total target); create a new evaluation"
                )
        else:
            if any(self.root.glob("case-*.json")):
                raise ValueError("evaluation receipts lack their frozen plan")
            durable_json(plan_path, self.plan)
        self.rows = {}
        for index, case_id in enumerate(self.ids):
            path = self.root / ("case-" + case_id + ".json")
            if not path.exists():
                continue
            receipt = self.read(path)
            if not isinstance(receipt, dict):
                raise ValueError("corrupt evaluation case receipt: " + str(path))
            checksum = receipt.pop("sha256", None)
            if (
                checksum != digest(receipt)
                or receipt.get("schema") != SCHEMA
                or receipt.get("id") != case_id
                or receipt.get("phase") != phase
                or receipt.get("index") != index
            ):
                raise ValueError("corrupt evaluation case receipt: " + str(path))
            self._validate(index, receipt.get("result", {}))
            self.rows[index] = receipt["result"]
        expected = {"case-" + key + ".json" for key in self.ids}
        if any(path.name not in expected for path in self.root.glob("case-*.json")):
            raise ValueError("unexpected evaluation case receipt")
        self.historical = len(self.rows)

    @staticmethod
    def read(path):
        try:
            return json.loads(native_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("corrupt evaluation case data: " + str(path)) from exc

    def _validate(self, index, row):
        case = self.cases[index]
        if (
            row.get("case_id") != case["case"]
            or row.get("replication") != case["replication"]
            or row.get("seed") != case["seed"]
            or row.get("world_sha256") != digest(case["scenario"])
            or row.get("engineering_failure") is not False
            or row.get("status") not in {"completed", "truncated"}
            or any(row.get(key) != value for key, value in self.annotations.items())
        ):
            raise ValueError(
                "evaluation case result does not match its frozen identity"
            )

    def commit(self, index, row):
        self._validate(index, row)
        if index in self.rows:
            raise ValueError("evaluation case already committed")
        receipt = {
            "schema": SCHEMA,
            "phase": self.phase,
            "id": self.ids[index],
            "index": index,
            "result": row,
        }
        receipt["sha256"] = digest(receipt)
        durable_json(self.root / ("case-" + self.ids[index] + ".json"), receipt)
        self.rows[index] = copy.deepcopy(row)

    def progress(self):
        pending = [i for i in range(len(self.cases)) if i not in self.rows]
        return {
            "evaluation_attempt": self.attempt,
            "evaluation_phase": self.phase,
            "evaluation_historical_completed": self.historical,
            "evaluation_current_completed": len(self.rows) - self.historical,
            "evaluation_remaining": len(pending),
            "evaluation_target": len(self.cases),
            "evaluation_finished": len(self.rows),
            "evaluation_requested": len(self.cases),
            "evaluation_remaining_ticks": sum(
                self.cases[i]["scenario"]["tick_limit"] for i in pending
            ),
        }
