"""Portable full-tick records and visual playback, independent of engine execution."""

import copy
import hashlib
import json
import os
from pathlib import Path

from smartsom.config.codec import primitive

RUN_SCHEMA = "smartsom.production-run/v1"
TRACE_SCHEMA = "smartsom.production-tick/v1"


def canonical(value):
    return json.dumps(
        primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def state_hash(state):
    return hashlib.sha256(canonical(state).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_checkpoint(directory):
    """Inventory immutable payload members before publishing checkpoint metadata."""
    directory = Path(directory)
    metadata = json.loads((directory / "checkpoint.json").read_text())
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path not in (
            directory / "checkpoint.json",
            directory / "update.json",
        ):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": str(path.relative_to(directory)), "sha256": digest})
    metadata["files"] = files
    atomic_json(directory / "checkpoint.json", metadata)


def observation_record(view, mode):
    """Freeze the public input actually passed to a rule decision owner."""
    import copy

    return {
        "sha256": state_hash(view),
        **({"observation": copy.deepcopy(view)} if mode == "full" else {}),
    }


class Recorder:
    def __init__(self, directory, inputs, initial, *, record=True, source=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "schema": RUN_SCHEMA,
            "id": self.directory.name,
            "name": self.directory.name,
            "kind": "simulation",
            "provider": primitive(inputs).get("algorithm", {}).get("provider"),
            "seed": primitive(inputs).get("scenario", {}).get("seed"),
            "inputs": primitive(inputs),
            "source": source,
            "initial_state": initial,
            "status": "running",
            "last_tick": 0,
            "trace": "trace.jsonl" if record else None,
        }
        atomic_json(self.directory / "run.json", self.manifest)
        self.stream = (
            (self.directory / "trace.jsonl").open("w", encoding="utf-8")
            if record
            else None
        )
        self.last = initial

    def append(self, result):
        if result["tick"] != self.last["tick"] + 1:
            raise ValueError("non-contiguous tick record")
        if self.stream:
            row = {
                "schema": TRACE_SCHEMA,
                **result,
                "state_hash": state_hash(result["state"]),
            }
            self.stream.write(canonical(row) + "\n")
            self.stream.flush()
        self.last = result["state"]

    def finish(self, status, error=None):
        try:
            if self.stream and not self.stream.closed:
                self.stream.flush()
                self.stream.close()
            self.manifest.update(
                status=status,
                last_tick=self.last["tick"],
                result=self.last,
                failure=error,
            )
            atomic_json(self.directory / "run.json", self.manifest)
        finally:
            if self.stream and not self.stream.closed:
                self.stream.close()


class Playback:
    def __init__(self, source):
        root = Path(source)
        if root.is_file():
            root = root.parent
        self.root = root
        self.manifest = json.loads((root / "run.json").read_text())
        if self.manifest.get("schema") != RUN_SCHEMA:
            raise ValueError(
                "unsupported run format; historical artifacts are not grid playback"
            )
        if self.manifest.get("trace") != "trace.jsonl":
            raise ValueError("this run has no recorded trajectory")
        if self.manifest.get("initial_state", {}).get("tick") != 0:
            raise ValueError("run requires a tick-zero initial state")
        self.offsets = [None]
        self.partial = False
        self._index()

    def _index(self):
        path = self.root / "trace.jsonl"
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self.partial = True
                    break
                row = json.loads(line)
                self._validate(row, len(self.offsets))
                self.offsets.append(offset)
        if self.manifest["status"] in ("completed", "truncated") and (
            self.partial or len(self.offsets) - 1 != self.manifest["last_tick"]
        ):
            raise ValueError("finished run has incomplete trajectory")

    @staticmethod
    def _validate(row, tick):
        if row.get("schema") != TRACE_SCHEMA or row["tick"] != tick:
            raise ValueError("invalid trace schema or tick sequence")
        if row.get("state", {}).get("tick") != tick:
            raise ValueError("state tick does not match trace tick")
        if row["state_hash"] != state_hash(row["state"]):
            raise ValueError("trace state hash mismatch")

    def row(self, tick):
        if type(tick) is not int or not 0 <= tick < len(self.offsets):
            raise ValueError("tick is outside recorded trajectory")
        if tick == 0:
            return {
                "tick": 0,
                "state": copy.deepcopy(self.manifest["initial_state"]),
                "events": [],
            }
        with (self.root / "trace.jsonl").open("rb") as stream:
            stream.seek(self.offsets[tick])
            row = json.loads(stream.readline())
        self._validate(row, tick)
        return row

    @property
    def last_tick(self):
        return len(self.offsets) - 1


def audit(source):
    """Re-execute semantic commands and compare every state, distinct from viewing."""
    from smartsom.config.production import scenario_from_snapshot

    root = Path(source)
    if root.is_file():
        root = root.parent
    if json.loads((root / "run.json").read_text()).get("schema") != RUN_SCHEMA:
        from smartsom.experiments.production_audit import audit_tree

        return audit_tree(root)
    recording = Playback(source)
    algorithm = recording.manifest["inputs"]["algorithm"]
    if recording.manifest["provider"] != algorithm["provider"]:
        raise ValueError("run provider differs from frozen algorithm")
    learned = algorithm["provider"].startswith(("sb3.", "rllib."))
    contract = recording.manifest.get("learning_contract")
    if learned != (contract is not None):
        raise ValueError("missing or unexpected learning evidence contract")
    if contract is not None:
        declared = {k: v for k, v in algorithm.items() if k != "checkpoint"}
        retained = {k: v for k, v in contract["algorithm"].items() if k != "checkpoint"}
        if canonical(declared) != canonical(retained):
            raise ValueError("learning evidence differs from frozen algorithm")
    checker = ExecutionAudit(
        scenario_from_snapshot(recording.manifest["inputs"]["scenario"]),
        recording.row(0)["state"],
        recording.manifest.get("learning_contract"),
        observations=recording.manifest.get("observations"),
    )
    for tick in range(1, recording.last_tick + 1):
        checker.append(recording.row(tick))
    checker.finish_pending(recording.manifest.get("pending_decisions", []))
    sim = checker.sim
    if recording.manifest["status"] in ("completed", "truncated"):
        if state_hash(recording.manifest.get("result")) != state_hash(sim.snapshot()):
            raise ValueError("run result differs from its recorded trajectory")
        if recording.manifest["status"] == "completed" and sim.status != "completed":
            raise ValueError(
                "run claims completion but recorded simulation is unfinished"
            )
    return {
        "status": "passed"
        if recording.manifest["status"] == "completed"
        else "partial_verified",
        "ticks": recording.last_tick,
        "trajectory_complete": sim.status == "completed",
        "run_status": recording.manifest["status"],
        "physical_status": sim.status,
        "qualified_demands": len(sim.completed),
        **({"learning": checker.learning.result()} if checker.learning else {}),
    }


class ExecutionAudit:
    """Streaming execution verification independent of recording and rendering."""

    def __init__(self, scenario, initial, learning_contract=None, *, observations=None):
        from pydantic import TypeAdapter

        from smartsom.domain.production import JointCommand
        from smartsom.engine.production import ProductionSimulator

        self.sim = ProductionSimulator(scenario)
        self.observations = observations
        self.learning = None
        if learning_contract is not None:
            from smartsom.learning.production_replay import GridLearningAudit

            self.learning = GridLearningAudit(scenario, learning_contract)
        self.adapter = TypeAdapter(JointCommand)
        if canonical(self.sim.snapshot()) != canonical(initial):
            raise ValueError("initial state differs from frozen inputs")

    def append(self, row):
        if not self.learning and self.observations:
            expected = {
                "ranking": observation_record(self.sim.decision(), self.observations),
                "action": observation_record(
                    self.sim.decision(row["rankings"]), self.observations
                ),
            }
            if canonical(expected) != canonical(row.get("rule_decision")):
                raise ValueError(
                    f"execution audit differs at tick {row['tick']}: rule observation"
                )
        if self.learning:
            self.learning.append(row)
        actual = self.sim.step(self.adapter.validate_json(canonical(row["actions"])))
        for field in ("state", "events", "rejections", "reward"):
            if canonical(actual[field]) != canonical(row[field]):
                raise ValueError(
                    f"execution audit differs at tick {row['tick']}: {field}"
                )

    def finish_pending(self, decisions):
        if self.learning:
            self.learning.pending(decisions)
        elif decisions:
            raise ValueError("pending decisions require a learning evidence contract")

    def result(self):
        return {
            "status": "passed"
            if self.sim.status == "completed"
            else "partial_verified",
            "ticks": self.sim.tick,
            "trajectory_complete": self.sim.status == "completed",
            "run_status": self.sim.status,
            "qualified_demands": len(self.sim.completed),
            **({"learning": self.learning.result()} if self.learning else {}),
        }
