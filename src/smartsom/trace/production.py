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
    def __init__(self, source, *, cache_summaries=False, progress=None):
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
        self._summaries = {}
        self._summary_bytes = 0
        self._cache_summaries = cache_summaries
        self._progress = progress
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
                if self._cache_summaries:
                    summary = json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "tick",
                                "state",
                                "events",
                                "actions",
                                "rejections",
                            )
                            if key in row
                        }
                    ).encode()
                    if self._summary_bytes + len(summary) <= 32 * 1024 * 1024:
                        self._summaries[row["tick"]] = summary
                        self._summary_bytes += len(summary)
                if self._progress and len(self.offsets) % 5 == 0:
                    self._progress(stream.tell(), path.stat().st_size)
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

    def summary_row(self, tick):
        """Detached validated state/event projection; full decisions stay on disk."""
        if tick in self._summaries:
            return json.loads(self._summaries[tick])
        return self.row(tick)

    @property
    def last_tick(self):
        return len(self.offsets) - 1
