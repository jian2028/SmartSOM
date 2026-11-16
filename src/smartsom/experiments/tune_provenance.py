"""Immutable execution observations, separate from Tune's retry decisions."""

import json
import time
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest
from smartsom.experiments.evidence import write_json


def directory(template, trial_id):
    if not trial_id or Path(trial_id).name != trial_id or trial_id in {".", ".."}:
        raise ValueError("invalid Tune trial identity")
    return Path(template) / "control" / "tune-provenance" / trial_id


def validate(execution, trial_id):
    if (
        not isinstance(execution, dict)
        or execution.get("trial_id") != trial_id
        or not isinstance(execution.get("events"), dict)
        or not execution.get("producer_attempt")
    ):
        raise ValueError("Tune checkpoint lacks matching execution provenance")
    started = set()
    for key, event in execution["events"].items():
        if (
            not isinstance(event, dict)
            or event.get("trial_id") != trial_id
            or not event.get("event_id")
            or digest(event) != key
        ):
            raise ValueError("execution provenance digest mismatch")
        if event.get("event") == "actor_started":
            started.add(event.get("attempt_id"))
    if execution["producer_attempt"] not in started:
        raise ValueError("checkpoint producer has no execution origin")
    for event in execution["events"].values():
        if event.get("event") == "checkpoint_restored" and (
            event.get("attempt_id") not in started
            or event.get("source_attempt") not in started
            or len(event.get("source_checkpoint_digest", "")) != 64
        ):
            raise ValueError("checkpoint recovery chain is incomplete")


def pending_events(template, trial_id, events):
    """Validate existing history before any scientific restore writes."""
    target = directory(template, trial_id)
    for path in target.glob("*.json"):
        existing = json.loads(path.read_text())
        if digest(existing) != path.stem or existing.get("trial_id") != trial_id:
            raise ValueError("execution provenance digest mismatch")
    pending = []
    for key, event in events.items():
        if digest(event) != key or event.get("trial_id") != trial_id:
            raise ValueError("execution provenance digest mismatch")
        path = target / (key + ".json")
        if path.exists():
            if json.loads(path.read_text()) != event:
                raise ValueError("execution provenance digest mismatch")
        else:
            pending.append((path, event))
    return pending


def merge(template, trial_id, events):
    pending = pending_events(template, trial_id, events)
    directory(template, trial_id).mkdir(parents=True, exist_ok=True)
    for path, event in pending:
        write_json(path, event)


def record(template, trial_id, event, **facts):
    value = {
        **facts,
        "event_id": uuid4().hex,
        "trial_id": trial_id,
        "event": event,
        "timestamp_unix": time.time(),
    }
    merge(template, trial_id, {digest(value): value})


def snapshot(template, trial_id, producer_attempt):
    events = {}
    for path in directory(template, trial_id).glob("*.json"):
        value = json.loads(path.read_text())
        if digest(value) != path.stem:
            raise ValueError("execution provenance digest mismatch")
        events[path.stem] = value
    execution = {
        "trial_id": trial_id,
        "producer_attempt": producer_attempt,
        "events": events,
    }
    validate(execution, trial_id)
    return execution
