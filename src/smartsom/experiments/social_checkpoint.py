"""Durable episode transactions for the isolated social-learning driver."""

import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    """Normalize JSON sequences before comparing persisted and Torch payload rows."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read(path):
    return json.loads(Path(path).read_text())


def save(policy, path, state):
    import torch

    path = Path(path)
    scratch = path.with_name(f".state-{os.getpid()}.pt")
    policy.save(scratch)
    payload = torch.load(scratch, map_location="cpu", weights_only=False)
    payload["long_state"] = canonical(state)
    payload["torch_rng"] = torch.get_rng_state()
    torch.save(payload, scratch)
    with scratch.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(scratch, path)
    sync_directory(path.parent)


def restore(policy, path):
    import torch

    policy.load(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(payload["torch_rng"])
    state = payload["long_state"]
    if state["updates"] != policy.updates:
        raise ValueError("checkpoint update counter mismatch")
    return state


def immutable_copy(source, target):
    """Publish identical bytes once; never replace an existing scientific snapshot."""
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"immutable checkpoint differs: {target}")
        return
    scratch = target.with_name(target.name + f".{os.getpid()}.tmp")
    atomic_bytes(scratch, source.read_bytes())
    try:
        os.link(scratch, target)
        sync_directory(target.parent)
    except FileExistsError:
        if digest(source) != digest(target):
            raise ValueError(f"immutable checkpoint differs: {target}") from None
    finally:
        scratch.unlink()


def reconcile(directory, state, threshold):
    """Finish the last transaction BEFORE permitting another episode update.

    Only a trailing incomplete JSON record can be repaired. All older rows must
    be consecutive and the last complete row must equal the checkpoint authority.
    """
    path = directory / "last.pt"
    episode = state["episodes"]
    row = canonical(state.get("last_episode"))
    ledger = directory / "episodes.jsonl"
    raw = ledger.read_bytes() if ledger.exists() else b""
    complete = raw[: raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines()]
    if [r["episodes"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("nonconsecutive episode ledger")
    if len(rows) not in ({0} if episode == 0 else {episode - 1, episode}):
        raise ValueError("ledger cannot be recovered from authoritative last row")
    if episode:
        if row is None or row["episodes"] != episode:
            raise ValueError("missing authoritative episode row")
        if len(rows) == episode:
            if canonical(rows[-1]) != row:
                raise ValueError("checkpoint/ledger last row mismatch")
        else:
            complete += (json.dumps(row, allow_nan=False) + "\n").encode()
    if complete != raw:
        atomic_bytes(ledger, complete)
    early = state.get("early")
    if state["physical_ticks"] >= threshold and early is None:
        raise ValueError("missing first-crossing identity; refusing later substitute")
    if early:
        target = directory / "early-160k.pt"
        if early["episodes"] == episode:
            immutable_copy(path, target)
        elif not target.exists():
            raise ValueError(
                "first-crossing checkpoint missing; refusing later substitute"
            )
        receipt_path = directory / "early-160k.json"
        receipt = {**early, "sha256": digest(target)}
        if receipt_path.exists() and read(receipt_path) != receipt:
            raise ValueError("early checkpoint receipt mismatch")
        write(receipt_path, receipt)
    if episode % 50 == 0:
        immutable_copy(path, directory / f"ep-{episode:04d}.pt")
    write(
        directory / "state.json",
        {k: v for k, v in state.items() if k != "last_episode"},
    )
