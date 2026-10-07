"""Portable model semantics and explicit caller provenance; no framework imports."""

import copy
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from smartsom.config.codec import digest

_CALLER = ContextVar("smartsom_execution_provenance", default=None)


def class_name(cls):
    return cls.__module__ + "." + cls.__qualname__


def model_contract(encoder, network, metadata):
    return {
        "schema": "smartsom.effective-model/v1",
        "encoder_class": class_name(type(encoder)),
        "encoder_semantics": copy.deepcopy(encoder.DIAGNOSTIC_CONTRACT),
        "physical_job_identity": getattr(encoder, "observation_identity", None),
        "context_size": encoder.context_size,
        "candidate_width": encoder.candidate_width,
        "observation_extension": copy.deepcopy(metadata.get("observation")),
        "network_class": class_name(type(network)),
        "network_semantics": copy.deepcopy(network.DIAGNOSTIC_CONTRACT),
        "branches": copy.deepcopy(metadata["network"]),
        "algorithm": metadata["algorithm"],
        "projection": copy.deepcopy(metadata["projection"]),
        "central": network.central,
        "central_private_end": network.central_private_end,
        "prefix": {
            "kind": "GRU",
            "width": network.actor_prefix.hidden_size,
            "input_width": network.actor_prefix.input_size,
            "scope": "conditional decision prefix, not recurrent state across ticks",
            "critic_prefix": "zeroed",
        },
    }


def validate_declared_contract(metadata, encoder_type, network_type):
    saved = metadata.get("effective_model_contract")
    if saved is None:
        return  # Historical packages retain their existing compatibility checks.
    if metadata.get("effective_model_sha256") != digest(saved):
        raise ValueError("effective model contract hash mismatch")
    for key, expected in (
        ("schema", "smartsom.effective-model/v1"),
        ("encoder_class", class_name(encoder_type)),
        ("encoder_semantics", encoder_type.DIAGNOSTIC_CONTRACT),
        ("network_class", class_name(network_type)),
        ("network_semantics", network_type.DIAGNOSTIC_CONTRACT),
    ):
        if saved.get(key) != expected:
            raise ValueError("effective model contract mismatch: " + key)


@contextmanager
def execution_provenance(name, files):
    """Thin harness integration: caller-chosen portable labels map to source files.

    Wrap native train/evaluate calls. This never installs an encoder or changes
    configuration. Frozen historical manifests must not be rewritten to add it.
    """
    hashes = {}
    for label, path in sorted(files.items()):
        if not label or "\\" in label or label.startswith("/") or ":" in label:
            raise ValueError("provenance labels must be portable relative names")
        if ".." in label.split("/"):
            raise ValueError("provenance label cannot escape its namespace")
        hashes[label] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    caller = {"name": name, "files": hashes}
    token = _CALLER.set({**caller, "sha256": digest(caller)})
    try:
        yield
    finally:
        _CALLER.reset(token)


def frozen_caller_json():
    from smartsom.config.codec import canonical_json

    return canonical_json(_CALLER.get())


def execution_identity(prepared):
    import json

    from smartsom.config.diagnostics import capture_identity

    composition = json.loads(prepared.composition_json)
    config = json.loads(getattr(prepared, "config_json", "{}"))
    frozen = json.loads(getattr(prepared, "execution_provenance_json", "null"))
    caller = frozen if frozen is not None else _CALLER.get()
    return {
        "schema": "smartsom.execution-provenance/v1",
        "caller": copy.deepcopy(caller),
        "caller_status": "declared" if caller else "not_declared",
        "diagnostics": capture_identity(config.get("diagnostics")),
        "prepared_matching": composition["matching"]["name"],
        "effective_matching": "first_arrival",
        "note": "source identity and scientific inputs are recorded separately",
    }
