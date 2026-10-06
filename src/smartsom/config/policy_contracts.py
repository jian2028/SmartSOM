"""Frozen built-in policy implementations, independent of optional frameworks."""

import hashlib
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.config.extensions import ExtensionModel, ExtensionRef

PHYSICAL_OBSERVATION = "builtin.physical_job"
PHYSICAL_NETWORK = "builtin.physical_job_candidate"
NATIVE_NETWORK = "builtin.candidate"


class PhysicalJobParameters(ExtensionModel):
    include_inspection: bool


class CandidateParameters(ExtensionModel):
    pass


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    files = (
        root / "config/policy_contracts.py",
        root / "learning/physical_job_observation.py",
        root / "learning/production_models.py",
        root / "learning/policy_factory.py",
        root / "engine/production_protocol.py",
    )
    return digest(
        [
            (
                path.name,
                hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            )
            for path in files
        ]
    )


def _pin(reference, names, parameters):
    if reference is None:
        return None
    if reference.name not in names or reference.version != "1":
        raise ValueError("unsupported built-in policy implementation")
    parameters.model_validate(reference.parameters)
    code = implementation_digest()
    if reference.code_sha256 is not None and reference.code_sha256 != code:
        raise ValueError("policy implementation digest changed")
    return reference.model_copy(update={"code_sha256": code})


def pin_contract(observation, network):
    """Pin explicit selectors; return None for existing observation extensions."""
    physical = observation is not None and observation.name == PHYSICAL_OBSERVATION
    if physical:
        observation = _pin(observation, {PHYSICAL_OBSERVATION}, PhysicalJobParameters)
    network = _pin(network, {PHYSICAL_NETWORK, NATIVE_NETWORK}, CandidateParameters)
    if physical != (network is not None and network.name == PHYSICAL_NETWORK):
        raise ValueError(
            "physical-job observation requires its matching masked network"
        )
    return observation, network


def refs(metadata):
    observation = metadata.get("observation")
    network = metadata.get("network_implementation")
    return pin_contract(
        ExtensionRef.model_validate(observation) if observation is not None else None,
        ExtensionRef.model_validate(network) if network is not None else None,
    )


def validate_factory(factory, observation):
    if observation is None or observation.name != PHYSICAL_OBSERVATION:
        return
    inputs = [buffer for buffer in factory.buffers if buffer.role == "system_input"]
    if len(inputs) != 1:
        raise ValueError("physical-job observation requires one finite system input")
    storage = inputs[0].storage
    capacity = (
        sum(slot.capacity for slot in storage.slots)
        if hasattr(storage, "slots")
        else storage.capacity
    )
    if capacity != {8: 10, 16: 20}.get(len(factory.machines)):
        raise ValueError(
            "physical-job observation requires its configured finite input"
        )
    for buffer in factory.buffers:
        if buffer.role not in ("system_input", "machine_pre", "machine_post"):
            continue
        storage = buffer.storage
        values = (
            [slot.capacity for slot in storage.slots]
            if hasattr(storage, "slots")
            else [storage.capacity]
        )
        if any(value is None for value in values):
            raise ValueError(
                "physical-job occupancy requires finite physical capacities"
            )
