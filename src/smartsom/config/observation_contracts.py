"""Opt-in registered field contracts; historical policy pinning is unchanged."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.extensions import ExtensionModel, ExtensionRef
from smartsom.config.policy_contracts import pin_contract as legacy_pin

OBSERVATION = "builtin.public_fields"
NETWORK = "builtin.scoped_candidate"
SCHEMA = "smartsom.public-fields/v2"

# Explicit Python registrations, not executable expressions or import paths.
# Values declare domain, permitted normalization, and whether rows aggregate.
COMPONENTS = {
    "decision": ("context", ("projection", "raw"), False),
    "self_target": ("context", ("raw",), False),
    "local_map": ("context", ("raw",), False),
    "vehicles": ("context", ("projection", "raw"), True),
    "machines": ("context", ("projection", "raw"), True),
    "inventory": ("context", ("capacity", "projection", "raw"), True),
    "topology": ("context", ("raw",), False),
    "work_summary": ("context", ("projection", "raw"), False),
    "job": ("candidate", ("projection", "raw"), False),
    "semantics": ("candidate", ("projection", "raw"), False),
    "machine_mode": ("candidate", ("projection", "raw"), False),
    "dispatch_target": ("candidate", ("capacity",), False),
    "cargo": ("candidate", ("projection", "raw"), False),
    "movement": ("candidate", ("projection", "raw"), False),
}


class FieldComponent(ExtensionModel):
    name: str
    normalization: Literal["raw", "projection", "capacity"]
    aggregation: Literal["per_entity", "mean", "max", "sum"]
    visibility: Literal["actor", "critic", "both"]
    applies_to: (
        tuple[Literal["machine", "buffer", "dispatcher", "mover"], ...] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)


class PublicFieldsParameters(ExtensionModel):
    rounding: Literal["ceil", "half_up"]
    include_inspection: bool
    context: Annotated[tuple[FieldComponent, ...], Field(min_length=1)]
    candidates: Annotated[tuple[FieldComponent, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def registered_components(self):
        for domain, selections in (
            ("context", self.context),
            ("candidate", self.candidates),
        ):
            names = [selection.name for selection in selections]
            if len(names) != len(set(names)):
                raise ValueError("duplicate observation field component")
            for selection in selections:
                if selection.applies_to is not None:
                    if not selection.applies_to or len(
                        set(selection.applies_to)
                    ) != len(selection.applies_to):
                        raise ValueError(
                            "component applicable roles must be nonempty and unique"
                        )
                    if selection.name == "local_map" and set(selection.applies_to) - {
                        "dispatcher",
                        "mover",
                    }:
                        raise ValueError(
                            "local_map applies only to Dispatcher/Mover owners"
                        )
                registration = COMPONENTS.get(selection.name)
                if registration is None or registration[0] != domain:
                    raise ValueError("unregistered observation field component")
                if selection.normalization not in registration[1]:
                    raise ValueError("unsupported component normalization")
                if selection.aggregation != "per_entity" and not registration[2]:
                    raise ValueError("component does not support aggregation")
                if domain == "candidate" and selection.visibility != "actor":
                    raise ValueError(
                        "candidate and conditional-prefix fields are actor-only"
                    )
                if (
                    selection.name == "work_summary"
                    and selection.visibility != "critic"
                ):
                    raise ValueError("task work summary remains critic-only")
        return self


def parameters(reference):
    return PublicFieldsParameters.model_validate_json(
        canonical_json(reference.parameters)
    )


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    files = (
        "config/observation_contracts.py",
        "learning/observation_factory.py",
        "learning/registered_observation.py",
        "learning/production_models.py",
        "learning/physical_job_observation.py",
    )
    files += (
        "config/extensions.py",
        "learning/production_inference.py",
        "learning/production_rllib_v3.py",
        "learning/production_sb3_v3.py",
        "learning/production_provenance.py",
    )
    return digest(
        [
            (
                name,
                hashlib.sha256(
                    (root / name).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
            for name in files
        ]
    )


def pin_contract(observation, network):
    selected = observation is not None and observation.name == OBSERVATION
    paired = network is not None and network.name == NETWORK
    if not selected and not paired:
        return legacy_pin(observation, network)
    if (
        not selected
        or not paired
        or observation.version != "compact"
        or network.version != observation.version
    ):
        raise ValueError(
            "public fields require their matching current scoped network; remove legacy version selectors and initialize fresh models"
        )
    frozen = parameters(observation)
    if network.parameters:
        raise ValueError("scoped network has no implementation parameters")
    code = implementation_digest()
    for reference in (observation, network):
        if reference.code_sha256 is not None and reference.code_sha256 != code:
            raise ValueError("public field implementation digest changed")
    return (
        observation.model_copy(
            update={"parameters": primitive(frozen), "code_sha256": code}
        ),
        network.model_copy(update={"code_sha256": code}),
    )


def refs(metadata):
    return pin_contract(
        *[
            ExtensionRef.model_validate(metadata[key])
            if metadata.get(key) is not None
            else None
            for key in ("observation", "network_implementation")
        ]
    )


def validate_factory(factory, observation, *, scenario=None):
    from smartsom.config.policy_contracts import validate_factory as legacy_validate

    if observation is None or observation.name != OBSERVATION:
        return legacy_validate(factory, observation)
    if (
        scenario is not None
        and parameters(observation).rounding != scenario.processing_rounding
    ):
        raise ValueError("observation nominal rounding differs from physical contract")
    # presence+PASS is sufficient only when FAIL cannot become actionable cargo.
    if any(
        station.auto_disposal_bin_id is None for station in factory.inspection_stations
    ):
        raise ValueError(
            "presence/PASS observations require automatic disposal at every inspection station"
        )


def is_full_observation(reference):
    from smartsom.config.policy_contracts import PHYSICAL_OBSERVATION

    return reference is not None and reference.name in (
        PHYSICAL_OBSERVATION,
        OBSERVATION,
    )
