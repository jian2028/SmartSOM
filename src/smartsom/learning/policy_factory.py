"""Build declared policies identically in drivers, spawn workers and learners."""

from smartsom.config.codec import primitive
from smartsom.config.policy_contracts import PHYSICAL_OBSERVATION, refs


def encoder(factory, metadata, *, training=False):
    observation, _ = refs(metadata)
    from smartsom.learning.production_models import PublicEncoder

    kwargs = dict(
        provider=metadata["provider"], role=metadata["role"], training=training
    )
    if observation is not None and observation.name == PHYSICAL_OBSERVATION:
        from smartsom.learning.physical_job_observation import PhysicalJobEncoder

        return PhysicalJobEncoder(
            factory,
            metadata["projection"],
            **kwargs,
            include_inspection=observation.parameters["include_inspection"],
        )
    return PublicEncoder(
        factory, metadata["projection"], observation=primitive(observation), **kwargs
    )


def network_class(reference=None):
    from smartsom.config.extensions import ExtensionRef
    from smartsom.config.policy_contracts import PHYSICAL_NETWORK, pin_contract
    from smartsom.learning.production_models import CandidateNetwork

    if reference is None:
        return CandidateNetwork
    reference = ExtensionRef.model_validate(reference)
    if reference.name == PHYSICAL_NETWORK:
        observation = ExtensionRef(
            name=PHYSICAL_OBSERVATION,
            version="1",
            parameters={"include_inspection": True},
        )
        pin_contract(observation, reference)
        from smartsom.learning.physical_job_observation import PhysicalJobNetwork

        return PhysicalJobNetwork
    pin_contract(None, reference)
    return CandidateNetwork


def validate_metadata(metadata):
    observation, _ = refs(metadata)
    if observation is not None and observation.name == PHYSICAL_OBSERVATION:
        from smartsom.learning.physical_job_observation import SCHEMA

        expected = (
            SCHEMA + "/inspection=" + str(observation.parameters["include_inspection"])
        )
        if metadata.get("physical_job_encoder") != expected:
            raise ValueError(
                "physical-job package identity does not match its selector"
            )
