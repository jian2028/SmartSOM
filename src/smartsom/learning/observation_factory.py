"""Route new contracts without changing historical implementation digest files."""

from smartsom.config.codec import digest
from smartsom.config.observation_contracts import (
    NETWORK,
    OBSERVATION,
    SCHEMA,
    parameters,
    refs,
)
from smartsom.learning import policy_factory as legacy


def encoder(factory, metadata, *, training=False):
    observation, _ = refs(metadata)
    if observation is None or observation.name != OBSERVATION:
        return legacy.encoder(factory, metadata, training=training)
    from smartsom.config.observation_contracts import validate_factory
    from smartsom.learning.registered_observation import PublicFieldsEncoder

    if observation.version == "2":
        from smartsom.learning.compact_observation import CompactPublicFieldsEncoder

        PublicFieldsEncoder = CompactPublicFieldsEncoder

    validate_factory(factory, observation)
    settings = parameters(observation)
    if settings.rounding != metadata["physical_contract"]["processing_rounding"]:
        raise ValueError("observation nominal rounding differs from physical contract")
    return PublicFieldsEncoder(
        factory,
        metadata["projection"],
        settings=settings,
        provider=metadata["provider"],
        role=metadata["role"],
        training=training,
        **({"algorithm": metadata["algorithm"]} if observation.version == "2" else {}),
    )


def network_class(reference=None):
    from smartsom.config.extensions import ExtensionRef
    from smartsom.config.observation_contracts import implementation_digest

    selected = ExtensionRef.model_validate(reference) if reference is not None else None
    if selected is not None and selected.name == NETWORK:
        if selected.version not in ("1", "2") or selected.parameters:
            raise ValueError("unsupported scoped network contract")
        if (
            selected.code_sha256 is not None
            and selected.code_sha256 != implementation_digest(selected.version)
        ):
            raise ValueError("public field implementation digest changed")
        from smartsom.learning.registered_observation import ScopedCandidateNetwork

        if selected.version == "2":
            from smartsom.learning.compact_observation import CompactCandidateNetwork

            return CompactCandidateNetwork

        return ScopedCandidateNetwork
    return legacy.network_class(reference)


def validate_metadata(metadata):
    observation, _ = refs(metadata)
    if observation is None or observation.name != OBSERVATION:
        return legacy.validate_metadata(metadata)
    schema = SCHEMA if observation.version == "1" else "smartsom.public-fields/v2"
    expected = schema + "/" + digest(observation.parameters)
    if observation.version == "2":
        expected += "/" + metadata["algorithm"]
    if metadata.get("physical_job_encoder") != expected:
        raise ValueError("public field package identity does not match its selector")


def validate_effective_contract(metadata):
    observation, _ = refs(metadata)
    if observation is None or observation.name != OBSERVATION:
        return legacy.validate_effective_contract(metadata)
    from smartsom.learning.production_provenance import validate_declared_contract
    from smartsom.learning.registered_observation import PublicFieldsEncoder

    if observation.version == "2":
        from smartsom.learning.compact_observation import CompactPublicFieldsEncoder

        PublicFieldsEncoder = CompactPublicFieldsEncoder

    validate_metadata(metadata)
    validate_declared_contract(
        metadata, PublicFieldsEncoder, network_class(metadata["network_implementation"])
    )
