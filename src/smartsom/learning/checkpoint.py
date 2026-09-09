"""Verified inference bundles and semantic policies; optional imports stay lazy."""

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from smartsom.config.codec import (
    ConfigurationError,
    digest,
    normalize_factory,
    normalize_workload,
    read_model,
)
from smartsom.config.models import (
    SHA256,
    AlgorithmFile,
    LearningAlgorithm,
    PPOParameters,
    ResourcePPOParameters,
    StrictModel,
)
from smartsom.learning.joint import COORDINATION_VERSION
from smartsom.learning.projection import (
    LearningProjection,
    ProjectionSpec,
    validate_capacity,
)
from smartsom.learning.resources import ResourceProjection, ResourceProjectionSpec

VERSIONS = {
    "gymnasium": "1.2.2",
    "torch": "2.14.0",
    "ray": "2.58.0",
    "stable-baselines3": "2.9.0",
    "sb3-contrib": "2.9.0",
    "pettingzoo": "1.27.0",
}


def require_backend(provider: str) -> dict[str, str]:
    packages = (
        ("gymnasium", "torch", "ray", "pettingzoo")
        if provider == "rllib.resource_ppo"
        else ("gymnasium", "torch", "ray")
        if provider == "rllib.ppo"
        else ("gymnasium", "torch", "stable-baselines3", "sb3-contrib")
        if provider == "sb3.maskable_ppo"
        else ()
    )
    if not packages:
        raise ConfigurationError(f"unsupported learning provider {provider!r}")
    extra = (
        "learning-marl"
        if provider == "rllib.resource_ppo"
        else "learning-rllib"
        if provider == "rllib.ppo"
        else "learning-sb3"
    )
    actual = {}
    for package in packages:
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ConfigurationError(
                f"{provider} requires uv sync --locked --extra {extra}"
            ) from exc
        if actual[package].split("+")[0] != VERSIONS[package]:
            raise ConfigurationError(
                f"{package} requires locked version {VERSIONS[package]}, found {actual[package]}"
            )
    return actual


class CheckpointFile(StrictModel):
    path: str
    sha256: SHA256


class CheckpointManifest(StrictModel):
    schema_id: Literal["smartsom.checkpoint/v1"] = Field(alias="schema")
    provider: Literal["rllib.ppo", "sb3.maskable_ppo"]
    projection: ProjectionSpec
    parameters: PPOParameters
    structure_sha256: SHA256
    files: tuple[CheckpointFile, ...]
    dependencies: tuple[tuple[str, str], ...]
    initial_weights_sha256: SHA256
    final_weights_sha256: SHA256
    environment_steps: Annotated[int, Field(gt=0)]
    learner_updates: Annotated[int, Field(gt=0)]
    framework_seed: Annotated[int, Field(ge=0, lt=2**31)]


class RoleWeights(StrictModel):
    role: Literal["machine_policy", "agv_policy"]
    initial_sha256: SHA256
    final_sha256: SHA256


class ResourceCheckpointManifest(CheckpointManifest):
    schema_id: Literal["smartsom.resource-checkpoint/v1"] = Field(alias="schema")
    provider: Literal["rllib.resource_ppo"]
    projection: ResourceProjectionSpec
    parameters: ResourcePPOParameters
    coordination_version: Literal["smartsom.resource-coordination/v1"] = (
        COORDINATION_VERSION
    )
    role_mapping: tuple[tuple[str, str], ...]
    role_weights: tuple[RoleWeights, ...]
    agent_steps: Annotated[int, Field(gt=0)]
    physical_actions: Annotated[int, Field(gt=0)]


def structural_identity(
    resolved, projection: ProjectionSpec | ResourceProjectionSpec
) -> str:
    identity = {
        "version": "smartsom.learning-structure/v1",
        "factory": resolved.factory,
        "workload": resolved.workload,
        "projection": projection,
        "modules": {
            field: getattr(resolved.scenario, field) is not None
            for field in (
                "arrivals",
                "machine_events",
                "processing_time",
                "transport",
                "buffers",
                "quality",
                "holding_buffer",
            )
        },
        "decision_trigger": resolved.scenario.decision_trigger,
        "probability_visibility": resolved.scenario.quality.probability_visibility
        if resolved.scenario.quality
        else None,
    }
    if isinstance(projection, ResourceProjectionSpec):
        identity["version"] = "smartsom.resource-structure/v1"
        identity["factory"] = normalize_factory(resolved.factory)
        identity["workload"] = normalize_workload(resolved.workload)
        identity["coordination"] = COORDINATION_VERSION
        p = ResourceProjection(
            resolved.factory, projection, transport_enabled=resolved.transport_enabled
        )
        identity["agents"] = p.agents
    return digest(identity)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_checkpoint_reference(
    algorithm: AlgorithmFile, owner: Path
) -> AlgorithmFile:
    spec = algorithm.algorithm
    if not isinstance(spec, LearningAlgorithm):
        return algorithm
    if spec.checkpoint is None:
        raise ConfigurationError(
            "run/study requires an existing checkpoint; use train to create one"
        )
    path = (owner.parent / spec.checkpoint).resolve()
    try:
        sha = file_hash(path / "checkpoint.json")
    except OSError as exc:
        raise ConfigurationError(f"cannot read checkpoint: {exc}") from exc
    if spec.checkpoint_sha256 is not None and spec.checkpoint_sha256 != sha:
        raise ConfigurationError("checkpoint manifest digest mismatch")
    return algorithm.model_copy(
        update={
            "algorithm": spec.model_copy(
                update={"checkpoint": str(path), "checkpoint_sha256": sha}
            )
        }
    )


def validate_checkpoint(resolved) -> CheckpointManifest | None:
    spec = resolved.algorithm.algorithm
    if not isinstance(spec, LearningAlgorithm):
        return None
    actual_dependencies = require_backend(spec.provider)
    validate_capacity(spec.projection, resolved.workload, resolved.quality)
    if spec.checkpoint is None or spec.checkpoint_sha256 is None:
        raise ConfigurationError("evaluation needs a resolved checkpoint identity")
    root = Path(spec.checkpoint).resolve()
    manifest_type = (
        ResourceCheckpointManifest
        if spec.provider == "rllib.resource_ppo"
        else CheckpointManifest
    )
    manifest, sha = read_model(root / "checkpoint.json", manifest_type)
    if any(
        actual_dependencies.get(name) != version
        for name, version in manifest.dependencies
    ):
        raise ConfigurationError("checkpoint dependency versions are incompatible")
    if sha != spec.checkpoint_sha256:
        raise ConfigurationError("checkpoint manifest digest mismatch")
    if (
        manifest.provider,
        manifest.projection,
        manifest.parameters,
        manifest.structure_sha256,
    ) != (
        spec.provider,
        spec.projection,
        spec.parameters,
        structural_identity(resolved, spec.projection),
    ):
        raise ConfigurationError(
            "checkpoint input structure or projection is incompatible"
        )
    if not manifest.files or len({f.path for f in manifest.files}) != len(
        manifest.files
    ):
        raise ConfigurationError("invalid checkpoint file inventory")
    for entry in manifest.files:
        path = (root / entry.path).resolve()
        if (
            not path.is_relative_to(root)
            or path == root
            or Path(entry.path).is_absolute()
        ):
            raise ConfigurationError("invalid checkpoint member path")
        try:
            if file_hash(path) != entry.sha256:
                raise ConfigurationError(
                    f"checkpoint file digest mismatch: {entry.path}"
                )
        except OSError as exc:
            raise ConfigurationError(
                f"checkpoint file unavailable: {entry.path}"
            ) from exc
    if (
        manifest.initial_weights_sha256 == manifest.final_weights_sha256
        or manifest.learner_updates <= 0
    ):
        raise ConfigurationError("checkpoint has no verified parameter update")
    if isinstance(manifest, ResourceCheckpointManifest):
        p = ResourceProjection(
            resolved.factory,
            spec.projection,
            transport_enabled=resolved.transport_enabled,
        )
        mapping = tuple(
            (a, "machine_policy" if a.startswith("machine:") else "agv_policy")
            for a in p.agents
        )
        roles = {role for _, role in mapping}
        if (
            manifest.role_mapping != mapping
            or {w.role for w in manifest.role_weights} != roles
            or len(manifest.role_weights) != len(roles)
            or any(w.initial_sha256 == w.final_sha256 for w in manifest.role_weights)
            or manifest.agent_steps != manifest.environment_steps * len(mapping)
            or digest({w.role: w.initial_sha256 for w in manifest.role_weights})
            != manifest.initial_weights_sha256
            or digest({w.role: w.final_sha256 for w in manifest.role_weights})
            != manifest.final_weights_sha256
        ):
            raise ConfigurationError("resource checkpoint role/update/count mismatch")
    return manifest


class CheckpointPolicy:
    """Inference has the existing OnlinePolicy contract and never advances physics."""

    def __init__(self, resolved):
        self.manifest = validate_checkpoint(resolved)
        spec = resolved.algorithm.algorithm
        self.projection = LearningProjection(resolved.factory, spec.projection)
        self.limits = resolved.run.budget.limits()
        self.decisions = 0
        if spec.provider == "rllib.ppo":
            from smartsom.learning.rllib import load_predictor
        else:
            from smartsom.learning.sb3 import load_predictor
        self.predict = load_predictor(Path(spec.checkpoint))

    def select_action(self, context):
        if (
            self.decisions >= self.limits.max_decisions
            or context.simulation_time >= self.limits.max_ticks
        ):
            raise RuntimeError("budget_exhausted: checkpoint evaluation episode limit")
        view = self.projection.project(context)
        if not any(view.action_mask):
            raise RuntimeError(
                "policy_stalled: no publicly representable learning action"
            )
        index = self.predict(view)
        action = view.decode(index)
        self.decisions += 1
        return action

    def check_outcome(self, outcome):
        tick = getattr(outcome, "simulation_time", getattr(outcome, "makespan", None))
        if tick > self.limits.max_ticks:
            raise RuntimeError(f"budget_exhausted: actual evaluation tick {tick}")
