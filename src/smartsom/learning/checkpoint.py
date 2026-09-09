"""Verified inference bundles and semantic policies; optional imports stay lazy."""

import hashlib
import importlib.metadata
import json
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
from smartsom.config.extensions import ExtensionSpec
from smartsom.config.models import (
    SHA256,
    AlgorithmFile,
    LearningAlgorithm,
    PPOParameters,
    ResourcePPOParameters,
    StrictModel,
)
from smartsom.dispatch import DecisionContext
from smartsom.engine import DeadlockError, SimulationResult
from smartsom.learning.episode import central_outcome
from smartsom.learning.extension_evidence import decision_record, reward_record
from smartsom.learning.extensions import (
    EncodedDecision,
    ExtensionsRuntime,
    RewardTransition,
    bind_extensions,
    central_observation,
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
    extensions: ExtensionSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    extension_state: CheckpointFile | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
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
    try:
        pinned = bind_extensions(manifest.extensions, spec.provider)
        if manifest.extensions != pinned or pinned != bind_extensions(
            spec.extensions, spec.provider
        ):
            raise ValueError("checkpoint extensions are unpinned or incompatible")
        if (pinned is None) != (manifest.extension_state is None):
            raise ValueError("checkpoint extension state coverage mismatch")
        if (
            manifest.extension_state is not None
            and manifest.extension_state not in manifest.files
        ):
            raise ValueError("checkpoint extension state is not in the file inventory")
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
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

    def __init__(
        self,
        resolved,
        *,
        deterministic=True,
        predictor=None,
        on_extension=None,
        on_reward=None,
    ):
        self.manifest = validate_checkpoint(resolved)
        spec = resolved.algorithm.algorithm
        self.checkpoint_sha256 = spec.checkpoint_sha256
        self.on_extension = on_extension
        self.on_reward = on_reward
        self.extension_pending = None
        self.extension_finished = False
        self.total_reward = 0.0
        self.rewarded_tick = 0
        self.projection = LearningProjection(resolved.factory, spec.projection)
        self.extensions = None
        if spec.extensions is not None:
            prototype = DecisionContext(0, (), (), ())
            layout = central_observation(
                prototype, self.projection.project(prototype)
            ).layout()
            self.extensions = ExtensionsRuntime(
                spec.extensions, spec.provider, {None: layout}
            )
            state = json.loads(
                (Path(spec.checkpoint) / self.manifest.extension_state.path).read_text()
            )
            self.extensions.load_state_dict(state)
            self.extensions.begin_episode()
        self.limits = resolved.run.budget.limits()
        self.decisions = 0
        if predictor is None:
            if spec.provider == "rllib.ppo":
                from smartsom.learning.rllib import load_predictor
            else:
                from smartsom.learning.sb3 import load_predictor
            predictor = (
                load_predictor(Path(spec.checkpoint))
                if deterministic
                else load_predictor(
                    Path(spec.checkpoint),
                    deterministic=False,
                    seed=next(
                        s.value for s in resolved.seeds if s.domain == "algorithm"
                    ),
                )
            )
        self.predict = predictor

    def select_action(self, context):
        if self.extensions:
            self.extension_pending = (context, ())
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
        state_before_sha256 = (
            digest(self.extensions.state_dict()) if self.extensions else None
        )
        encoded = (
            EncodedDecision(
                view, self.extensions.encode(central_observation(context, view))
            )
            if self.extensions
            else view
        )
        index = self.predict(encoded)
        action = view.decode(index)
        if self.extensions:
            self.extension_pending = (context, (action,))
        if self.extensions and self.on_extension:
            self.on_extension(
                decision_record(
                    decision_index=self.decisions,
                    context=context,
                    checkpoint_sha256=self.checkpoint_sha256,
                    runtime=self.extensions,
                    state_before_sha256=state_before_sha256,
                    views=(encoded,),
                    indices=(int(index),),
                )
            )
        self.decisions += 1
        return action

    def check_outcome(self, outcome):
        tick = getattr(outcome, "simulation_time", getattr(outcome, "makespan", None))
        if self.extensions:
            reason = "completed" if isinstance(outcome, SimulationResult) else None
            if reason is None and not any(self.projection.project(outcome).action_mask):
                reason = "policy_stalled"
            reason = self._extension_reward(
                tick, reason, None if isinstance(outcome, SimulationResult) else outcome
            )
            if reason in ("policy_stalled", "budget_exhausted"):
                raise RuntimeError(
                    f"{reason}: checkpoint evaluation episode limit or stalled policy"
                )
            return
        if tick > self.limits.max_ticks:
            raise RuntimeError(f"budget_exhausted: actual evaluation tick {tick}")

    def _extension_reward(self, tick, reason, after=None):
        before, actions = self.extension_pending or (None, ())
        reason, raw, _, _ = central_outcome(
            tick=tick,
            decisions=self.decisions,
            reason=reason,
            limits=self.limits,
            rewarded_tick=self.rewarded_tick,
            total_reward=self.total_reward,
        )
        transition = RewardTransition(before, after, actions, raw, tick, reason)
        state_before = digest(self.extensions.state_dict())
        values = self.extensions.reward(transition)
        self.total_reward += raw
        self.rewarded_tick = tick
        self.extension_pending = None
        self.extension_finished = reason is not None
        if self.on_reward:
            self.on_reward(
                reward_record(
                    decision_index=self.decisions - 1 if actions else None,
                    checkpoint_sha256=self.checkpoint_sha256,
                    runtime=self.extensions,
                    state_before_sha256=state_before,
                    transition=transition,
                    values=values,
                )
            )
        return reason

    def fail(self, exc, *, tick, trace_end=None):
        """Public failure notification, including initialization with no context."""
        if not self.extensions or self.extension_finished:
            return
        reason = (
            "deadlock"
            if isinstance(exc, DeadlockError)
            else "policy_stalled"
            if "policy_stalled" in str(exc)
            else "budget_exhausted"
            if "budget_exhausted" in str(exc)
            else "execution_failed"
        )
        self._extension_reward(tick, reason)
