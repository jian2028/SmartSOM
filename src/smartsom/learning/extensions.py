"""Research hooks over public inputs, with explicit identities and resumable state.

No simulator, complete workload, event calendar, NumPy or framework is imported.
Registration is an explicit Python API; configuration never imports executable code.
"""

import copy
import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from math import isfinite, prod
from numbers import Real
from pathlib import Path
from typing import Callable, Literal

from smartsom.config.extensions import ExtensionRef, ExtensionSpec
from smartsom.dispatch import DecisionContext, SemanticAction

BACKENDS = ("rllib.ppo", "sb3.maskable_ppo", "rllib.resource_ppo")
type ComponentKind = Literal["observation", "reward", "torch_encoder"]


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _tensor(value):
    if isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
        if not isfinite(value):
            raise ValueError("extension observation/reward must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_tensor(child) for child in value)
    if callable(getattr(value, "tolist", None)):
        return _tensor(value.tolist())
    raise ValueError("extension observation must contain only numeric tensor values")


def _shape(value):
    if isinstance(value, float):
        return ()
    shapes = {_shape(child) for child in value}
    if len(shapes) != 1:
        raise ValueError("extension tensors must be nonempty and rectangular")
    return (len(value), *next(iter(shapes)))


@dataclass(frozen=True, slots=True)
class ObservationSpace:
    """A fixed vector or a fixed set of named, fixed-shape float tensors."""

    vector: tuple[int, ...] | None = None
    fields: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        if (self.vector is None) == (not self.fields):
            raise ValueError("observation space must be exactly a vector or a Dict")
        shapes = (
            (self.vector,) if self.vector else tuple(shape for _, shape in self.fields)
        )
        if any(
            not shape or any(type(size) is not int or size < 1 for size in shape)
            for shape in shapes
        ):
            raise ValueError("extension observation shapes must be fixed and positive")
        names = [name for name, _ in self.fields]
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError(
                "extension observation Dict keys must be unique nonempty strings"
            )

    @property
    def flat_size(self):
        return (
            prod(self.vector)
            if self.vector
            else sum(prod(shape) for _, shape in self.fields)
        )

    def validate(self, value):
        if self.vector:
            actual = _tensor(value)
            if _shape(actual) != self.vector:
                raise ValueError("extension changed its declared observation shape")
            return actual
        if not isinstance(value, dict) or set(value) != {
            name for name, _ in self.fields
        }:
            raise ValueError("extension changed its declared observation Dict keys")
        result = {}
        for name, shape in self.fields:
            result[name] = _tensor(value[name])
            if _shape(result[name]) != shape:
                raise ValueError(f"extension changed observation field {name!r} shape")
        return result

    def zeros(self):
        def tensor(shape):
            return tuple(tensor(shape[1:]) for _ in range(shape[0])) if shape else 0.0

        return (
            tensor(self.vector)
            if self.vector
            else {name: tensor(shape) for name, shape in self.fields}
        )


@dataclass(frozen=True, slots=True)
class PublicObservation:
    context: DecisionContext
    vector: tuple[float, ...]
    groups: tuple[tuple[str, tuple], ...]
    role: str | None = None
    agent_id: str | None = None

    def layout(self):
        return ObservationLayout(
            ObservationSpace(vector=(len(self.vector),)),
            ObservationSpace(
                fields=tuple(
                    (name, _shape(_tensor(value))) for name, value in self.groups
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    vector: ObservationSpace
    grouped: ObservationSpace


def central_observation(context, mapping):
    return PublicObservation(
        context, mapping.observations, (("state", mapping.observations),)
    )


def resource_observation(context, view):
    return PublicObservation(
        context,
        view.observations,
        (
            ("global", view.global_features),
            ("local", view.local_features),
            ("candidates", view.candidate_features),
        ),
        view.role,
        view.agent_id,
    )


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    """Only observations change; semantic decoding delegates to the frozen mapper."""

    mapping: object
    observations: object

    @property
    def action_mask(self):
        return self.mapping.action_mask

    @property
    def actions(self):
        return self.mapping.actions

    @property
    def bindings(self):
        return self.mapping.bindings

    @property
    def role(self):
        return self.mapping.role

    @property
    def agent_id(self):
        return self.mapping.agent_id

    def decode(self, index):
        return self.mapping.decode(index)


@dataclass(frozen=True, slots=True)
class EncodedResourceDecision:
    mapping: object
    views: tuple[EncodedDecision, ...]

    @property
    def context(self):
        return self.mapping.context

    @property
    def bindings(self):
        return self.mapping.bindings

    def for_agent(self, agent_id):
        return next(view for view in self.views if view.agent_id == agent_id)


@dataclass(frozen=True, slots=True)
class RewardTransition:
    """Only already-public decisions, semantic actions and the current outcome."""

    before: DecisionContext | None
    after: DecisionContext | None
    actions: tuple[SemanticAction, ...]
    raw_reward: float
    simulation_time: int
    reason: str | None
    role: str | None = None
    agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class RewardValues:
    raw: float
    research: float
    learner: float


@dataclass(frozen=True, slots=True)
class RewardBatch:
    team: RewardValues
    roles: tuple[tuple[str, RewardValues], ...]


@dataclass(frozen=True, slots=True)
class Registration:
    kind: ComponentKind
    name: str
    version: str
    factory: Callable
    backends: tuple[str, ...]
    stateful: bool
    sources: tuple[Path, ...]
    loaded_sha256: str

    @property
    def code_sha256(self):
        return self.loaded_sha256

    def verify_sources(self):
        if self.source_digest() != self.loaded_sha256:
            raise ValueError(f"extension {self.name} source changed after registration")

    def source_digest(self):
        return _digest(
            [
                (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in self.sources
            ]
        )

    def identity(self):
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "code_sha256": self.code_sha256,
            "supported_backends": list(self.backends),
            "stateful": self.stateful,
        }


_REGISTRY: dict[tuple[str, str, str], Registration] = {}


def register_extension(
    kind: ComponentKind,
    name: str,
    version: str,
    factory: Callable,
    *,
    supported_backends=BACKENDS,
    stateful=False,
    source_files=(),
):
    """Register inspected Python implementations explicitly, without an import path.

    Include additional implementation files in source_files when a factory delegates
    to them. Dependency versions remain part of the framework checkpoint identity.
    """
    ExtensionRef(name=name, version=version)
    if kind not in ("observation", "reward", "torch_encoder") or not callable(factory):
        raise ValueError("invalid extension kind or factory")
    backends = tuple(supported_backends)
    if (
        not backends
        or len(set(backends)) != len(backends)
        or set(backends) - set(BACKENDS)
    ):
        raise ValueError("extension must declare supported SmartSOM backends")
    if type(stateful) is not bool:
        raise ValueError("extension stateful declaration must be boolean")
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError("extension factory must have inspectable Python source")
    files = tuple(
        dict.fromkeys(Path(path).resolve() for path in (source, *source_files))
    )
    if any(not path.is_file() for path in files):
        raise ValueError("extension implementation source is unavailable")
    key = (kind, name, version)
    source_hash = _digest(
        [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    )
    entry = Registration(
        kind, name, version, factory, backends, stateful, files, source_hash
    )
    if key in _REGISTRY:
        if _REGISTRY[key] != entry:
            raise ValueError(f"extension {name}@{version} is already registered")
        return _REGISTRY[key]
    _REGISTRY[key] = entry
    return entry


def registration(
    kind: ComponentKind, reference: ExtensionRef, provider: str
) -> Registration:
    try:
        entry = _REGISTRY[(kind, reference.name, reference.version)]
    except KeyError as exc:
        raise ValueError(
            f"unregistered {kind} extension {reference.name}@{reference.version}"
        ) from exc
    if provider not in entry.backends:
        raise ValueError(f"extension {reference.name} does not support {provider}")
    entry.verify_sources()
    if reference.code_sha256 is not None and reference.code_sha256 != entry.code_sha256:
        raise ValueError(f"extension {reference.name} implementation digest changed")
    return entry


def export_registrations(
    spec: ExtensionSpec | None, provider: str
) -> tuple[Registration, ...]:
    """Pass explicitly registered implementations to local spawn workers."""
    if spec is None:
        return ()
    references = [
        ("observation", spec.observation),
        *(("observation", value) for value in spec.role_observations.values()),
    ]
    if spec.reward:
        references += [
            ("reward", spec.reward.team),
            *(("reward", value) for value in spec.reward.roles.values()),
        ]
    if spec.network:
        for network in (spec.network, *spec.network.roles.values()):
            references += [
                ("torch_encoder", network.actor.encoder),
                ("torch_encoder", network.critic.encoder),
            ]
    entries = [
        registration(kind, ref, provider) for kind, ref in references if ref is not None
    ]
    return tuple(
        {(entry.kind, entry.name, entry.version): entry for entry in entries}.values()
    )


def install_registrations(entries):
    """Install only Python capability objects passed by the calling experiment."""
    for entry in entries:
        if not isinstance(entry, Registration):
            raise ValueError(
                "worker extension registration must be an explicit Registration"
            )
        entry.verify_sources()
        key = (entry.kind, entry.name, entry.version)
        if key in _REGISTRY and _REGISTRY[key].identity() != entry.identity():
            raise ValueError(
                "worker extension registration conflicts with an installed implementation"
            )
        _REGISTRY[key] = entry


def bind_extensions(spec: ExtensionSpec | None, provider: str) -> ExtensionSpec | None:
    """Pin every declared implementation before model/environment construction."""
    if spec is None:
        return None
    if provider not in BACKENDS:
        raise ValueError("unknown extension backend")
    if provider != "rllib.resource_ppo" and (
        spec.role_observations
        or spec.network
        and spec.network.roles
        or spec.reward
        and spec.reward.roles
    ):
        raise ValueError("role extensions require the resource-agent backend")

    def pin(kind, reference):
        if reference is None:
            return None
        entry = registration(kind, reference, provider)
        return reference.model_copy(
            update={"code_sha256": entry.code_sha256}, deep=True
        )

    def branches(network):
        return network.model_copy(
            update={
                key: getattr(network, key).model_copy(
                    update={
                        "encoder": pin("torch_encoder", getattr(network, key).encoder)
                    },
                    deep=True,
                )
                for key in ("actor", "critic")
            },
            deep=True,
        )

    updates = {
        "observation": pin("observation", spec.observation),
        "role_observations": {
            role: pin("observation", ref)
            for role, ref in spec.role_observations.items()
        },
    }
    if spec.network:
        network = branches(spec.network)
        updates["network"] = network.model_copy(
            update={
                "roles": {
                    role: branches(value) for role, value in network.roles.items()
                }
            },
            deep=True,
        )
    if spec.reward:
        updates["reward"] = spec.reward.model_copy(
            update={
                "team": pin("reward", spec.reward.team),
                "roles": {
                    role: pin("reward", ref) for role, ref in spec.reward.roles.items()
                },
            },
            deep=True,
        )
    return spec.model_copy(update=updates, deep=True)


class _Component:
    def __init__(self, kind, reference, provider, *args):
        self.reference = reference
        self.registration = registration(kind, reference, provider)
        self.instance = self.registration.factory(
            copy.deepcopy(reference.parameters), *args
        )
        if self.registration.stateful and (
            not callable(getattr(self.instance, "state_dict", None))
            or not callable(getattr(self.instance, "load_state_dict", None))
        ):
            raise ValueError(
                f"stateful extension {reference.name} lacks its save/restore protocol"
            )

    def save(self):
        value = self.instance.state_dict() if self.registration.stateful else None
        json.dumps(value, allow_nan=False)
        return {
            "identity": self.registration.identity(),
            "parameters": copy.deepcopy(self.reference.parameters),
            "state": copy.deepcopy(value),
        }

    def restore(self, saved):
        if (
            saved["identity"] != self.registration.identity()
            or saved["parameters"] != self.reference.parameters
        ):
            raise ValueError("extension state identity mismatch")
        if self.registration.stateful:
            self.instance.load_state_dict(copy.deepcopy(saved["state"]))
        elif saved["state"] is not None:
            raise ValueError(
                "stateless extension checkpoint contains unexplained state"
            )


class ExtensionsRuntime:
    """Independent observation/reward instances; action catalogs remain external."""

    def __init__(
        self,
        spec: ExtensionSpec | None,
        provider: str,
        layouts: dict[str | None, ObservationLayout],
        *,
        learner_scale=1.0,
    ):
        if (
            not isinstance(learner_scale, Real)
            or isinstance(learner_scale, bool)
            or not isfinite(learner_scale)
            or learner_scale <= 0
        ):
            raise ValueError("learner reward scale must be positive and finite")
        self.spec = bind_extensions(spec, provider)
        self.provider, self.layouts = provider, dict(layouts)
        self.learner_scale = float(learner_scale) * (
            self.spec.reward.learner_scale if self.spec and self.spec.reward else 1.0
        )
        if not isfinite(self.learner_scale):
            raise ValueError("combined learner reward scale must be finite")
        self.components = {}
        self.spaces = {}
        for role, layout in layouts.items():
            reference = self.spec.observation_for_role(role) if self.spec else None
            if reference is None:
                self.spaces[role] = layout.vector
                continue
            component = _Component("observation", reference, provider, layout)
            space = component.instance.output_space
            if not isinstance(space, ObservationSpace):
                raise ValueError(
                    "observation extension must declare an ObservationSpace"
                )
            self.components[("observation", role)] = component
            self.spaces[role] = space
        if self.spec and self.spec.reward:
            if self.spec.reward.team:
                self.components[("reward", None)] = _Component(
                    "reward", self.spec.reward.team, provider
                )
            for role, reference in self.spec.reward.roles.items():
                self.components[("reward", role)] = _Component(
                    "reward", reference, provider
                )

    def encode(self, observation: PublicObservation):
        component = self.components.get(("observation", observation.role))
        value = (
            component.instance.encode(observation) if component else observation.vector
        )
        return self.spaces[observation.role].validate(value)

    def _transform_reward(self, transition, role, research):
        component = self.components.get(("reward", role))
        if component:
            research = component.instance.transform(transition, research)
        if not isinstance(research, Real) or isinstance(research, bool):
            raise ValueError("reward extension must return a finite scalar")
        return float(research)

    def _reward_values(self, raw, research):
        result = RewardValues(float(raw), research, research * self.learner_scale)
        if not all(
            isfinite(value) for value in (result.raw, result.research, result.learner)
        ):
            raise ValueError("reward extension produced nonfinite rewards")
        return result

    def reward(self, transition: RewardTransition) -> RewardValues:
        """One team/central transform; never call this once per resource agent."""
        if transition.role is not None or transition.agent_id is not None:
            raise ValueError("use rewards() to transform a resource joint round once")
        research = self._transform_reward(transition, None, transition.raw_reward)
        return self._reward_values(transition.raw_reward, research)

    def rewards(self, transition: RewardTransition, roles) -> RewardBatch:
        """Transform the team once and each role once, including terminal outcomes."""
        team = self.reward(transition)
        values = []
        for role in sorted(set(roles)):
            if role not in ("machine_policy", "agv_policy"):
                raise ValueError("unknown resource reward role")
            value = self._transform_reward(
                replace(transition, role=role), role, team.research
            )
            values.append((role, self._reward_values(transition.raw_reward, value)))
        return RewardBatch(team, tuple(values))

    def begin_episode(self):
        for component in self.components.values():
            hook = getattr(component.instance, "begin_episode", None)
            if hook:
                hook()

    def state_dict(self):
        return {
            "schema": "smartsom.extension-state/v1",
            "provider": self.provider,
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
            "learner_scale": self.learner_scale,
            "spaces": [
                {
                    "role": role,
                    "vector": list(space.vector) if space.vector else None,
                    "fields": [[name, list(shape)] for name, shape in space.fields],
                }
                for role, space in self.spaces.items()
            ],
            "components": [
                {"kind": kind, "role": role, **component.save()}
                for (kind, role), component in self.components.items()
            ],
        }

    def load_state_dict(self, state):
        json.dumps(state, allow_nan=False)
        if state.get("schema") != "smartsom.extension-state/v1":
            raise ValueError("unsupported extension state schema")
        expected = self.state_dict()
        if any(
            state.get(key) != expected[key]
            for key in ("provider", "spec", "learner_scale", "spaces")
        ):
            raise ValueError(
                "extension state numerical/configuration contract mismatch"
            )
        rows = state.get("components", [])
        keys = [(row["kind"], row["role"]) for row in rows]
        if len(set(keys)) != len(keys) or set(keys) != set(self.components):
            raise ValueError("extension state coverage mismatch")
        for row in rows:
            self.components[(row["kind"], row["role"])].restore(row)


class _Vector:
    def __init__(self, parameters, layout):
        if parameters:
            raise ValueError("builtin.vector has no parameters")
        self.output_space = layout.vector

    def encode(self, observation):
        return observation.vector


class _Dict:
    def __init__(self, parameters, layout):
        if parameters:
            raise ValueError("builtin.dict has no parameters")
        self.output_space = layout.grouped

    def encode(self, observation):
        return dict(observation.groups)


class _RewardScale:
    def __init__(self, parameters):
        if set(parameters) - {"scale", "terminal_offset"}:
            raise ValueError(
                "builtin.reward_scale only accepts scale and terminal_offset"
            )
        if any(
            not isinstance(value, Real) or isinstance(value, bool)
            for value in parameters.values()
        ):
            raise ValueError("reward parameters must be numeric scalars")
        self.scale = float(parameters.get("scale", 1.0))
        self.terminal_offset = float(parameters.get("terminal_offset", 0.0))
        if not isfinite(self.scale) or not isfinite(self.terminal_offset):
            raise ValueError("reward parameters must be finite")

    def transform(self, transition, reward):
        return reward * self.scale + (
            self.terminal_offset if transition.reason is not None else 0.0
        )


def _flatten_factory(parameters, space):
    from smartsom.learning.torch_extensions import FlattenEncoder

    return FlattenEncoder(parameters, space)


def _mlp_factory(parameters, space):
    from smartsom.learning.torch_extensions import MLPEncoder

    return MLPEncoder(parameters, space)


register_extension("observation", "builtin.vector", "1", _Vector)
register_extension("observation", "builtin.dict", "1", _Dict)
register_extension("reward", "builtin.reward_scale", "1", _RewardScale)
# Torch implementation files are hashed without importing the optional framework.
for _name, _factory in (
    ("builtin.flatten", _flatten_factory),
    ("builtin.mlp", _mlp_factory),
):
    register_extension(
        "torch_encoder",
        _name,
        "1",
        _factory,
        source_files=(Path(__file__).with_name("torch_extensions.py"),),
    )
