"""Explicit, framework-free student rules over detached public decisions.

Python modules are loaded only by an explicit caller capability. YAML contains
registered names and versions, never an executable import path.
"""

import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib.machinery import PathFinder
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from smartsom.domain.production_decisions import ROLE_NAMES, DecisionRequest

RULE_CONTRACT = "smartsom.public-rule/v1"
_BUILTINS = {
    "machine": {"spt", "normal_first", "random"},
    "buffer": {"edd", "spt", "random"},
    "dispatcher": {"nearest", "random"},
    "mover": {"shortest_path", "random", "automatic_travel"},
}
_LOADING_MODULE = ContextVar("smartsom_loading_rule_module", default=None)
_REGISTRY = {}
_MODULES = {}

_JOB_FEATURES = (
    "remaining_operations_scaled",
    "due_slack_scaled",
    "priority_scaled",
    "waiting_time_scaled",
    "replacement_attempt_scaled",
    "quality_unknown",
    "quality_pass",
    "quality_fail",
)
_FEATURES = {
    "machine": (
        *_JOB_FEATURES,
        "processing_time_scaled",
        "mode_time_scale",
        "mode_error_rate",
        "normal_mode",
    ),
    "buffer": (*_JOB_FEATURES, "processing_time_scaled", None, None, None),
    "dispatcher": (
        "has_target",
        "travel_time_scaled",
        "target_x_scaled",
        "target_y_scaled",
        "carrying_job",
        "source_supply_scaled",
        "source_reserved_scaled",
        "inventory_scaled",
        "capacity_scaled",
        "finite_capacity",
        "full_capacity",
        "current_target",
    ),
    "mover": (
        "dx",
        "dy",
        "distance_to_target_scaled",
        "has_target",
        "carrying_job",
        "in_service",
        "next_cell_is_port",
        "next_cell_occupied",
        None,
        None,
        None,
        None,
    ),
}


def _json_copy(value):
    """Validate finite JSON state/parameters and detach their mutable contents."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _source_digest(files):
    return _digest(
        [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    )


@dataclass(frozen=True, slots=True)
class PublicRuleCandidate:
    """A stable semantic choice with named, normalized public features."""

    identity: str
    action: object
    features: Mapping[str, float]

    def feature(self, name: str) -> float:
        try:
            return self.features[name]
        except KeyError as exc:
            raise ValueError(f"unavailable public candidate feature {name!r}") from exc


@dataclass(frozen=True, slots=True)
class PublicRuleRequest:
    """No simulator, event schedule, unrevealed jobs or clock-advancing methods."""

    tick: int
    stage: str
    role: str
    owner: str
    candidates: tuple[PublicRuleCandidate, ...]
    observation: dict
    prefix: tuple[str, ...]
    count: int
    contract: str = RULE_CONTRACT

    @property
    def identity(self):
        return f"{self.tick}:{self.stage}:{self.role}:{self.owner}"

    @property
    def time_scale(self):
        return self.observation.get("feature_time_scale", 1.0)

    def job(self, job_id: str) -> dict:
        """Only already-public jobs are accessible; future IDs fail explicitly."""
        try:
            return copy.deepcopy(self.observation["jobs"][job_id])
        except KeyError as exc:
            raise ValueError(f"job {job_id!r} is not publicly revealed") from exc


def public_rule_request(request: DecisionRequest) -> PublicRuleRequest:
    if not isinstance(request, DecisionRequest) or request.role not in ROLE_NAMES:
        raise ValueError("rule requires a public production DecisionRequest")
    observation = copy.deepcopy(request.observation)
    candidates = []
    for candidate in request.candidates:
        if not candidate.legal:
            continue
        names = _FEATURES[request.role]
        if len(candidate.features) != len(names):
            raise ValueError("unsupported rule candidate feature contract")
        features = {
            name: float(value)
            for name, value in zip(names, candidate.features, strict=True)
            if name is not None
        }
        if request.role in ("machine", "buffer"):
            job_id = (
                candidate.action[0] if request.role == "machine" else candidate.action
            )
            job = observation.get("jobs", {}).get(job_id, {})
            features["rush"] = float(job.get("rush", False))
        candidates.append(
            PublicRuleCandidate(
                candidate.identity,
                copy.deepcopy(candidate.action),
                MappingProxyType(features),
            )
        )
    return PublicRuleRequest(
        request.tick,
        request.stage,
        request.role,
        request.owner,
        tuple(candidates),
        observation,
        tuple(request.prefix),
        request.count,
    )


class Rule(Protocol):
    def choose(self, request: PublicRuleRequest) -> object:
        """Return one of request.candidates' semantic actions."""


class StatefulRule(Rule, Protocol):
    def reset(self) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


@dataclass(frozen=True, slots=True)
class RuleRegistration:
    name: str
    version: str
    factory: Callable
    roles: tuple[str, ...]
    stateful: bool
    sources: tuple[Path, ...]
    code_sha256: str
    extension_module: str | None = None

    def verify_sources(self):
        if _source_digest(self.sources) != self.code_sha256:
            raise ValueError(
                f"rule {self.name}@{self.version} source changed after loading"
            )

    def identity(self):
        self.verify_sources()
        return {
            "name": self.name,
            "version": self.version,
            "code_sha256": self.code_sha256,
            "roles": list(self.roles),
            "stateful": self.stateful,
            "contract": RULE_CONTRACT,
            "extension_module": self.extension_module,
        }


def register_rule(
    name: str,
    version: str,
    factory: Callable,
    *,
    roles=ROLE_NAMES,
    stateful=False,
    source_files=(),
) -> RuleRegistration:
    """Factory(parameters=dict, seed=int) returns a Rule or StatefulRule.

    Declare helper files in source_files so their changes invalidate a frozen
    run. Stateful rules must implement reset/state_dict/load_state_dict.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
        raise ValueError(
            "rule name must be a registered name, not a Python import path"
        )
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", version):
        raise ValueError("rule requires an explicit nonempty version")
    if name in {item for values in _BUILTINS.values() for item in values}:
        raise ValueError("custom rules cannot replace builtin rule names")
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or set(roles) - set(ROLE_NAMES):
        raise ValueError("rule must declare supported production roles")
    if not callable(factory) or type(stateful) is not bool:
        raise ValueError("rule requires a factory and boolean stateful declaration")
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError("rule factory must have inspectable Python source")
    files = tuple(
        dict.fromkeys(Path(path).resolve() for path in (source, *source_files))
    )
    if any(not path.is_file() for path in files):
        raise ValueError("rule implementation source is unavailable")
    entry = RuleRegistration(
        name,
        version,
        factory,
        roles,
        stateful,
        files,
        _source_digest(files),
        _LOADING_MODULE.get(),
    )
    key = (name, version)
    if key in _REGISTRY:
        installed = _REGISTRY[key]
        if installed.factory is not factory or installed.identity() != entry.identity():
            raise ValueError(f"rule {name}@{version} is already registered")
        return installed
    _REGISTRY[key] = entry
    return entry


def rule_registration(role, name, version=None, code_sha256=None):
    version = version or "1"
    if name in _BUILTINS.get(role, ()):
        if version != "1":
            raise ValueError(f"unsupported builtin rule version {name}@{version}")
        return None
    try:
        entry = _REGISTRY[(name, version)]
    except KeyError as exc:
        raise ValueError(f"unregistered {role} rule {name}@{version}") from exc
    if role not in entry.roles:
        raise ValueError(f"rule {name}@{version} does not support {role}")
    entry.verify_sources()
    if code_sha256 is not None and code_sha256 != entry.code_sha256:
        raise ValueError(f"rule {name}@{version} implementation digest changed")
    return entry


def _module_name(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name
    ):
        raise ValueError("extension module must be a Python module name")


def _module_sources(name):
    """Find Python files without find_spec's implicit parent-package imports."""
    _module_name(name)
    chain, search = [], None
    parts = name.split(".")
    for index in range(len(parts)):
        qualified = ".".join(parts[: index + 1])
        loaded = sys.modules.get(qualified)
        if loaded is not None:
            origin = (
                inspect.getsourcefile(loaded)
                if getattr(loaded, "__file__", None)
                else None
            )
            locations = getattr(loaded, "__path__", None)
        else:
            spec = PathFinder.find_spec(qualified, search)
            if spec is None:
                raise ValueError(f"extension module {qualified} source is unavailable")
            origin, locations = spec.origin, spec.submodule_search_locations
        if origin is not None:
            path = Path(origin).resolve()
            if path.suffix != ".py" or not path.is_file():
                raise ValueError(
                    f"extension module {qualified} has no inspectable Python source"
                )
            chain.append((qualified, path))
        elif locations is None:
            raise ValueError(f"extension module {qualified} source is unavailable")
        if index < len(parts) - 1:
            if locations is None:
                raise ValueError(f"extension module {qualified} is not a package")
            search = list(locations)
    if not chain or chain[-1][0] != name:
        raise ValueError(f"extension module {name} has no inspectable Python source")
    return chain


def _hash_file(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"extension source {path.name} is unavailable") from exc


def _chain_identity(chain):
    return {name: _hash_file(path) for name, path in chain}


def _module_record(name):
    chain = _module_sources(name)
    path = chain[-1][1]
    dependencies = {
        module: digest
        for module, digest in _chain_identity(chain).items()
        if module != name
    }
    files = {}
    for entry in _REGISTRY.values():
        if entry.extension_module != name:
            continue
        entry.verify_sources()
        for module, source in _module_sources(entry.factory.__module__):
            if module != name:
                dependencies[module] = _hash_file(source)
        for source in entry.sources:
            if source != path:
                relative = Path(os.path.relpath(source, path.parent)).as_posix()
                files[relative] = _hash_file(source)
    return {
        "module": name,
        "source_sha256": _hash_file(path),
        "dependencies": [
            {"module": module, "source_sha256": value}
            for module, value in sorted(dependencies.items())
        ],
        "source_files": [
            {"relative_path": relative, "source_sha256": value}
            for relative, value in sorted(files.items())
        ],
    }


def _verify_module_files(record):
    """Reject leaf, package or declared-helper drift before executing imports."""
    name = record["module"]
    chain = _module_sources(name)
    path = chain[-1][1]
    if _hash_file(path) != record["source_sha256"]:
        raise ValueError("extension module source changed since configuration freeze")
    dependencies = {
        item["module"]: item["source_sha256"] for item in record.get("dependencies", ())
    }
    actual = {
        module: value
        for module, value in _chain_identity(chain).items()
        if module != name
    }
    for module, expected in dependencies.items():
        for qualified, value in _chain_identity(_module_sources(module)).items():
            if qualified != name:
                actual[qualified] = value
        if actual.get(module) != expected:
            raise ValueError(
                "extension package/helper module source changed since configuration freeze"
            )
    if actual != dependencies:
        raise ValueError(
            "extension package dependencies changed since configuration freeze"
        )
    for item in record.get("source_files", ()):
        relative = Path(item["relative_path"])
        if relative.is_absolute():
            raise ValueError(
                "frozen extension helper source must use a relative locator"
            )
        if _hash_file((path.parent / relative).resolve()) != item["source_sha256"]:
            raise ValueError(
                "extension helper source changed since configuration freeze"
            )


def load_rule_modules(module_names) -> tuple[dict, ...]:
    """Load explicitly authorized modules, returning identities for spawn workers."""
    records = []
    for name in dict.fromkeys(module_names):
        _module_name(name)
        if name in _MODULES:
            try:
                _verify_module_files(_MODULES[name])
            except ValueError as exc:
                raise ValueError(
                    f"extension module {name} source changed after loading"
                ) from exc
        before = _chain_identity(_module_sources(name))
        token = _LOADING_MODULE.set(name)
        try:
            module = importlib.import_module(name)
        finally:
            _LOADING_MODULE.reset(token)
        path = Path(inspect.getsourcefile(module)).resolve()
        for key, entry in tuple(_REGISTRY.items()):
            if entry.extension_module is None and (
                entry.factory.__module__ == name or path in entry.sources
            ):
                _REGISTRY[key] = replace(entry, extension_module=name)
        if _chain_identity(_module_sources(name)) != before:
            raise ValueError(f"extension module {name} source changed during loading")
        record = _module_record(name)
        if name in _MODULES and _MODULES[name] != record:
            raise ValueError(f"extension module {name} source changed after loading")
        _MODULES[name] = record
        records.append(copy.deepcopy(record))
    return tuple(records)


def verify_rule_modules(records, *, load=False):
    """Frozen records are execution capabilities, never taken from rule YAML."""
    records = tuple(records)
    for record in records:
        if not load and _MODULES.get(record["module"]) != record:
            raise ValueError(
                "frozen extension module is missing or its identity changed"
            )
        _verify_module_files(record)
    actual = load_rule_modules([record["module"] for record in records]) if load else ()
    if load and list(actual) != list(records):
        raise ValueError("extension module identity changed since configuration freeze")
    for record in records:
        if _MODULES.get(record["module"]) != record:
            raise ValueError(
                "frozen extension module is missing or its identity changed"
            )


def freeze_rule(role, name, version=None, parameters=None, code_sha256=None) -> dict:
    """Validate and pin code, version, parameters and explicitly loaded modules."""
    parameters = _json_copy(parameters or {})
    if not isinstance(parameters, dict):
        raise ValueError("rule parameters must be a JSON object")
    entry = rule_registration(role, name, version, code_sha256)
    if entry is None:
        path = Path(__file__).with_name("production_rules.py")
        identity = {
            "name": name,
            "version": "1",
            "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "roles": [role],
            "stateful": True,
            "contract": RULE_CONTRACT,
            "extension_module": None,
        }
        if code_sha256 is not None and code_sha256 != identity["code_sha256"]:
            raise ValueError("builtin rule implementation digest changed")
    else:
        identity = entry.identity()
    module = identity["extension_module"]
    modules = [_MODULES[module]] if module in _MODULES else []
    verify_rule_modules(modules)
    return {
        **identity,
        "role": role,
        "parameters": parameters,
        "parameters_sha256": _digest(parameters),
        "modules": copy.deepcopy(modules),
    }


def instantiate_rule(entry: RuleRegistration, *, parameters, seed):
    entry.verify_sources()
    rule = entry.factory(parameters=_json_copy(parameters), seed=seed)
    required = (
        ("choose", "reset", "state_dict", "load_state_dict")
        if entry.stateful
        else ("choose",)
    )
    if any(not callable(getattr(rule, method, None)) for method in required):
        raise ValueError("rule does not implement its declared state protocol")
    if entry.stateful:
        rule.reset()
        state = _json_copy(rule.state_dict())
        if not isinstance(state, dict):
            raise ValueError("stateful rule state must be a finite JSON object")
    return rule
