"""Explicit component resolution and relocatable, hashed source bundles."""

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import random
import sys
import types
from pathlib import Path


def source_files(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if any(
            p in {"__pycache__", ".git", ".venv", ".pytest_cache"}
            for p in path.relative_to(root).parts
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"extension source cannot contain symlinks: {path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def digest_files(files):
    h = hashlib.sha256()
    for name, raw in sorted(files.items()):
        h.update(name.encode())
        h.update(b"\0")
        h.update(raw)
        h.update(b"\0")
    return h.hexdigest()


def builtin_sources():
    import smartsom

    root = Path(smartsom.__file__).parent
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*.py"))
    }


@contextlib.contextmanager
def isolated_random():
    """Checking trusted factories must not advance training random streams."""
    state = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    ns = numpy.random.get_state() if numpy is not None else None
    ts = torch.get_rng_state() if torch is not None else None
    try:
        yield
    finally:
        random.setstate(state)
        if ns is not None:
            numpy.random.set_state(ns)
        if ts is not None:
            torch.set_rng_state(ts)


def _external_files(spec):
    path = Path(spec.path).resolve()
    if spec.source_root:
        root = Path(spec.source_root).resolve()
        if not path.is_relative_to(root):
            raise ValueError("component entry must be inside source_root")
        files = source_files(root)
        relative = path.relative_to(root)
    else:
        root, relative = path.parent, Path(path.name)
        files = {path.name: path.read_bytes()}
    # Local imports use relative package imports; ambient cwd is never a source.
    for name, raw in files.items():
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(raw, filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level and not spec.source_root:
                raise ValueError("relative imports require an explicit source_root")
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    and not node.level
                    and node.module
                    else []
                )
            )
            for module in modules:
                first = module.split(".")[0]
                if (root / (first + ".py")).exists() or (root / first).is_dir():
                    raise ValueError(
                        f"local import {module!r} must be relative inside source_root"
                    )
    return root, relative, files


def resolve(spec):
    if spec.target:
        module_name, attribute = spec.target.rsplit(".", 1)
        if not module_name.startswith("smartsom."):
            raise ValueError(
                "module targets must name packaged smartsom components; use path/source_root for extensions"
            )
        if spec.sha256 and digest_files(builtin_sources()) != spec.sha256:
            raise ValueError("builtin component source mismatch")
        module = importlib.import_module(module_name)
    else:
        root, relative, files = _external_files(spec)
        digest = digest_files(files)
        if spec.sha256 and digest != spec.sha256:
            raise ValueError(f"extension source mismatch: {spec.path}")
        prefix = (
            "smartsom_extension_"
            + digest
            + "_"
            + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        )
        if prefix not in sys.modules:
            initializer = root / "__init__.py"
            if spec.source_root and initializer.is_file():
                package_spec = importlib.util.spec_from_file_location(
                    prefix, initializer, submodule_search_locations=[str(root)]
                )
                package = importlib.util.module_from_spec(package_spec)
                sys.modules[prefix] = package
                try:
                    package_spec.loader.exec_module(package)
                except BaseException:
                    for key in list(sys.modules):
                        if key == prefix or key.startswith(prefix + "."):
                            sys.modules.pop(key, None)
                    raise
            else:
                package = types.ModuleType(prefix)
                package.__path__ = [str(root)]
                package.__package__ = prefix
                sys.modules[prefix] = package
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join([prefix, *parts])
        try:
            module = importlib.import_module(name)
        except BaseException:
            for key in list(sys.modules):
                if key == prefix or key.startswith(prefix + "."):
                    sys.modules.pop(key, None)
            raise
        attribute = spec.factory
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"component must export callable {attribute!r}")
    return factory


def instantiate(factory, params, **context):
    overlap = set(context) & set(params)
    if overlap:
        raise ValueError(
            f"component parameters cannot override runtime context: {sorted(overlap)}"
        )
    return factory(**context, **params)


def construct(spec, **context):
    return instantiate(resolve(spec), spec.params, **context)


def freeze_components(algorithm, base):
    """Return canonical declarations, source bytes and resolved inspection data."""
    document = algorithm.model_dump(mode="json", by_alias=True)
    files, report = {}, {}
    builtins = None
    for key, original in algorithm.component_items():
        spec = original
        if spec.target:
            if builtins is None:
                builtins = builtin_sources()
                files.update(
                    {"components/builtin/" + n: raw for n, raw in builtins.items()}
                )
            digest = digest_files(builtins)
            location = spec.target.rsplit(".", 1)[0]
        else:
            spec = spec.model_copy(
                update={
                    "path": str((base / spec.path).resolve()),
                    "source_root": str((base / spec.source_root).resolve())
                    if spec.source_root
                    else None,
                }
            )
            _, relative, bundle = _external_files(spec)
            digest = digest_files(bundle)
            location = spec.path
            folder = "components/" + key.replace(".", "-")
            files.update({folder + "/" + n: raw for n, raw in bundle.items()})
        if original.sha256 and original.sha256 != digest:
            raise ValueError(f"component source mismatch: {key}")
        with isolated_random():
            factory = resolve(spec)
        node = document
        for part in key.split("."):
            node = node[part]
        node["sha256"] = digest
        if not spec.target:
            node["path"] = folder + "/" + relative.as_posix()
            node["source_root"] = folder if spec.source_root else None
        report[key] = {
            "source": str(Path(sys.modules[factory.__module__].__file__).resolve()),
            "entry": spec.target or spec.factory,
            "sha256": digest,
            "reference": location,
            "contract": getattr(factory, "CONTRACT", None),
            "input_schema": getattr(factory, "INPUT_SCHEMA", None),
            "output_schema": getattr(factory, "OUTPUT_SCHEMA", None),
        }
    return document, tuple(sorted(files.items())), report


def absolute_components(algorithm, base):
    from smartsom.config.base_diagnostic import BaseAlgorithm

    data = algorithm.model_dump(mode="json", by_alias=True)
    for key, spec in algorithm.component_items():
        if spec.path:
            node = data
            for part in key.split("."):
                node = node[part]
            node["path"] = str((base / spec.path).resolve())
            if spec.source_root:
                node["source_root"] = str((base / spec.source_root).resolve())
    result = BaseAlgorithm.model_validate(data)
    result._owner_ids = algorithm._owner_ids
    return result


def validate_components(algorithm):
    with isolated_random():
        contexts = {
            "learner": (
                "options",
                "seed",
                "training",
                "selection",
                "numerical_threads",
            ),
            "observation": (),
            "encoder": ("group", "options"),
            "network": ("group", "options", "value_head"),
            "implementation": ("seed", "control"),
            "mover": ("routes", "seed"),
        }
        for key, spec in algorithm.component_items():
            context = {name: None for name in contexts[key.split(".")[-1]]}
            if set(context) & set(spec.params):
                raise ValueError("component params cannot override runtime context")
            inspect.signature(resolve(spec)).bind(**context, **spec.params)
        for name, policy in algorithm.production.policies.items():
            if policy.kind == "learning":
                learner = resolve(policy.learner)
                if getattr(learner, "CONTRACT", None) != "smartsom.learner/v1":
                    raise ValueError("learner factory must declare smartsom.learner/v1")
                modes = {
                    policy.parameter_sharing.mode(g)
                    for g in ("machine", "dispatcher", "pre", "post", "input")
                }
                supported = set(getattr(learner, "PARAMETER_SHARING", ("shared",)))
                if modes - supported:
                    raise ValueError(
                        f"{name}: learner does not support requested parameter sharing"
                    )
                if policy.role_params and not getattr(learner, "ROLE_PARAMS", False):
                    raise ValueError(f"{name}: learner does not support role_params")
                observation = resolve(policy.observation)
                network = resolve(policy.network or policy.encoder)
                output = getattr(observation, "OUTPUT_SCHEMA", None)
                if not output or output != getattr(network, "INPUT_SCHEMA", None):
                    raise ValueError(
                        f"{name}: observation/network INPUT_SCHEMA mismatch"
                    )
                required_features = getattr(learner, "FEATURE_SCHEMA", None)
                if (
                    policy.encoder
                    and required_features
                    and getattr(network, "OUTPUT_SCHEMA", None) != required_features
                ):
                    raise ValueError(
                        f"{name}: encoder output does not match learner heads"
                    )
                if not callable(getattr(network, "pack", None)):
                    raise ValueError("encoder/network factory must expose pack")
                construct(policy.observation)
            else:
                if policy.observation is not None:
                    projection = resolve(policy.observation)
                    if not getattr(projection, "OUTPUT_SCHEMA", None):
                        raise ValueError(
                            "observation factory must declare OUTPUT_SCHEMA"
                        )
                    construct(policy.observation)
                policy_factory = resolve(policy.implementation)
                if getattr(policy_factory, "CONTRACT", None) != "smartsom.policy/v1":
                    raise ValueError("policy factory must declare smartsom.policy/v1")
        if (
            getattr(resolve(algorithm.traffic.mover), "CONTRACT", None)
            != "smartsom.mover/v1"
        ):
            raise ValueError("mover factory must declare smartsom.mover/v1")
