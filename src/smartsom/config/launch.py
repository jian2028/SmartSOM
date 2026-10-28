"""Explicit operation profiles; no scientific settings or implicit inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field

from smartsom.config.codec import ConfigurationError, StrictModel, digest, read_document


class Launch(StrictModel):
    mode: Literal["foreground", "background"] = "foreground"


class Resources(StrictModel):
    cpus: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class Concurrency(StrictModel):
    mode: Literal["manual", "adaptive"] = "manual"
    max_concurrent: int = Field(default=2, gt=0, le=1024)


class Display(StrictModel):
    progress: Literal["auto", "on", "off"] = "auto"
    log_format: Literal["text", "json"] = "text"
    summary_interval: float = Field(default=5, gt=0, allow_inf_nan=False)


class Tracking(StrictModel):
    mode: Literal["online", "offline", "disabled"] | None = None
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    tensorboard: bool = False


class Profile(StrictModel):
    launch: Launch = Field(default_factory=Launch)
    concurrency: Concurrency = Field(default_factory=Concurrency)
    resources: Resources = Field(default_factory=Resources)
    tracking: Tracking = Field(default_factory=Tracking)
    display: Display = Field(default_factory=Display)
    output_root: str | None = None


def project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs"
        ).is_dir():
            return candidate
    return start


def experiment_names(root=None):
    folder = project_root(root) / "configs/runs"
    return sorted(
        {
            p.stem if p.is_file() else p.name
            for p in folder.rglob("*")
            if (p.is_file() and p.suffix in {".yaml", ".yml"})
            or (p.is_dir() and any(p.glob("*.yaml")))
        }
    )


def resolve_experiment(value, root=None):
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    folder = project_root(root) / "configs/runs"
    matches = sorted(
        p
        for p in folder.rglob("*")
        if (p.is_file() and p.suffix in {".yaml", ".yml"} and p.stem == str(value))
        or (p.is_dir() and p.name == str(value) and any(p.glob("*.yaml")))
    )
    if len(matches) != 1:
        raise ConfigurationError(
            f"experiment name {value!r}: expected one match; candidates: "
            + ", ".join(map(str, matches))
        )
    return matches[0].resolve()


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + "."))
        else:
            out[name] = item
    return out


def merge(target, incoming):
    for key, value in incoming.items():
        if isinstance(value, dict):
            merge(target.setdefault(key, {}), value)
        else:
            target[key] = value


def resolve_operations(experiment, *, profile=None, cli=None):
    result = Profile().model_dump(mode="json")
    origins = dict.fromkeys(flatten(result), "default")
    authored = {key: deepcopy(experiment[key]) for key in result if key in experiment}
    # Existing current configs keep their scheduling defaults until explicitly overridden.
    run = experiment.get("run", {})
    if "max_concurrent" in run:
        authored.setdefault("concurrency", {}).setdefault(
            "max_concurrent", run["max_concurrent"]
        )
    if "resource_execution" in run:
        authored.setdefault("concurrency", {}).setdefault(
            "mode", "adaptive" if run["resource_execution"] == "adaptive" else "manual"
        )
    sources = [("experiment", authored)]
    if profile:
        path = Path(profile).expanduser()
        if not path.exists():
            path = project_root() / "configs/profiles" / (str(profile) + ".yaml")
        value = Profile.model_validate(read_document(path)).model_dump(
            mode="json", exclude_unset=True
        )
        if value.get("output_root"):
            value["output_root"] = str(
                (path.resolve().parent / value["output_root"]).resolve()
            )
        sources.append((str(path.resolve()), value))
    sources.append(("CLI", cli or {}))
    for origin, values in sources:
        Profile.model_validate(values)
        merge(result, values)
        origins.update(dict.fromkeys(flatten(values), origin))
    result = Profile.model_validate(result).model_dump(mode="json")
    return result, origins


OPERATION_RUN_KEYS = {
    "output",
    "background",
    "max_concurrent",
    "max_failures",
    "resource_execution",
    "diagnostics",
    "metrics",
    "metric_interval",
    "tags",
}


def scientific_documents(documents):
    value = deepcopy(documents)
    ex = value["experiment"]
    for key in (
        "launch",
        "tracking",
        "concurrency",
        "resources",
        "display",
        "output_root",
        "batch",
    ):
        ex.pop(key, None)
    for key in OPERATION_RUN_KEYS:
        ex.get("run", {}).pop(key, None)
    return value


def scientific_identity(documents):
    return digest(scientific_documents(documents))
