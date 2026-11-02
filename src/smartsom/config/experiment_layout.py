"""Directory-defined experiments; no separately maintained name registry."""

from pathlib import Path

from smartsom.config.codec import ConfigurationError, read_document

EXCLUDED = {"common", "extensions", "analysis", "tests", "inputs", "source"}


def entrypoints(directory):
    directory = Path(directory)
    direct = directory / "experiment.yaml"
    if direct.is_file():
        return [direct]
    candidates = sorted(
        p
        for p in directory.rglob("*")
        if p.suffix in {".yaml", ".yml"}
        and not EXCLUDED.intersection(p.relative_to(directory).parts[:-1])
    )
    return [
        p
        for p in candidates
        if p.name == "experiment.yaml"
        or read_document(p).get("schema") == "smartsom.base-experiment/v1"
    ]


def entry_id(path, directory):
    """Canonical directory IDs and named YAML IDs share one relative namespace."""
    relative = path.relative_to(directory)
    return (
        relative.parent if path.name == "experiment.yaml" else relative.with_suffix("")
    ).as_posix()


def location(path):
    """Return repository, study-relative path and variant for a canonical entry."""
    path = Path(path).resolve()
    if path.name != "experiment.yaml":
        return None
    roots = [p for p in path.parents if p.name == "experiments"]
    if not roots:
        return None
    root = roots[0]
    parts = path.relative_to(root).parts
    if len(parts) < 4:
        raise ConfigurationError(
            "experiment entry requires experiments/TOPIC/STUDY/VARIANT/experiment.yaml"
        )
    return root.parent, Path(*parts[:-2]), parts[-2]


def batch_directory(path):
    """Scientific batch ownership does not depend on the submitted selection."""
    path = Path(path).resolve()
    canonical = location(path)
    return path.parent.parent if canonical else path.parent


def default_batch_root(directory):
    """Place external named batches under the project's runs tree, never inputs."""
    import hashlib

    from smartsom.config.launch import project_root

    directory = Path(directory).resolve()
    experiment_roots = [
        p for p in (directory, *directory.parents) if p.name == "experiments"
    ]
    if experiment_roots:
        experiments = experiment_roots[0]
        return experiments.parent / "runs" / directory.relative_to(experiments)
    binding = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    return project_root() / "runs" / "batches" / (directory.name + "-" + binding)
