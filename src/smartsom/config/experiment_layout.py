"""Directory-defined experiments; no separately maintained name registry."""

from pathlib import Path

from smartsom.config.codec import ConfigurationError

EXCLUDED = {"common", "extensions", "analysis", "tests", "inputs", "source"}


def entrypoints(directory):
    directory = Path(directory)
    direct = directory / "experiment.yaml"
    if direct.is_file():
        return [direct]
    return sorted(
        p
        for p in directory.rglob("experiment.yaml")
        if not EXCLUDED.intersection(p.relative_to(directory).parts[:-1])
    )


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
