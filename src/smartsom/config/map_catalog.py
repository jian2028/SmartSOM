"""The four bundled factory designs, shared by authoring and Studio."""

from importlib.resources import as_file, files

from smartsom.config.factory_design import load_factory_design_file

MAP_NAMES = (
    "small-wide-factory",
    "medium-wide-factory",
    "small-factory",
    "medium-factory",
)


def load_map(name):
    """Load one named package resource without requiring a repository checkout."""
    if name not in MAP_NAMES:
        raise ValueError(f"Unknown bundled factory: {name}")
    resource = files("smartsom.config").joinpath("maps", f"{name}.yaml")
    with as_file(resource) as path:
        return load_factory_design_file(path)[0]
