"""Load bundled designs as new documents without changing their source."""

from importlib.resources import as_file, files

from smartsom.config.factory_design import load_factory_design, load_factory_design_file
from smartsom.domain.factory_design import FactoryDesign

BUILTIN_TEMPLATE_NUMBERS = (1, 2, 3, 4, 5, 6)


def _load_template(filename: str) -> FactoryDesign:
    resource = files("smartsom.studio").joinpath("templates", filename)
    with as_file(resource) as path:
        design, _ = load_factory_design(path)
    return design


def load_template_1() -> FactoryDesign:
    """Return the default compact four-machine design."""
    return _load_template("template_001.yaml")


def load_template_2() -> FactoryDesign:
    """Return the preserved eight-machine design."""
    return _load_template("template_002.yaml")


def load_template_3() -> FactoryDesign:
    """Return the compact eight-machine design with split inspection stations."""
    return _load_template("template_003.yaml")


def load_template_4() -> FactoryDesign:
    """Return Small: eight machines and eight symmetrically placed AGVs."""
    return _load_template("template_004.yaml")


def load_template_5() -> FactoryDesign:
    """Return Medium: twenty machines and twenty symmetrically placed AGVs."""
    return _load_template("template_005.yaml")


def load_template_6() -> FactoryDesign:
    """Return Large: forty machines and forty symmetrically placed AGVs."""
    return _load_template("template_006.yaml")


def load_template_file(number):
    if number not in BUILTIN_TEMPLATE_NUMBERS:
        raise ValueError(f"Unknown built-in template: {number}")
    resource = files("smartsom.studio").joinpath(
        "templates", f"template_{number:03d}.yaml"
    )
    with as_file(resource) as path:
        return load_factory_design_file(path)[0]
