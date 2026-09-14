"""Load bundled designs as new documents without changing their source."""

from importlib.resources import as_file, files

from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import FactoryDesign


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
