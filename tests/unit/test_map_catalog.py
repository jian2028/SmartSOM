"""Only four independently stored maps ship with core and Studio."""

import importlib.resources
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from smartsom.config.factory_design import (
    load_factory_design_file,
    save_factory_design_file,
)
from smartsom.config.map_catalog import MAP_NAMES, load_map
from smartsom.studio.persistence import TemplateCatalog


@pytest.mark.parametrize("name", MAP_NAMES)
def test_catalog_roundtrip_and_packaged_resources(name, tmp_path):
    original = load_map(name)
    path = tmp_path / "copy.yaml"
    save_factory_design_file(path, original)
    assert load_factory_design_file(path)[0] == original
    assert TemplateCatalog(tmp_path).load_builtin_file(name)[0] == original


def test_exactly_four_independent_factory_documents():
    root = Path(__file__).resolve().parents[2]
    documents = []
    for folder in ("configs", "src", "tests/fixtures", "data", "experiments"):
        for path in (root / folder).rglob("*"):
            if path.suffix not in {".yaml", ".yml", ".json"}:
                continue
            document = yaml.safe_load(path.read_text())
            if isinstance(document, dict) and str(
                document.get("schema", "")
            ).startswith("smartsom.factory/"):
                documents.append(path.relative_to(root).as_posix())
    assert sorted(documents) == sorted(
        f"src/smartsom/config/maps/{name}.yaml" for name in MAP_NAMES
    )
    resource = importlib.resources.files("smartsom.config").joinpath("maps")
    assert sorted(
        p.name for p in resource.iterdir() if p.name.endswith(".yaml")
    ) == sorted(f"{name}.yaml" for name in MAP_NAMES)


def test_catalog_loading_has_no_optional_framework_imports(tmp_path):
    code = """
import importlib.abc
import sys
class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6','numpy','gymnasium','ray','torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectFrameworks())
from smartsom.config.map_catalog import MAP_NAMES, load_map
for name in MAP_NAMES:
    assert load_map(name).factory.machines
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        check=True,
    )


def test_retired_builtin_is_not_silently_replaced(tmp_path):
    catalog = TemplateCatalog(tmp_path)
    with pytest.raises(ValueError, match="Retired"):
        catalog.load_builtin_file(1)
    assert not catalog.directory.exists()
    with pytest.raises(ValueError, match="Unknown"):
        load_map("large-factory")


def test_existing_local_override_stays_readable_and_is_never_relabelled(tmp_path):
    from smartsom.studio.persistence import write_json

    catalog = TemplateCatalog(tmp_path)
    catalog.directory.mkdir(parents=True)
    legacy_path = catalog.builtin_path(2)
    save_factory_design_file(legacy_path, load_map("small-factory"))
    write_json(catalog.records_path, {"local": [2], "files": []})
    before = {p.name: p.read_bytes() for p in catalog.directory.iterdir()}
    envelope, origin = catalog.load_builtin_file(2)
    assert origin.builtin == 2 and origin.target_path == legacy_path
    assert envelope == load_map("small-factory")
    assert before == {p.name: p.read_bytes() for p in catalog.directory.iterdir()}
