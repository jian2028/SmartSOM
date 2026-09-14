"""Data preservation, strict parsing and safe file replacement without a GUI."""

import hashlib
import os
from dataclasses import replace

import pytest
import yaml

from smartsom.config.codec import ConfigurationError, read_model
from smartsom.config.factory_design import load_factory_design, save_factory_design
from smartsom.config.models import FactoryFile
from smartsom.domain.factory_design import (
    BufferDesign,
    Cell,
    FactoryDesign,
    Footprint,
    GridDesign,
    MachineDesign,
    PoolStorage,
    PortDesign,
    SlotDesign,
    SlotStorage,
    validate_factory_design,
)


def blank():
    return FactoryDesign(
        factory_id="Factory-A", name="空白工厂", grid=GridDesign(20, 15)
    )


def test_processing_categories_roundtrip_and_legacy_names_are_not_inferred(tmp_path):
    path = tmp_path / "categories.yaml"
    original = replace(
        blank(),
        machines=(
            MachineDesign(
                "M1",
                "Operation 4",
                Footprint(2, 2, 1, 1),
                operation_types=("operation_1", "operation_2"),
            ),
            MachineDesign(
                "M2",
                "Machine 2",
                Footprint(4, 2, 1, 1),
                operation_types=("operation_1",),
            ),
        ),
    )
    save_factory_design(path, original)
    assert load_factory_design(path)[0] == original
    legacy = yaml.safe_load(path.read_text())
    for machine in legacy["factory"]["machines"]:
        del machine["operation_types"]
    path.write_text(yaml.safe_dump(legacy))
    loaded, _ = load_factory_design(path)
    assert all(not machine.operation_types for machine in loaded.machines)
    assert loaded.machines[0].name == "Operation 4"


def test_roundtrip_retains_unicode_and_complete_explicit_data(tmp_path):
    path = tmp_path / "factory.yaml"
    original = replace(
        blank(),
        machines=(MachineDesign("M1", "第一台机器", Footprint(2, 2, 2, 3, 90)),),
        buffers=(
            BufferDesign(
                "B1",
                "Rotated storage",
                Footprint(6, 2, 3, 2, 90),
                storage=SlotStorage(slots=(SlotDesign("S1", Cell(0, 0), 3),)),
            ),
        ),
    )
    digest = save_factory_design(path, original)
    restored, actual = load_factory_design(path)
    assert restored == original
    assert actual == digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "空白工厂" in path.read_text()
    assert path.read_text().startswith("# SmartSOM factory design.")
    # The stored occupied dimensions and local coordinates are not rotated again.
    assert restored.buffers[0].footprint.width == 3
    assert restored.buffers[0].storage.slots[0].local_cell == Cell(0, 0)


def test_draft_warnings_roundtrip_without_dangling_references(tmp_path):
    design = replace(
        blank(),
        ports=(PortDesign("P1", "Unbound", Cell(0, 0)),),
        buffers=(
            BufferDesign(
                "B1",
                "Unassigned pre",
                Footprint(2, 2, 2, 2),
                role="machine_pre",
                storage=PoolStorage(capacity=None),
            ),
        ),
    )
    assert any(i.severity == "warning" for i in validate_factory_design(design))
    path = tmp_path / "draft.yml"
    save_factory_design(path, design)
    assert load_factory_design(path)[0] == design


def test_save_requires_current_digest_and_keeps_external_changes(tmp_path):
    path = tmp_path / "factory.yaml"
    digest = save_factory_design(path, blank())
    with pytest.raises(ConfigurationError, match="expected_digest"):
        save_factory_design(path, blank())
    updated = replace(blank(), name="Updated")
    new_digest = save_factory_design(path, updated, expected_digest=digest)
    assert load_factory_design(path) == (updated, new_digest)
    external = path.read_bytes() + b"# external edit\n"
    path.write_bytes(external)
    with pytest.raises(ConfigurationError, match="changed externally"):
        save_factory_design(path, blank(), expected_digest=new_digest)
    assert path.read_bytes() == external
    path.unlink()
    with pytest.raises(ConfigurationError, match="removed externally"):
        save_factory_design(path, blank(), expected_digest=new_digest)
    assert not path.exists()


def test_replace_failure_preserves_original_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "factory.yaml"
    digest = save_factory_design(path, blank())
    before = path.read_bytes()

    def fail(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(ConfigurationError, match="replacement failed"):
        save_factory_design(
            path, replace(blank(), name="Changed"), expected_digest=digest
        )
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_create_does_not_overwrite_a_concurrent_creator(tmp_path, monkeypatch):
    path = tmp_path / "factory.yaml"
    original_link = os.link

    def concurrent_link(source, target):
        target.write_text("external file")
        original_link(source, target)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(ConfigurationError):
        save_factory_design(path, blank())
    assert path.read_text() == "external file"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_design_never_replaces_or_creates_file(tmp_path):
    path = tmp_path / "factory.yaml"
    design = replace(blank(), ports=(PortDesign("P1", "Outside", Cell(20, 0)),))
    with pytest.raises(ConfigurationError):
        save_factory_design(path, design)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "nested_unknown",
        "v1",
        "float_grid",
        "bool_grid",
        "bad_role",
        "outside",
    ],
)
def test_strict_file_validation(tmp_path, change):
    path = tmp_path / "factory.yaml"
    save_factory_design(path, blank())
    data = yaml.safe_load(path.read_text())
    if change == "unknown":
        data["surprise"] = True
    elif change == "nested_unknown":
        data["factory"]["grid"]["meters_per_cell"] = 1
    elif change == "v1":
        data["schema"] = "smartsom.factory/v1"
    elif change in {"float_grid", "bool_grid"}:
        data["factory"]["grid"]["width"] = 20.0 if change == "float_grid" else True
    elif change == "bad_role":
        data["factory"]["buffers"] = [
            {
                "buffer_id": "B1",
                "name": "B",
                "footprint": {"x": 1, "y": 1, "width": 2, "height": 2},
                "role": "holding",
            }
        ]
    else:
        data["factory"]["ports"] = [
            {"port_id": "P1", "name": "P", "cell": {"x": 20, "y": 0}}
        ]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError):
        load_factory_design(path)


def test_duplicate_keys_and_wrong_file_extension_rejected(tmp_path):
    path = tmp_path / "factory.yaml"
    path.write_text("schema: smartsom.factory/v2\nschema: smartsom.factory/v2\n")
    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_factory_design(path)
    with pytest.raises(ConfigurationError, match="yaml"):
        save_factory_design(tmp_path / "factory.json", blank())


def test_v2_never_silently_enters_v1_runtime(tmp_path):
    path = tmp_path / "factory.yaml"
    save_factory_design(path, blank())
    with pytest.raises(ConfigurationError):
        read_model(path, FactoryFile)


@pytest.mark.parametrize("rotation", [False, True, 0.0, 90.0, "90"])
def test_file_rotation_requires_an_actual_integer(tmp_path, rotation):
    path = tmp_path / "factory.yaml"
    design = replace(
        blank(), machines=(MachineDesign("M1", "M", Footprint(1, 1, 2, 2)),)
    )
    save_factory_design(path, design)
    data = yaml.safe_load(path.read_text())
    data["factory"]["machines"][0]["footprint"]["rotation"] = rotation
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError):
        load_factory_design(path)


def test_oversized_out_of_bounds_rectangle_is_not_enumerated(monkeypatch):
    import smartsom.domain.factory_design as module

    design = replace(
        blank(), machines=(MachineDesign("M1", "Huge", Footprint(0, 0, 10**6, 10**6)),)
    )

    def forbidden_enumeration(footprint):
        pytest.fail(
            "Out-of-bounds dimensions must be rejected before enumerating cells"
        )

    monkeypatch.setattr(module, "occupied_cells", forbidden_enumeration)
    issues = validate_factory_design(design)
    assert any(
        i.code == "footprint_out_of_bounds" and i.severity == "error" for i in issues
    )
