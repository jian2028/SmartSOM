"""Factory catalogs describe capabilities; authoring preferences are separate."""

from dataclasses import replace

import pytest
import yaml

from smartsom.config.codec import ConfigurationError
from smartsom.config.factory_design import (
    FactoryAuthoring,
    FactoryDesignFile,
    load_factory_design,
    load_factory_design_file,
    save_factory_design,
    save_factory_design_file,
)
from smartsom.domain.factory_design import (
    Footprint,
    MachineDesign,
    validate_factory_design,
)
from smartsom.studio import editing
from smartsom.studio.document import blank_design


def four_machines():
    design = blank_design()
    for n in range(4):
        design, _ = editing.create_resource(design, "machine", n * 2, 1)
    return design


def test_defaults_shared_capabilities_unused_types_and_stable_ids():
    design = four_machines()
    assert design.operation_types == tuple(f"operation_{n}" for n in range(1, 5))
    assert [m.operation_types for m in design.machines] == [
        (t,) for t in design.operation_types
    ]
    first = design.machines[0]
    design = editing.replace_resources(
        design, {first.machine_id: replace(first, operation_types=("operation_2",))}
    )
    assert not validate_factory_design(design)
    design = editing.remove_operation_type(design, "operation_1")
    assert design.operation_types == ("operation_2", "operation_3", "operation_4")
    with pytest.raises(ValueError, match="machine_001, machine_002"):
        editing.remove_operation_type(design, "operation_2")
    design = editing.add_operation_type(design)
    assert design.operation_types[-1] == "operation_5"
    design = editing.delete(design, ("machine_004",))
    assert design.operation_types[-2:] == ("operation_4", "operation_5")


def test_manual_creation_and_reenabling_auto_keep_existing_choices():
    design = replace(blank_design(), operation_types=("operation_1", "custom"))
    design, _ = editing.create_resource(design, "machine", 1, 1, catalog_mode="manual")
    candidate, _ = editing.create_resource(
        design, "machine", 3, 1, catalog_mode="manual", operation_types=("custom",)
    )
    assert candidate.operation_types == design.operation_types
    assert candidate.machines[-1].operation_types == ("custom",)
    automatic = editing.auto_operation_catalog(candidate)
    assert automatic.operation_types == ("operation_1", "operation_2", "custom")
    assert automatic.machines == candidate.machines


def test_document_roundtrip_and_design_only_update_preserve_manual_mode(tmp_path):
    path = tmp_path / "factory.yaml"
    envelope = FactoryDesignFile(
        schema="smartsom.factory/v2",
        factory=four_machines(),
        authoring=FactoryAuthoring(operation_catalog_mode="manual"),
    )
    digest = save_factory_design_file(path, envelope)
    assert load_factory_design_file(path) == (envelope, digest)
    changed = replace(envelope.factory, name="Updated")
    digest = save_factory_design(path, changed, expected_digest=digest)
    restored, actual = load_factory_design_file(path)
    assert actual == digest and restored.factory == changed
    assert restored.authoring == envelope.authoring
    assert load_factory_design(path) == (changed, digest)


def test_legacy_catalog_migration_is_read_only_and_never_assigns_capabilities(tmp_path):
    path = tmp_path / "old.yaml"
    save_factory_design(path, four_machines())
    data = yaml.safe_load(path.read_text())
    del data["factory"]["operation_types"]
    del data["authoring"]
    data["factory"]["machines"][0]["operation_types"] = ["drilling", "operation_2"]
    del data["factory"]["machines"][1]["operation_types"]
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    envelope, _ = load_factory_design_file(path)
    assert path.read_bytes() == original
    assert envelope.factory.operation_types == (
        "operation_1",
        "operation_2",
        "operation_3",
        "operation_4",
        "drilling",
    )
    assert envelope.factory.machines[0].operation_types == ("drilling", "operation_2")
    assert envelope.factory.machines[1].operation_types == ()
    assert envelope.authoring.operation_catalog_mode == "auto"
    assert [i.code for i in validate_factory_design(envelope.factory)] == [
        "unspecified_machine_capability"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["factory"].update(operation_types=["operation_1"]),
        lambda d: d["factory"].update(operation_types=["operation_1", "operation_1"]),
        lambda d: d["factory"].update(operation_types=["bad id"]),
        lambda d: d["factory"]["machines"][0].update(unknown=1),
        lambda d: d["authoring"].update(operation_catalog_mode="other"),
        lambda d: d["factory"]["machines"][0]["footprint"].update(width=True),
    ],
)
def test_explicit_catalog_and_nested_fields_are_strict(tmp_path, mutation):
    path = tmp_path / "factory.yaml"
    save_factory_design(path, four_machines())
    data = yaml.safe_load(path.read_text())
    mutation(data)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError):
        load_factory_design_file(path)


def test_empty_capability_is_warning_but_unknown_reference_is_error():
    machine = MachineDesign("M", "Machine", Footprint(1, 1, 1, 1))
    design = replace(blank_design(), machines=(machine,))
    assert [(i.severity, i.code) for i in validate_factory_design(design)] == [
        ("warning", "unspecified_machine_capability")
    ]
    design = replace(design, machines=(replace(machine, operation_types=("missing",)),))
    assert [(i.severity, i.code) for i in validate_factory_design(design)] == [
        ("error", "unknown_operation_type")
    ]
