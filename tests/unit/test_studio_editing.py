"""Static document transactions preserve spatial identities and references."""

from dataclasses import replace

import pytest

from smartsom.domain.factory_design import (
    BufferSlotTarget,
    Cell,
    GridDesign,
    PoolStorage,
    PortBinding,
    PortDesign,
    SlotDesign,
    SlotStorage,
    entity_id,
)
from smartsom.studio import editing as e
from smartsom.studio.document import blank_design


@pytest.fixture
def design():
    d, machine = e.create_resource(blank_design(), "machine", 1, 1, 2, 2)
    d, buffer = e.create_resource(d, "buffer", 4, 1, 2, 2)
    b = replace(d.buffers[0], machine_id=machine, role="machine_pre")
    d = e.replace_resources(d, {buffer: b})
    return replace(
        d,
        ports=(
            PortDesign(
                "port_001",
                "Access",
                Cell(4, 4),
                bindings=(PortBinding(BufferSlotTarget(buffer, "slot_004")),),
            ),
        ),
    )


@pytest.mark.parametrize("kind", list(e.COLLECTIONS))
def test_create_all_kinds_default_one_cell(kind):
    d, identifier = e.create_resource(blank_design(), kind, 2, 3)
    r = e.resource(d, identifier)
    assert e.rect_of(r) == (2, 3, 1, 1)
    if (slots := e.slots_of(r)) is not None:
        assert slots == (SlotDesign("slot_001", Cell(0, 0), 1),)
    assert e.checked(d) is d


def test_create_and_move_reject_collisions_without_changing_document(design):
    original = design
    with pytest.raises(ValueError):
        e.create_resource(design, "buffer", 1, 1)
    with pytest.raises(ValueError):
        e.move(design, ("buffer_001",), -3, 0)
    with pytest.raises(ValueError):
        e.create_resource(design, "buffer", 0, 0, 10**8, 10**8)
    assert design == original
    moved = e.move(design, ("machine_001",), 0, 5)
    assert moved.buffers == design.buffers and moved.ports == design.ports


def test_expand_preserves_holes_and_ids_shrink_prunes_only_removed_targets(design):
    d = e.set_slots(design, "buffer_001", design.buffers[0].storage.slots[1:])
    expanded = e.resize(d, "buffer_001", 3, 2)
    slots = expanded.buffers[0].storage.slots
    assert slots[:3] == d.buffers[0].storage.slots
    assert Cell(0, 0) not in {s.local_cell for s in slots}
    assert {s.slot_id for s in slots[-2:]} == {"slot_005", "slot_006"}
    assert expanded.ports == d.ports
    shrunk = e.resize(expanded, "buffer_001", 2, 1)
    assert not shrunk.ports[0].bindings
    assert any("slot_004" in text for text in e.impact(expanded, shrunk))


def test_rename_rewrites_owner_and_target_id_atomically(design):
    d = e.rename(design, "machine_001", "machine_A")
    assert d.buffers[0].machine_id == "machine_A"
    d = e.rename(d, "buffer_001", "buffer_A")
    assert d.ports[0].bindings[0].target == BufferSlotTarget("buffer_A", "slot_004")
    slots = tuple(
        replace(s, slot_id="last") if s.slot_id == "slot_004" else s
        for s in d.buffers[0].storage.slots
    )
    d = e.set_slots(d, "buffer_A", slots, {"slot_004": "last"})
    assert d.ports[0].bindings[0].target == BufferSlotTarget("buffer_A", "last")
    with pytest.raises(ValueError):
        e.rename(d, "buffer_A", "machine_A")


def test_factory_and_resource_rename_use_separate_identity_namespaces(design):
    d = replace(design, factory_id="machine_001")
    assert e.checked(d) is d
    renamed = e.rename(d, "machine_001", "machine_A")
    assert renamed.factory_id == "machine_001"
    assert renamed.machines[0].machine_id == "machine_A"
    assert renamed.buffers[0].machine_id == "machine_A"
    factory = e.rename(d, None, "factory_A")
    assert factory.factory_id == "factory_A"
    assert factory.machines == d.machines


def test_delete_machine_keeps_buffer_and_delete_slot_removes_reference(design):
    d = e.delete(design, ("machine_001",))
    assert len(d.buffers) == 1 and d.buffers[0].machine_id is None
    assert d.ports == design.ports
    d = e.delete(d, ("buffer_001",))
    assert not d.ports[0].bindings


def test_rotation_preserves_slot_references_and_four_turns_restore(design):
    d = design
    for _ in range(4):
        d = e.rotate(d, ("machine_001", "buffer_001", "port_001"))
    assert d == design
    d = e.rotate(design, ("buffer_001",))
    assert d.ports == design.ports
    assert d.buffers[0].storage.slots[-1].local_cell == Cell(0, 1)


def test_clipboard_rewires_internal_and_clears_external_references(design):
    copied = e.copy_selection(design, ("machine_001", "buffer_001", "port_001"))
    target = replace(blank_design(), grid=GridDesign(40, 30))
    d, identifiers = e.paste(target, copied, 10, 10)
    assert len(identifiers) == 3
    assert d.buffers[0].machine_id == d.machines[0].machine_id
    assert d.ports[0].bindings[0].target.buffer_id == d.buffers[0].buffer_id
    second, new_ids = e.paste(design, copied[1:], 10, 8)
    assert not set(new_ids) & {entity_id(r) for r in copied}
    assert second.buffers[-1].machine_id is None
    port_only, _ = e.paste(design, copied[-1:], 10, 8)
    assert not port_only.ports[-1].bindings


@pytest.mark.parametrize("capacity", [0, None, 7])
def test_storage_conversion_accepts_explicit_zero_or_unlimited(design, capacity):
    d = e.convert_storage(design, "buffer_001", "pool", capacity)
    assert d.buffers[0].storage == PoolStorage(capacity)
    assert not d.ports[0].bindings
    d = e.convert_storage(d, "buffer_001", "slots", 3)
    assert isinstance(d.buffers[0].storage, SlotStorage)
    assert sum(s.capacity for s in d.buffers[0].storage.slots) == 12
    with pytest.raises(ValueError):
        e.convert_storage(d, "buffer_001", "slots", 0)
