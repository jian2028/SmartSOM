"""Native target decoding retains the previous asdict contract for extensions."""

import copy
from dataclasses import asdict, dataclass, replace

import pytest

from smartsom.domain.factory_design import (
    BufferSlotTarget,
    BufferTarget,
    ChargerTarget,
    InspectionSlotTarget,
    MachineTarget,
    PortBinding,
    ScrapBinTarget,
)
from smartsom.engine.production import ProductionSimulator


def original_target(target):
    data = asdict(target)
    owner = next(v for k, v in data.items() if k.endswith("_id") and k != "slot_id")
    return owner, data.get("slot_id", "pool")


@dataclass(frozen=True)
class ExtendedBufferTarget(BufferTarget):
    payload: list


@pytest.mark.parametrize(
    "target",
    [
        MachineTarget("same"),
        BufferTarget("same"),
        BufferSlotTarget("same", "slot"),
        InspectionSlotTarget("same", "slot"),
        ScrapBinTarget("same"),
        ChargerTarget("same"),
    ],
)
def test_native_target_decoding_matches_original(target):
    assert ProductionSimulator._target(target) == original_target(target)
    assert ProductionSimulator._target(copy.deepcopy(target)) == original_target(target)


def test_replacement_and_frozen_field_bypass_do_not_reuse_old_value():
    target = BufferSlotTarget("owner", "before")
    changed = replace(target, slot_id="after")
    assert ProductionSimulator._target(changed) == ("owner", "after")
    assert ProductionSimulator._target(target) == ("owner", "before")
    object.__setattr__(target, "buffer_id", "other")
    object.__setattr__(target, "slot_id", "changed")
    assert ProductionSimulator._target(target) == ("other", "changed")


def test_direct_binding_accepts_unhashable_target_extension():
    target = ExtendedBufferTarget("owner", [1])
    assert PortBinding(target).target is target
    with pytest.raises(TypeError):
        hash(target)
    assert ProductionSimulator._target(target) == original_target(target)
    target.payload.append(2)
    assert ProductionSimulator._target(target) == original_target(target)


def test_nonprimitive_bypass_field_retains_detachment():
    target = BufferSlotTarget("owner", "slot")
    owner, slot = ["owner"], {"nested": ["slot"]}
    object.__setattr__(target, "buffer_id", owner)
    object.__setattr__(target, "slot_id", slot)
    decoded = ProductionSimulator._target(target)
    assert decoded == original_target(target)
    assert decoded[0] is not owner
    assert decoded[1] is not slot
    owner.append("other")
    slot["nested"].append("changed")
    assert decoded == (["owner"], {"nested": ["slot"]})
    assert ProductionSimulator._target(target) == original_target(target)


def test_ignored_kind_field_keeps_custom_deepcopy_behavior():
    calls = []

    class Payload:
        def __deepcopy__(self, memo):
            calls.append("copy")
            return self

    target = BufferTarget("owner")
    object.__setattr__(target, "kind", Payload())
    assert ProductionSimulator._target(target) == ("owner", "pool")
    assert ProductionSimulator._target(target) == ("owner", "pool")
    assert calls == ["copy", "copy"]


def test_missing_native_field_preserves_original_error():
    target = BufferTarget("owner")
    object.__delattr__(target, "kind")
    with pytest.raises(AttributeError):
        original_target(target)
    with pytest.raises(AttributeError):
        ProductionSimulator._target(target)
