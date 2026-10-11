"""Guarded port lookup follows mapping, nested-field and decoder changes."""

import copy
import pickle
from dataclasses import dataclass, replace

import pytest
from test_port_target_decode import ExtendedBufferTarget
from test_production_runtime import small_scenario

from smartsom.domain.factory_design import (
    BufferSlotTarget,
    BufferTarget,
    Cell,
    PortBinding,
    PortDesign,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.production_protocol import ProductionProtocol


def protocol():
    result = ProductionProtocol.__new__(ProductionProtocol)
    result.core = ProductionSimulator.__new__(ProductionSimulator)
    result.ports = {
        key: PortDesign(
            key,
            key,
            Cell(index, 0),
            bindings=(
                PortBinding(BufferSlotTarget("owner", "a")),
                PortBinding(BufferSlotTarget("owner", "b"), ("pickup",)),
            ),
        )
        for index, key in enumerate(("first", "second", "third"))
    }
    return result


def original_ports_for(result, owner, operation):
    return tuple(
        p
        for p in result.ports.values()
        if any(
            result.core._target(b.target)[0] == owner and operation in b.operations
            for b in p.bindings
        )
    )


def check(result):
    for owner in ("owner", "other", "missing"):
        for operation in ("pickup", "drop_off", "charge", "missing"):
            expected = original_ports_for(result, owner, operation)
            actual = result.ports_for(owner, operation)
            assert actual == expected
            assert all(a is b for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize(
    "change", ["remove", "clear", "reverse", "replace", "reinsert", "alias", "dict"]
)
def test_index_tracks_direct_port_mapping_changes(change):
    result = protocol()
    check(result)
    if change == "remove":
        result.ports.pop("second")
    elif change == "clear":
        result.ports.clear()
    elif change == "reverse":
        result.ports = dict(reversed(tuple(result.ports.items())))
    elif change == "replace":
        result.ports["second"] = replace(
            result.ports["second"],
            bindings=(
                PortBinding(BufferSlotTarget("other", "changed"), ("drop_off",)),
            ),
        )
    elif change == "reinsert":
        result.ports["first"] = result.ports.pop("first")
    elif change == "alias":
        result.ports["alias"] = result.ports["first"]
    else:
        result.ports = dict(result.ports)
    check(result)
    result.ports.update(protocol().ports)
    check(result)


def test_initial_empty_ports_and_per_port_binding_deduplication():
    result = protocol()
    ports = result.ports
    result.ports = {}
    check(result)
    result.ports = ports
    check(result)
    assert tuple(p.port_id for p in result.ports_for("owner", "pickup")) == (
        "first",
        "second",
        "third",
    )


def test_protocol_subclass_constructor_and_empty_scan_leave_cache_hook_untouched():
    class CustomProtocol(ProductionProtocol):
        @property
        def _port_lookup(self):
            raise RuntimeError("extension cache property must not be inspected")

    core = ProductionSimulator(small_scenario())
    result = CustomProtocol(core)
    assert result.ports_for("input", "pickup") == original_ports_for(
        result, "input", "pickup"
    )
    result.ports.clear()
    assert result.ports_for("input", "pickup") == ()


def test_unknown_existing_cache_field_keeps_legacy_scan_without_matches_hook():
    class Cache:
        def matches(self, ports):
            raise RuntimeError("unrecognized extension cache must not be used")

    result = protocol()
    cache = result._port_lookup = Cache()
    check(result)
    result.ports.clear()
    assert result.ports_for("owner", "pickup") == ()
    assert result._port_lookup is cache


@pytest.mark.parametrize("clone", [copy.deepcopy, pickle.loads])
def test_index_clone_and_lazy_restore_keep_detached_mapping(clone):
    result = protocol()
    check(result)
    restored = clone(pickle.dumps(result) if clone is pickle.loads else result)
    check(restored)
    restored.ports.pop("first")
    check(restored)
    check(result)
    assert "first" in result.ports
    del restored._port_lookup
    check(restored)


def test_stable_topology_reuses_one_index_without_decoding(monkeypatch):
    import smartsom.engine.port_lookup as lookup

    result = protocol()
    check(result)
    saved = result._port_lookup

    def fail(*args):
        pytest.fail("a native lookup must not call asdict")

    monkeypatch.setattr(lookup, "asdict", fail)
    for _ in range(20):
        check(result)
    assert result._port_lookup is saved
    result.ports["first"] = replace(result.ports["first"], name="replacement")
    check(result)
    assert result._port_lookup is not saved


@pytest.mark.parametrize(
    "field", ["bindings", "target", "operations", "owner", "slot", "kind"]
)
def test_index_rechecks_frozen_field_bypass(field):
    result = protocol()
    check(result)
    port = result.ports["first"]
    binding = port.bindings[0]
    if field == "bindings":
        object.__setattr__(port, "bindings", ())
    elif field == "target":
        object.__setattr__(binding, "target", BufferTarget("other"))
    elif field == "operations":
        object.__setattr__(binding, "operations", ("charge",))
    elif field == "owner":
        object.__setattr__(binding.target, "buffer_id", "other")
    elif field == "slot":
        object.__setattr__(binding.target, "slot_id", "changed")
    else:
        object.__setattr__(binding.target, "kind", "changed")
    previous = result._port_lookup
    check(result)
    assert result._port_lookup is not previous


def test_unhashable_extension_and_nested_mutable_owner_fall_back():
    result = protocol()
    check(result)
    target = ExtendedBufferTarget("owner", [1])
    port = result.ports["first"]
    result.ports["first"] = replace(port, bindings=(PortBinding(target),))
    check(result)
    assert result._port_lookup is None
    target.payload.append(2)
    object.__setattr__(target, "buffer_id", ["owner"])
    for owner in (["owner"], "owner", "other"):
        assert result.ports_for(owner, "pickup") == original_ports_for(
            result, owner, "pickup"
        )
    target.buffer_id.append("other")
    assert result.ports_for(["owner", "other"], "pickup") == (result.ports["first"],)


def test_native_target_with_mutable_operations_observes_in_place_changes():
    result = protocol()
    check(result)
    binding = result.ports["first"].bindings[0]
    operations = ["charge"]
    object.__setattr__(binding, "operations", operations)
    check(result)
    assert result._port_lookup is None
    operations[:] = ["drop_off"]
    check(result)


@dataclass(frozen=True)
class ExtendedBinding(PortBinding):
    payload: list = None


@dataclass(frozen=True)
class ExtendedPort(PortDesign):
    payload: list = None


@pytest.mark.parametrize("extension", ["port", "binding", "mapping"])
def test_unknown_record_or_mapping_types_use_legacy_scan(extension):
    result = protocol()
    check(result)
    if extension == "port":
        result.ports["first"] = ExtendedPort(
            "first", "first", Cell(0, 0), bindings=result.ports["first"].bindings
        )
    elif extension == "binding":
        object.__setattr__(
            result.ports["first"],
            "bindings",
            (ExtendedBinding(BufferTarget("other")),),
        )
    else:

        class Mapping(dict):
            def values(self):
                return reversed(tuple(super().values()))

        result.ports = Mapping(result.ports)
    check(result)
    assert result._port_lookup is None


def test_dynamic_decoder_override_keeps_call_order_and_live_result():
    result = protocol()
    check(result)
    native, calls, owner = result.core._target, [], ["other"]

    def decode(target):
        calls.append(target)
        return owner[0], "pool"

    result.core._target = decode
    assert result.ports_for("other", "pickup") == tuple(result.ports.values())
    assert calls == [port.bindings[0].target for port in result.ports.values()]
    owner[0] = "owner"
    check(result)
    result.core._target = native
    check(result)


@pytest.mark.parametrize("kind", ["property", "getattribute", "native_property"])
def test_dynamic_decoder_descriptors_keep_access_order_and_empty_scan(
    monkeypatch, kind
):
    result = protocol()
    result.ports = {"first": result.ports["first"]}
    accesses = []

    def descriptor(instance):
        accesses.append(len(accesses) + 1)
        owner = f"owner_{accesses[-1]}"
        return lambda target: (owner, "pool")

    class DescriptorCore:
        _target = property(descriptor)

    class AttributeCore:
        def __getattribute__(self, name):
            if name == "_target":
                return descriptor(self)
            return object.__getattribute__(self, name)

    if kind == "property":
        result.core = DescriptorCore()
    elif kind == "getattribute":
        result.core = AttributeCore()
    else:
        monkeypatch.setattr(ProductionSimulator, "_target", property(descriptor))
    assert result.ports_for("owner_1", "pickup") == tuple(result.ports.values())
    assert accesses == [1]
    result.ports.clear()
    assert result.ports_for("missing", "pickup") == ()
    assert accesses == [1]


def test_empty_scan_does_not_touch_raising_decoder():
    result = protocol()
    result.ports.clear()

    class Core:
        @property
        def _target(self):
            raise RuntimeError("must not inspect the decoder for an empty scan")

    result.core = Core()
    assert result.ports_for("owner", "pickup") == ()


def test_native_core_dictionary_extension_does_not_invoke_get_hook():
    result = protocol()

    class State(dict):
        def get(self, *args):
            raise RuntimeError("instance attribute lookup must not call dict.get")

    result.core.__dict__ = State(result.core.__dict__)
    check(result)
    assert not hasattr(result, "_port_lookup")


def test_instance_dictionary_nonstring_collision_keeps_legacy_hook_order():
    result = protocol()
    port = result.ports["first"]
    calls = []

    class Key:
        def __hash__(self):
            return hash("_target")

        def __eq__(self, other):
            calls.append(other)
            return False

    result.core.__dict__[Key()] = "unrelated"
    result.ports = {}
    assert result.ports_for("owner", "pickup") == ()
    assert calls == []
    result.ports = {"first": port}
    expected = original_ports_for(result, "owner", "pickup")
    expected_calls = list(calls)
    calls.clear()
    assert result.ports_for("owner", "pickup") == expected
    assert calls == expected_calls
    assert not hasattr(result, "_port_lookup")


@pytest.mark.parametrize("record", ["port", "binding", "target"])
def test_same_layout_class_bypass_invalidates_native_proof(record):
    result = protocol()
    check(result)
    port = result.ports["first"]
    binding = port.bindings[0]
    value = {"port": port, "binding": binding, "target": binding.target}[record]

    class Extension(type(value)):
        __slots__ = ()

    object.__setattr__(value, "__class__", Extension)
    check(result)
    assert result._port_lookup is None


def test_same_layout_extension_added_field_keeps_deepcopy_hook():
    result = protocol()
    result.ports = {"first": result.ports["first"]}
    target = result.ports["first"].bindings[0].target
    check(result)
    copies = []

    class Payload:
        def __deepcopy__(self, memo):
            copies.append("copy")
            return self

    @dataclass(frozen=True)
    class Extension(BufferSlotTarget):
        __slots__ = ()
        payload: object = Payload()

    object.__setattr__(target, "__class__", Extension)
    assert result.ports_for("owner", "pickup") == tuple(result.ports.values())
    assert copies == ["copy"]
    assert result._port_lookup is None


def test_class_level_decoder_override_does_not_use_native_index(monkeypatch):
    sim = ProductionSimulator(small_scenario(), contract="v3")
    monkeypatch.setattr(
        ProductionSimulator, "_target", staticmethod(lambda target: ("other", "pool"))
    )
    for operation in ("pickup", "drop_off"):
        assert sim.protocol.ports_for("other", operation) == original_ports_for(
            sim.protocol, "other", operation
        )


def test_nonstring_queries_keep_equality_and_do_not_require_hashing():
    result = protocol()
    check(result)

    class EqualString(str):
        __hash__ = None

    for owner in ([], EqualString("owner")):
        for operation in ([], EqualString("pickup")):
            assert result.ports_for(owner, operation) == original_ports_for(
                result, owner, operation
            )


def test_unknown_later_binding_preserves_short_circuit_and_original_error():
    result = protocol()
    result.ports = {"first": result.ports["first"]}
    check(result)
    object.__setattr__(result.ports["first"].bindings[1], "target", object())
    assert result.ports_for("owner", "pickup") == tuple(result.ports.values())
    with pytest.raises(TypeError):
        original_ports_for(result, "other", "pickup")
    with pytest.raises(TypeError):
        result.ports_for("other", "pickup")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("open_boundary", [False, True])
def test_real_simulator_pickle_abort_and_fresh_reset(legacy, open_boundary):
    sim = ProductionSimulator(small_scenario(), contract="v3")
    check(sim.protocol)
    before = sim.snapshot()
    if open_boundary:
        sim.protocol.begin()
        sim.metrics["uncommitted"] += 1
        if legacy:
            sim.protocol.backup = copy.deepcopy(sim.protocol.backup.values)
    if legacy:
        del sim.protocol._port_lookup
    restored = pickle.loads(pickle.dumps(sim))
    if open_boundary:
        restored.protocol.abort()
        sim.protocol.abort()
    assert restored.snapshot() == before
    check(restored.protocol)
    for owner in ("input", "machine", "output"):
        for operation in ("pickup", "drop_off"):
            assert restored.protocol.ports_for(owner, operation) == original_ports_for(
                restored.protocol, owner, operation
            )
    fresh = ProductionSimulator(sim.scenario, contract="v3")
    check(sim.protocol)
    assert fresh.protocol._port_lookup is not sim.protocol._port_lookup
    assert fresh.snapshot() == sim.snapshot()


def test_abort_rechecks_protocol_ports_without_changing_scratch_ownership():
    sim = ProductionSimulator(small_scenario(), contract="v3")
    port = sim.protocol.ports["in_port"]
    sim.protocol.ports_for("input", "pickup")
    sim.protocol.begin()
    object.__setattr__(port.bindings[0].target, "buffer_id", "other")
    assert sim.protocol.ports_for("other", "pickup") == (port,)
    sim.protocol.abort()
    # Historically abort restores the core graph, not protocol port scratch.
    assert sim.protocol.ports["in_port"] is port
    assert sim.protocol.ports_for("other", "pickup") == original_ports_for(
        sim.protocol, "other", "pickup"
    )
    assert sim.protocol.ports_for("input", "pickup") == original_ports_for(
        sim.protocol, "input", "pickup"
    )
