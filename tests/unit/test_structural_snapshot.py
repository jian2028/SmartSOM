"""Runtime snapshots preserve deepcopy values, aliases, and rollback semantics."""

import copy
import copyreg
import os
import pickle
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom import api
from smartsom.domain.factory_design import Cell, GridDesign
from smartsom.domain.production import ProductionStep
from smartsom.domain.travel_time import TravelTimeMatrix
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.transaction_state import TransactionState


def copy_runtime_state(state):
    return TransactionState().capture(SimpleNamespace(**state)).values


@dataclass(frozen=True)
class ExtendedCell(Cell):
    values: list


@dataclass(frozen=True)
class UnknownFrozen:
    values: list


class CopyingGrid(GridDesign):
    __slots__ = ()
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        result = type(self)(self.width, self.height, self.blocked_cells)
        memo[id(self)] = result
        return result


class CopyingKey(str):
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        return str(self)


class CopyingMatrix(TravelTimeMatrix):
    __slots__ = ()
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        result = type(self)(self.points, self.times, self.source)
        memo[id(self)] = result
        return result


def test_configuration_and_dynamic_aliases_copied():
    step = ProductionStep("op", "operation_1", 1)
    complete = {"finished"}
    jobs = {"job": {"location": "input", "payload": [1]}}
    state = dict(
        demands={"step": step},
        completed=complete,
        shipped=complete,
        jobs=jobs,
        alias=jobs["job"],
    )
    saved = copy_runtime_state(state)
    assert saved == copy.deepcopy(state)
    assert saved["demands"] is not state["demands"]
    assert saved["demands"]["step"] is not step
    assert saved["completed"] is saved["shipped"]
    assert saved["completed"] is not complete
    assert saved["alias"] is saved["jobs"]["job"]
    jobs["job"]["payload"].append(2)
    assert saved["alias"]["payload"] == [1]


def test_valid_frozen_input_with_mutable_nested_pair_is_not_shared():
    pair = ["machine", 3]
    step = ProductionStep("op", "operation_1", 1, (pair,))
    assert step.machine_nominal_ticks[0] is pair
    state = {"demands": {"step": step}, "alias": pair}
    saved = copy_runtime_state(state)
    assert saved == copy.deepcopy(state)
    assert saved["demands"]["step"] is not step
    assert saved["demands"]["step"].machine_nominal_ticks[0] is saved["alias"]
    pair[1] = 9
    assert saved["alias"] == ["machine", 3]


def test_unknown_frozen_extension_and_cycles_keep_deepcopy_behavior():
    unknown = UnknownFrozen([1])
    cycle = []
    cycle.append(cycle)
    state = {"demands": {"unknown": unknown}, "cycle": cycle}
    saved = copy_runtime_state(state)
    assert saved["demands"]["unknown"] is not unknown
    assert saved["demands"]["unknown"].values is not unknown.values
    assert saved["cycle"] is not cycle
    assert saved["cycle"][0] is saved["cycle"]


def test_corrupted_known_cycle_falls_back_without_recursing_forever():
    cell = Cell(0, 0)
    object.__setattr__(cell, "x", cell)
    saved = copy_runtime_state({"factory": cell})
    assert saved["factory"] is not cell
    assert saved["factory"].x is saved["factory"]


@pytest.fixture
def sim():
    root = Path(__file__).resolve().parents[2]
    p = api.prepare(
        api.load_config(root / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    return ProductionSimulator(p.scenario, contract="v3")


def test_boundary_snapshot_copies_factory_and_restores_aliases(sim):
    original = sim.snapshot()
    factory = sim.factory
    scenario = sim.scenario
    sim.protocol.begin()
    assert sim.protocol.backup.values["factory"] is factory
    # Configuration retains the original deepcopy contract.
    assert sim.protocol.backup.values["scenario"] == scenario
    assert sim.protocol.backup.values["storage"] is not sim.storage
    sim.metrics["uncommitted"] += 1
    sim.completed.add("uncommitted")
    sim.protocol.abort()
    assert sim.snapshot() == original
    assert sim.shipped is sim.completed
    assert sim.factory is factory


def test_configuration_replacement_is_rechecked_next_boundary(sim):
    sim.protocol.begin()
    sim.protocol.abort()
    changed = replace(sim.scenario, reward_time_scale=211)
    sim.scenario = changed
    machine = next(iter(sim.machines))
    old = sim.machines[machine]
    sim.machines[machine] = replace(old, name="replacement")
    sim.protocol.begin()
    assert sim.protocol.backup.values["scenario"].reward_time_scale == 211
    assert sim.protocol.backup.values["machines"][machine] is sim.machines[machine]
    sim.protocol.abort()
    assert sim.scenario.reward_time_scale == 211
    assert sim.machines[machine].name == "replacement"


def controller(sim):
    from smartsom.algorithms.production_composition import BoundaryCoordinator
    from smartsom.algorithms.production_rules import RulePolicy

    names = dict(
        machine="normal_first",
        buffer="edd",
        dispatcher="nearest",
        mover="automatic_travel",
    )
    return BoundaryCoordinator(
        sim,
        {role: RulePolicy(role, name) for role, name in names.items()},
        {role: {"default": role} for role in names},
    )


@pytest.mark.parametrize(
    "phase",
    ["begin", "resolve_machines", "accept_proposals", "prepare_services", "step"],
)
@pytest.mark.parametrize("extension", [False, True])
def test_exception_in_each_open_phase_restores_complete_dynamic_graph(
    sim, monkeypatch, phase, extension
):
    import random

    if extension:
        sim.extension_payload = {"values": [1], "rng": random.Random(8)}
        sim.extension_alias = sim.extension_payload["values"]
        rng_before = sim.extension_payload["rng"].getstate()
    before, private = sim.snapshot(), copy.deepcopy(sim.jobs)
    target = sim if phase == "step" else sim.protocol
    original = getattr(type(target), phase)

    def fail(instance, *args, **kwargs):
        original(instance, *args, **kwargs)
        assert sim.protocol.backup.fast_path is (not extension)
        sim.completed.add("uncommitted")
        sim.metrics["uncommitted"] += 1
        if extension:
            sim.extension_payload["values"].append(2)
            sim.extension_payload["rng"].random()
        job = next(iter(sim.jobs))
        sim.jobs[job]["defective"] = not sim.jobs[job]["defective"]
        raise RuntimeError("injected transaction failure")

    monkeypatch.setattr(type(target), phase, fail)
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        controller(sim).tick()
    assert sim.snapshot() == before
    assert sim.jobs == private
    if extension:
        assert sim.extension_payload["values"] == [1]
        assert sim.extension_alias is sim.extension_payload["values"]
        assert sim.extension_payload["rng"].getstate() == rng_before
    assert sim.shipped is sim.completed
    assert sim.protocol.stage == "closed"
    assert sim.protocol.backup is None
    sim._check()


def test_stage_only_then_real_tick_matches_fresh_reset(sim):
    before = sim.snapshot()
    c = controller(sim)
    c.tick(stage_only=True)
    assert sim.snapshot() == before
    assert sim.shipped is sim.completed
    fresh = ProductionSimulator(sim.scenario, contract="v3")
    assert c.tick() == controller(fresh).tick()


def test_known_class_subclass_with_mutable_extension_uses_fallback():
    value = ExtendedCell(0, 0, [1])
    saved = copy_runtime_state({"factory": value})
    assert saved["factory"] == value
    assert saved["factory"] is not value
    assert saved["factory"].values is not value.values


def test_open_transaction_pickle_abort_preserves_dynamic_aliases(sim):
    before, jobs = sim.snapshot(), copy.deepcopy(sim.jobs)
    sim.protocol.begin()
    sim.completed.add("uncommitted")
    sim.metrics["uncommitted"] += 1
    restored = pickle.loads(pickle.dumps(sim))
    assert restored.protocol.backup.values["factory"] is restored.factory
    restored.protocol.abort()
    assert restored.snapshot() == before
    assert restored.jobs == jobs
    assert restored.completed is restored.shipped
    restored._check()
    sim.protocol.abort()


def test_replacement_with_mutable_nested_input_is_not_reused(sim):
    sim.protocol.begin()
    sim.protocol.abort()
    key = next(iter(sim.demands))
    demand = sim.demands[key]
    pair = [next(iter(sim.machines)), 3]
    step = replace(demand.steps[0], machine_nominal_ticks=(pair,))
    sim.demands[key] = replace(demand, steps=(step, *demand.steps[1:]))
    sim.protocol.begin()
    saved_step = sim.protocol.backup.values["demands"][key].steps[0]
    assert saved_step is not step
    assert saved_step.machine_nominal_ticks[0] is not pair
    pair[1] = 5
    sim.protocol.abort()
    assert sim.demands[key].steps[0].machine_nominal_ticks[0] == [pair[0], 3]


@pytest.mark.parametrize("unknown_first", [False, True])
def test_known_unknown_alias_cycles_and_rng(unknown_first):
    import random

    shared = []
    pair = (shared,)
    shared.extend([shared, pair])
    rng = random.Random(7)
    state = {"jobs": {"cycle": shared, "rng": rng}, "unknown": shared, "rng": rng}
    if unknown_first:
        state = dict(reversed(tuple(state.items())))
    state["events"] = [state]
    out = copy_runtime_state(state)
    assert out["jobs"]["cycle"] is out["unknown"]
    assert out["unknown"][0] is out["unknown"]
    assert out["unknown"][1][0] is out["unknown"]
    assert out["events"][0]["jobs"] is out["jobs"]
    assert out["rng"] is out["jobs"]["rng"]
    assert out["rng"] is not rng
    assert out["rng"].getstate() == rng.getstate()


def test_container_subclass_deepcopy_hook_is_honored():
    class CustomDict(dict):
        def __deepcopy__(self, memo):
            result = type(self)(copied=True)
            memo[id(self)] = result
            return result

    value = CustomDict(original=True)
    out = copy_runtime_state({"jobs": value, "unknown": value})
    assert out["jobs"] == {"copied": True}
    assert out["jobs"] is out["unknown"]


def test_custom_hook_memo_override_is_respected_for_atomic_values():
    original = "shared value"

    class Hook:
        def __deepcopy__(self, memo):
            memo[id(original)] = "replacement"
            return "hook"

    state = {"unknown": Hook(), "jobs": [original]}
    assert copy_runtime_state(state) == copy.deepcopy(state)


def test_custom_hook_cannot_replace_completed_container_memo():
    parent = []

    class Hook:
        def __deepcopy__(self, memo):
            memo[id(parent)] = ["replacement"]
            return "copied_hook"

    parent.append(Hook())
    state = {"jobs": parent, "unknown": parent}
    out = copy_runtime_state(state)
    assert out == copy.deepcopy(state)
    assert out["jobs"] is out["unknown"]


def test_static_plan_reused_and_mutable_tables_restored(sim):
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    old_plan = sim.protocol.state_plan.design
    # Commit/stage progression is not needed for this snapshot contract check.
    first = sim.protocol.backup.values
    assert first["demands"] is not sim.demands
    assert first["capacity"] is not sim.capacity
    key = next(iter(sim.capacity))
    assert first["capacity"][key] is not sim.capacity[key]
    sim.capacity[key]["new"] = 7
    sim.demands.clear()
    sim.added_during_transaction = [1]
    sim.protocol.abort()
    assert "new" not in sim.capacity[key]
    assert sim.demands
    assert not hasattr(sim, "added_during_transaction")
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    assert sim.protocol.state_plan.design is not old_plan  # restored shell identities
    sim.protocol.abort()
    plan = sim.protocol.state_plan
    a = plan.capture(sim)
    compiled = plan.design
    b = plan.capture(sim)
    assert a.fast_path and b.fast_path
    assert plan.design is compiled


def test_known_runtime_cross_field_alias_and_cycles_preserved(sim):
    job = next(iter(sim.jobs.values()))
    job["self"] = job
    sim._phase_events = [job]
    sim.events = sim._phase_events
    sim.protocol.begin()
    out = sim.protocol.backup.values
    assert sim.protocol.backup.fast_path
    assert out["events"] is out["_phase_events"]
    assert out["events"][0] is out["jobs"][next(iter(sim.jobs))]
    assert out["events"][0] is not job
    assert out["events"][0]["self"] is out["events"][0]
    sim.protocol.abort()
    restored = sim.jobs[next(iter(sim.jobs))]
    assert restored["self"] is restored
    assert sim.events[0] is restored


def test_static_class_change_invalidates_cached_proof_and_calls_copy_hook(sim):
    plan = sim.protocol.state_plan
    assert plan.capture(sim).fast_path
    grid = sim.factory.grid
    object.__setattr__(grid, "__class__", CopyingGrid)
    CopyingGrid.copies = 0
    saved = plan.capture(sim)
    assert not saved.fast_path
    assert CopyingGrid.copies == 1
    assert saved.values["factory"].grid is not grid
    assert type(saved.values["factory"].grid) is CopyingGrid


def test_open_boundary_class_change_recovers_captured_native_class(sim):
    before = sim.snapshot()
    grid = sim.factory.grid
    sim.protocol.begin()
    object.__setattr__(grid, "__class__", CopyingGrid)
    CopyingGrid.copies = 0
    sim.protocol.abort()
    assert sim.snapshot() == before
    assert type(sim.factory.grid) is GridDesign
    assert sim.factory.grid is not grid
    assert sim.factory is sim.scenario.factory
    assert CopyingGrid.copies == 0


def test_state_key_extension_uses_original_graph_copy(sim):
    sim.__dict__[CopyingKey("events")] = sim.__dict__.pop("events")
    CopyingKey.copies = 0
    saved = sim.protocol.state_plan.capture(sim)
    assert not saved.fast_path
    assert CopyingKey.copies == 1
    assert all(type(key) is str for key in saved.values)


def test_opaque_matrix_class_change_keeps_capture_time_copy_contract(sim):
    matrix = sim.scenario.transport_matrix
    sim._phase_events = [matrix]
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    object.__setattr__(matrix, "__class__", CopyingMatrix)
    CopyingMatrix.copies = 0
    sim.protocol.abort()
    # Its original deepcopy retained this exact opaque object.
    assert sim.scenario.transport_matrix is matrix
    assert sim._phase_events[0] is matrix
    assert CopyingMatrix.copies == 0
    saved = sim.protocol.state_plan.capture(sim)
    assert not saved.fast_path
    # The next capture must use the extension's now-active deepcopy hook.
    assert saved.values["scenario"].transport_matrix is not matrix
    assert type(saved.values["scenario"].transport_matrix) is CopyingMatrix
    assert saved.values["_phase_events"][0] is saved.values["scenario"].transport_matrix
    assert CopyingMatrix.copies == 1


def test_unknown_nested_runtime_uses_baseline_fallback(sim):
    import random

    job = next(iter(sim.jobs.values()))
    job["extension_rng"] = random.Random(12)
    sim.protocol.begin()
    assert not sim.protocol.backup.fast_path
    out = sim.protocol.backup.values["jobs"][next(iter(sim.jobs))]
    assert out["extension_rng"] is not job["extension_rng"]
    assert out["extension_rng"].getstate() == job["extension_rng"].getstate()
    sim.protocol.abort()


def test_same_tick_invalid_action_can_abort_without_state_change(sim):
    before = copy.deepcopy({k: v for k, v in sim.__dict__.items() if k != "protocol"})
    sim.protocol.begin()
    with pytest.raises((ValueError, TypeError)):
        sim.protocol.prepare_services({}, ())
    sim.protocol.abort()
    assert {k: v for k, v in sim.__dict__.items() if k != "protocol"} == before
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    sim.protocol.abort()


def test_open_boundary_frozen_bypass_edit_recovers_old_static_graph(sim):
    before = copy.deepcopy({k: v for k, v in sim.__dict__.items() if k != "protocol"})
    old_grid = sim.factory.grid
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    object.__setattr__(old_grid, "width", old_grid.width + 1)
    sim.protocol.abort()
    assert {k: v for k, v in sim.__dict__.items() if k != "protocol"} == before
    assert sim.factory.grid is not old_grid
    assert sim.factory is sim.scenario.factory


def test_counter_extension_attributes_trigger_whole_graph_fallback(sim):
    import random

    sim.metrics.extension_rng = random.Random(12)
    saved = sim.protocol.state_plan.capture(sim)
    assert not saved.fast_path
    assert vars(saved.values["metrics"]) == vars(copy.deepcopy(sim.metrics))


@pytest.mark.parametrize("opened", [False, True])
def test_legacy_protocol_pickle_restores_and_can_begin(sim, opened):
    before = sim.snapshot()
    sim.protocol.ports_for("unused", "pickup")
    assert sim.protocol._port_lookup is not None
    if opened:
        sim.protocol.backup = copy.deepcopy(
            {k: v for k, v in sim.__dict__.items() if k != "protocol"}
        )
        sim.protocol.stage = "proposals"
        sim.tick += 1
    del sim.protocol.state_plan
    del sim.protocol._port_lookup
    restored = pickle.loads(pickle.dumps(sim))
    if opened:
        restored.protocol.abort()
    assert restored.snapshot() == before
    restored.protocol.begin()
    assert restored.protocol.backup.fast_path
    restored.protocol.abort()


def test_deleted_frozen_field_during_open_boundary_recovers(sim):
    width = sim.factory.grid.width
    sim.protocol.begin()
    object.__delattr__(sim.factory.grid, "width")
    sim.protocol.abort()
    assert sim.factory.grid.width == width


def test_frozen_recovery_preserves_travel_matrix_tuple_alias(sim):
    matrix = sim.scenario.transport_matrix
    assert matrix is not None
    sim._phase_events = [matrix.points, matrix.times]
    before = copy.deepcopy({k: v for k, v in sim.__dict__.items() if k != "protocol"})
    sim.protocol.begin()
    assert sim.protocol.backup.fast_path
    object.__setattr__(sim.factory.grid, "width", sim.factory.grid.width + 1)
    sim.protocol.abort()
    assert sim._phase_events[0] is sim.scenario.transport_matrix.points
    assert sim._phase_events[1] is sim.scenario.transport_matrix.times
    assert {k: v for k, v in sim.__dict__.items() if k != "protocol"} == before


def test_opaque_matrix_keeps_baseline_copy_contract_for_mutable_internals(sim):
    from smartsom.domain.travel_time import TravelTimeMatrix

    matrix = sim.scenario.transport_matrix
    mutable = TravelTimeMatrix(list(matrix.points), list(matrix.times), matrix.source)
    sim.scenario = replace(sim.scenario, transport_matrix=mutable)
    sim._phase_events = [mutable.points]
    expected = copy.deepcopy({k: v for k, v in sim.__dict__.items() if k != "protocol"})
    saved = sim.protocol.state_plan.capture(sim)
    assert saved.fast_path
    assert saved.values["scenario"].transport_matrix is mutable
    assert saved.values["_phase_events"][0] is not mutable.points
    assert saved.values == expected


def _core_state(sim):
    return {key: value for key, value in vars(sim).items() if key != "protocol"}


def _install_grid_hook(patch, name, calls):
    def changed_copy(value, memo):
        calls.append(name)
        result = GridDesign(value.width + 1, value.height, value.blocked_cells)
        memo[id(value)] = result
        return result

    def reduce(value, *args):
        calls.append(name)
        return GridDesign, (value.width + 1, value.height, value.blocked_cells)

    def getstate(value):
        calls.append(name)
        return [value.width + 1, value.height, value.blocked_cells]

    def setstate(value, state):
        calls.append(name)
        for field, child in zip(("width", "height", "blocked_cells"), state):
            object.__setattr__(value, field, child + 1 if field == "width" else child)

    def newargs(value):
        calls.append(name)
        return ((), {}) if name == "__getnewargs_ex__" else ()

    def getattribute(value, field):
        if field == "__deepcopy__":
            return lambda memo: changed_copy(value, memo)
        return object.__getattribute__(value, field)

    def getattr_missing(value, field):
        if field == "__deepcopy__":
            return lambda memo: changed_copy(value, memo)
        raise AttributeError(field)

    hooks = {
        "__deepcopy__": changed_copy,
        "__reduce_ex__": reduce,
        "__reduce__": reduce,
        "__getstate__": getstate,
        "__setstate__": setstate,
        "__getnewargs__": newargs,
        "__getnewargs_ex__": newargs,
        "__getattribute__": getattribute,
        "__getattr__": getattr_missing,
    }
    patch.setattr(GridDesign, name, hooks[name], raising=False)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "hook",
    [
        "__deepcopy__",
        "__reduce_ex__",
        "__reduce__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getattribute__",
        "__getattr__",
    ],
)
def test_native_copy_protocol_changes_use_whole_graph_fallback(
    sim, monkeypatch, warm, hook
):
    plan = TransactionState()
    sim._phase_events = [sim.factory.grid]
    if warm:
        assert plan.capture(sim).fast_path
    calls = []
    with monkeypatch.context() as patch:
        _install_grid_hook(patch, hook, calls)
        expected = copy.deepcopy(_core_state(sim))
        expected_calls = list(calls)
        calls.clear()
        saved = plan.capture(sim)
        assert not saved.fast_path
        assert expected_calls and calls == expected_calls
        assert saved.values == expected
        assert saved.values["_phase_events"][0] is saved.values["factory"].grid
    assert plan.capture(sim).fast_path


def _install_dispatch(patch, kind, calls):
    original = copy._deepcopy_dispatch.get(kind)

    def changed(value, memo):
        calls.append(kind.__name__)
        if original is not None:
            return original(value, memo)
        # Delegate the test hook's result to that type's regular copy protocol.
        del copy._deepcopy_dispatch[kind]
        try:
            return copy.deepcopy(value, memo)
        finally:
            copy._deepcopy_dispatch[kind] = changed

    patch.setitem(copy._deepcopy_dispatch, kind, changed)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        GridDesign,
        dict,
        list,
        tuple,
        int,
        float,
        str,
        bytes,
        bool,
        type(None),
        Decimal,
        Fraction,
        Counter,
        type,
    ],
)
def test_copy_dispatch_changes_cover_static_nodes_and_scalar_rows(
    sim, monkeypatch, warm, kind
):
    row = next(iter(sim.jobs.values()))
    row.update(decimal=Decimal("1.25"), fraction=Fraction(2, 3), byte_value=b"native")
    sim._phase_events = [row, sim.factory.grid]
    plan = TransactionState()
    if warm:
        assert plan.capture(sim).fast_path
    calls = []
    with monkeypatch.context() as patch:
        _install_dispatch(patch, kind, calls)
        expected = copy.deepcopy(_core_state(sim))
        expected_calls = list(calls)
        calls.clear()
        saved = plan.capture(sim)
        assert not saved.fast_path
        assert expected_calls and calls == expected_calls
        assert saved.values == expected
        assert saved.values["_phase_events"][0] is next(
            iter(saved.values["jobs"].values())
        )
    assert plan.capture(sim).fast_path


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("kind", [GridDesign, Counter, set, frozenset])
def test_copyreg_reductions_use_original_copy_order_and_aliases(
    sim, monkeypatch, warm, kind
):
    sim._phase_events = [
        sim.factory.grid,
        sim.metrics,
        sim.completed,
        frozenset({3, 4}),
    ]
    plan = TransactionState()
    if warm:
        assert plan.capture(sim).fast_path
    calls = []

    def reducer(value):
        calls.append(kind.__name__)
        if kind is GridDesign:
            return kind, (value.width + 1, value.height, value.blocked_cells)
        return kind, (dict(value) if kind is Counter else tuple(value),)

    with monkeypatch.context() as patch:
        patch.setitem(copyreg.dispatch_table, kind, reducer)
        expected = copy.deepcopy(_core_state(sim))
        expected_calls = list(calls)
        calls.clear()
        saved = plan.capture(sim)
        assert not saved.fast_path
        assert expected_calls and calls == expected_calls
        assert saved.values == expected
        assert saved.values["_phase_events"][1] is saved.values["metrics"]
        assert saved.values["_phase_events"][2] is saved.values["completed"]
    assert plan.capture(sim).fast_path


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("kind", [Fraction, TravelTimeMatrix])
def test_native_self_copy_method_replacement_is_not_blessed(
    sim, monkeypatch, warm, kind
):
    original = Fraction(2, 3) if kind is Fraction else sim.scenario.transport_matrix
    row = next(iter(sim.jobs.values()))
    if kind is Fraction:
        row["fraction"] = original
    sim._phase_events = [original]
    plan = TransactionState()
    if warm:
        assert plan.capture(sim).fast_path
    calls = []

    def changed(value, memo):
        calls.append(kind.__name__)
        result = (
            Fraction(value.numerator + 1, value.denominator)
            if kind is Fraction
            else TravelTimeMatrix(value.points, value.times, value.source)
        )
        memo[id(value)] = result
        return result

    with monkeypatch.context() as patch:
        patch.setattr(kind, "__deepcopy__", changed)
        expected = copy.deepcopy(_core_state(sim))
        expected_calls = list(calls)
        calls.clear()
        saved = plan.capture(sim)
        assert not saved.fast_path
        assert expected_calls and calls == expected_calls
        assert saved.values == expected
        assert saved.values["_phase_events"][0] is not original
    assert plan.capture(sim).fast_path


@pytest.mark.parametrize(
    "hook",
    ["__deepcopy__", "__reduce_ex__", "__reduce__", "__getstate__", "__setstate__"],
)
def test_copy_hooks_added_after_capture_only_apply_to_next_capture(
    sim, monkeypatch, hook
):
    plan = TransactionState()
    before = sim.factory.grid.width
    saved = plan.capture(sim)
    assert saved.fast_path
    calls = []
    with monkeypatch.context() as patch:
        _install_grid_hook(patch, hook, calls)
        object.__setattr__(sim.factory.grid, "width", before + 7)
        saved.restore_into(sim)
        assert sim.factory.grid.width == before
        assert calls == []
        assert not plan.capture(sim).fast_path
        assert calls
    assert plan.capture(sim).fast_path


@pytest.mark.parametrize(
    "kind", [dict, list, tuple, int, Fraction, Counter, GridDesign, TravelTimeMatrix]
)
def test_late_dispatch_does_not_recopy_materialized_rollback(sim, monkeypatch, kind):
    row = next(iter(sim.jobs.values()))
    row["fraction"] = Fraction(2, 3)
    cycle = []
    linked = (cycle,)
    cycle.append(linked)
    sim._phase_events = [row, cycle, sim.metrics, frozenset({3, 4}), sim.completed]
    width = sim.factory.grid.width
    plan = TransactionState()
    saved = plan.capture(sim)
    assert saved.fast_path
    calls = []
    with monkeypatch.context() as patch:
        _install_dispatch(patch, kind, calls)
        object.__setattr__(sim.factory.grid, "width", width + 7)
        saved.restore_into(sim)
        assert calls == []
        assert sim.factory.grid.width == width
        assert sim._phase_events[0] is next(iter(sim.jobs.values()))
        assert sim._phase_events[1][0][0] is sim._phase_events[1]
        assert sim._phase_events[2] is sim.metrics
        assert sim._phase_events[3] == frozenset({3, 4})
        assert sim._phase_events[4] is sim.completed is sim.shipped
        assert not plan.capture(sim).fast_path
        assert calls
    assert plan.capture(sim).fast_path


def test_changed_dispatch_memo_overrides_are_not_hidden_by_scalar_row_preseeding(
    sim, monkeypatch
):
    row = next(iter(sim.jobs.values()))
    scalar = row["fraction"] = Fraction(2, 3)
    original = copy._deepcopy_dispatch[list]

    def changed(value, memo):
        memo[id(scalar)] = "replacement"
        return original(value, memo)

    monkeypatch.setitem(copy._deepcopy_dispatch, list, changed)
    expected = copy.deepcopy(_core_state(sim))
    saved = TransactionState().capture(sim)
    assert not saved.fast_path
    assert saved.values == expected
    assert next(iter(saved.values["jobs"].values()))["fraction"] == "replacement"


def test_native_allocation_hook_fallback_and_late_recovery_are_isolated():
    # CPython's allocation slot can remain changed after assigning/removing
    # __new__. Exercise this supported hook in a subprocess, not a shared class
    # later fixtures will instantiate.
    source = """
import copy
from types import SimpleNamespace
from smartsom.domain.factory_design import GridDesign
from smartsom.engine.transaction_state import TransactionState
for warm in (False, True):
    grid = object.__new__(GridDesign)
    for name, value in (("width", 3), ("height", 4), ("blocked_cells", ())):
        object.__setattr__(grid, name, value)
    core = SimpleNamespace(scenario=None, factory=grid, demands={}, machines={},
        stations={}, buffers={}, scrap={}, ports={}, protocol=None)
    plan = TransactionState()
    saved = plan.capture(core) if warm else None
    calls = []
    def allocate(cls, *args, **kwargs):
        calls.append("allocate")
        return object.__new__(cls)
    GridDesign.__new__ = staticmethod(allocate)
    try:
        expected = copy.deepcopy({k: v for k, v in vars(core).items() if k != "protocol"})
        expected_calls = list(calls)
        calls.clear()
        actual = plan.capture(core)
        assert not actual.fast_path and actual.values == expected
        assert expected_calls and calls == expected_calls
        if saved is not None:
            calls.clear()
            object.__setattr__(grid, "width", 17)
            saved.restore_into(core)
            assert core.factory.width == 3 and not calls
    finally:
        del GridDesign.__new__
    assert plan.capture(core).fast_path
"""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )


@pytest.mark.parametrize("kind", [GridDesign, Counter])
def test_late_copyreg_registration_only_changes_next_capture(sim, monkeypatch, kind):
    plan = TransactionState()
    saved = plan.capture(sim)
    assert saved.fast_path
    width = sim.factory.grid.width
    calls = []

    def reducer(value):
        calls.append(kind.__name__)
        if kind is GridDesign:
            return kind, (value.width + 1, value.height, value.blocked_cells)
        return kind, (dict(value),)

    with monkeypatch.context() as patch:
        # copy.deepcopy reads this active table, even when a caller replaces it
        # separately from the original copyreg.dispatch_table dictionary.
        patch.setattr(copy, "dispatch_table", {**copy.dispatch_table, kind: reducer})
        object.__setattr__(sim.factory.grid, "width", width + 7)
        saved.restore_into(sim)
        assert sim.factory.grid.width == width
        assert calls == []
        assert not plan.capture(sim).fast_path
        assert calls
    assert plan.capture(sim).fast_path


def test_replaced_deepcopy_entrypoint_keeps_one_whole_graph_call(sim, monkeypatch):
    original, calls = copy.deepcopy, []

    def changed(value, memo=None):
        calls.append(type(value))
        return original(value, memo)

    with monkeypatch.context() as patch:
        patch.setattr(copy, "deepcopy", changed)
        saved = TransactionState().capture(sim)
        assert not saved.fast_path
        assert calls == [dict]
        assert saved.values == original(_core_state(sim))


def test_late_copy_attribute_hook_cannot_intercept_rollback_field_guards(
    sim, monkeypatch
):
    plan = TransactionState()
    saved = plan.capture(sim)
    assert saved.fast_path
    width = sim.factory.grid.width
    calls = []

    def changed(value, field):
        calls.append(field)
        if field == "width":
            raise RuntimeError("new attribute behavior must not run during abort")
        return object.__getattribute__(value, field)

    with monkeypatch.context() as patch:
        patch.setattr(GridDesign, "__getattribute__", changed)
        saved.restore_into(sim)
        assert calls == []
        assert object.__getattribute__(sim.factory.grid, "width") == width
        with pytest.raises(RuntimeError, match="new attribute behavior"):
            plan.capture(sim)
    assert plan.capture(sim).fast_path


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("hook", ["__getnewargs__", "__getnewargs_ex__", "__getattr__"])
def test_noncallable_hook_is_not_confused_with_absence(sim, monkeypatch, warm, hook):
    plan = TransactionState()
    if warm:
        assert plan.capture(sim).fast_path
    with monkeypatch.context() as patch:
        patch.setattr(GridDesign, hook, None, raising=False)
        with pytest.raises(TypeError) as expected:
            copy.deepcopy(_core_state(sim))
        with pytest.raises(TypeError) as actual:
            plan.capture(sim)
        assert str(actual.value) == str(expected.value)
    assert plan.capture(sim).fast_path
