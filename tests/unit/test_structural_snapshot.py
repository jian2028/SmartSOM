"""Runtime snapshots preserve deepcopy values, aliases, and rollback semantics."""

import copy
import pickle
from dataclasses import dataclass, replace
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
