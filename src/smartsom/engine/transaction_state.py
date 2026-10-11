"""Explicit static-design / runtime transaction boundary.

Static plans contain proven immutable records and existing self-copying leaves.
Public mapping shells and all runtime containers still belong to the snapshot.
Unsupported extension state takes the original single-graph deepcopy path.
"""

import copy
import dataclasses
from collections import Counter
from dataclasses import dataclass, fields
from decimal import Decimal
from fractions import Fraction

from smartsom.domain import factory_design as design
from smartsom.domain import production
from smartsom.domain.quality import QualityMode
from smartsom.domain.travel_time import TravelTimeMatrix

# Exact concrete classes only: frozen subclasses and unknown extension types
# retain deepcopy's existing behavior. Frozen alone does not prove immutability.
_INPUT_TYPES = (
    design.Cell,
    design.Footprint,
    design.GridDesign,
    design.SlotDesign,
    design.PoolStorage,
    design.SlotStorage,
    design.MachineDesign,
    design.BufferDesign,
    design.InspectionStationDesign,
    design.ScrapBinDesign,
    design.ChargerDesign,
    design.BatteryDesign,
    design.AGVDesign,
    design.MachineTarget,
    design.BufferTarget,
    design.BufferSlotTarget,
    design.InspectionSlotTarget,
    design.ScrapBinTarget,
    design.ChargerTarget,
    design.PortBinding,
    design.PortDesign,
    design.FactoryDesign,
    QualityMode,
    production.ProductionStep,
    production.Demand,
    production.Outage,
    production.ProcessingSample,
    production.QualitySample,
    production.ProductionScenario,
    TravelTimeMatrix,
)
_FIELDS = {
    cls: tuple(field.name for field in fields(cls))
    for cls in _INPUT_TYPES
    if cls.__dataclass_params__.frozen
}
_ATOMIC_TYPES = (type(None), bool, int, float, str, bytes, Decimal, Fraction)
_INPUT_MAPPINGS = ("demands", "machines", "stations", "buffers", "scrap", "ports")


_RUNTIME_FIELDS = frozenset(
    (
        "tick",
        "events",
        "jobs",
        "completed",
        "shipped",
        "shipment_times",
        "_qualified_shipments",
        "_overdue_time",
        "_last_reward_components",
        "attempts",
        "released",
        "queue",
        "total_reward",
        "metrics",
        "storage",
        "machine_state",
        "station_state",
        "agvs",
        "rankings",
        "_phase_events",
    )
)
_LOOKUP_FIELDS = frozenset(
    (
        "demands",
        "processing_samples",
        "quality_samples",
        "machines",
        "stations",
        "buffers",
        "scrap",
        "capacity",
        "roles",
        "pre",
        "post",
        "ports",
        "solids",
    )
)
_KNOWN_FIELDS = _RUNTIME_FIELDS | _LOOKUP_FIELDS | {"scenario", "factory"}
_ATOMIC = frozenset(_ATOMIC_TYPES)
_MISSING = object()

# Trust module-initial stdlib implementations and the native matrix self-copy
# method, not a class's state when a plan first compiles. Runtime hook/registry
# changes decline the optimization; arbitrary edits to copy's internals or these
# reference functions are outside this contract.
_DEEPCOPY = copy.deepcopy
_HOOK_NAMES = (
    "__deepcopy__",
    "__reduce_ex__",
    "__reduce__",
    "__getstate__",
    "__setstate__",
    "__getnewargs__",
    "__getnewargs_ex__",
    "__new__",
    "__getattribute__",
    "__getattr__",
)
_OBJECT_HOOKS = {name: vars(object).get(name, _MISSING) for name in _HOOK_NAMES}
_DESIGN_HOOKS = {
    **_OBJECT_HOOKS,
    "__getstate__": dataclasses._dataclass_getstate,
    "__setstate__": dataclasses._dataclass_setstate,
}
_MATRIX_COPY = TravelTimeMatrix.__deepcopy__


def _raw_hook(kind, name):
    # Do not invoke a newly installed descriptor merely to select a copy path.
    return next(
        (vars(base)[name] for base in kind.__mro__ if name in vars(base)), _MISSING
    )


_STDLIB_HOOKS = tuple(
    (kind, tuple((name, _raw_hook(kind, name)) for name in _HOOK_NAMES))
    for kind in (Fraction, Counter)
)
_COPY_TYPES = tuple(
    dict.fromkeys(
        (
            *_INPUT_TYPES,
            *_ATOMIC_TYPES,
            dict,
            list,
            tuple,
            set,
            frozenset,
            Counter,
            type,
        )
    )
)
_DISPATCH = {
    **{
        kind: copy._deepcopy_atomic
        for kind in (type(None), bool, int, float, str, bytes, type)
    },
    dict: copy._deepcopy_dict,
    list: copy._deepcopy_list,
    tuple: copy._deepcopy_tuple,
}


def _native_copy_semantics():
    """Check once per capture/type, before bypassing any deepcopy operation."""
    dispatch, reductions = copy._deepcopy_dispatch, copy.dispatch_table
    if (
        copy.deepcopy is not _DEEPCOPY
        or type(dispatch) is not dict
        or type(reductions) is not dict
    ):
        return False
    # Registrations can mutate a shared memo, even when the registered object
    # itself would not be shared. Include runtime containers and scalar rows.
    if any(
        dispatch.get(kind) is not _DISPATCH.get(kind)
        or reductions.get(kind) is not None
        for kind in _COPY_TYPES
    ):
        return False
    for kind in _INPUT_TYPES:
        if kind.__bases__ != (object,):
            return False
        attributes = vars(kind)
        for name, expected in _DESIGN_HOOKS.items():
            if kind is TravelTimeMatrix and name == "__deepcopy__":
                expected = _MATRIX_COPY
            if attributes.get(name, _OBJECT_HOOKS[name]) is not expected:
                return False
    return all(
        _raw_hook(kind, name) is expected
        for kind, hooks in _STDLIB_HOOKS
        for name, expected in hooks
    )


@dataclass(slots=True)
class StaticDesign:
    """Compiled input-copy contract with guards for public replacements."""

    roots: tuple
    nodes: tuple
    guards: tuple

    @classmethod
    def compile(cls, state):
        roots, nodes, guards, active, proven = [], [], [], set(), set()

        def immutable(value):
            kind = type(value)
            if kind in _ATOMIC:
                return True
            identity = id(value)
            if kind is TravelTimeMatrix:
                # Baseline defines this exact class as an opaque self-copy leaf.
                # Do not index its potentially large matrix internals.
                if identity not in proven:
                    proven.add(identity)
                    nodes.append(value)
                    guards.append((value, kind, (), ()))
                return True
            if identity in proven:
                return True
            if identity in active:
                return False
            if kind is tuple:
                children = value
            elif kind in _FIELDS:
                children = tuple(getattr(value, name) for name in _FIELDS[kind])
            else:
                return False
            active.add(identity)
            okay = all(immutable(child) for child in children)
            active.remove(identity)
            if okay:
                proven.add(identity)
                nodes.append(value)
                if kind in _FIELDS:
                    guards.append((value, kind, _FIELDS[kind], children))
            return okay

        for name in ("scenario", "factory"):
            value = state.get(name)
            if not immutable(value):
                return None
            roots.append((name, value, None))
        for name in _INPUT_MAPPINGS:
            value = state.get(name)
            if type(value) is not dict:
                return None
            entries = tuple(value.items())
            if not all(immutable(k) and immutable(v) for k, v in entries):
                return None
            roots.append((name, value, entries))
        return cls(tuple(roots), tuple(nodes), tuple(guards))

    def current(self, state):
        for name, value, entries in self.roots:
            if state.get(name) is not value:
                return False
            if entries is not None:
                if len(value) != len(entries):
                    return False
                if any(
                    k is not old_k or v is not old_v
                    for (k, v), (old_k, old_v) in zip(value.items(), entries)
                ):
                    return False
        return self.unchanged_nodes()

    def unchanged_nodes(self):
        # Guards also retain preimages for exceptional rollback materialization.
        return all(
            type(value) is kind for value, kind, names, children in self.guards
        ) and all(
            getattr(value, name, _MISSING) is child
            for value, kind, names, children in self.guards
            for name, child in zip(names, children)
        )

    def recovery_memo(self):
        """Rebuild captured frozen fields only if bypass edits occurred mid-tick."""
        records = {
            id(value): (kind, names, children)
            for value, kind, names, children in self.guards
        }
        memo = {}

        def restore(value):
            identity = id(value)
            if identity in memo:
                return memo[identity]
            record = records.get(identity)
            if record is not None:
                kind, names, children = record
                if kind is TravelTimeMatrix:
                    # Baseline shares this opaque leaf even across bypass edits.
                    memo[identity] = value
                    return value
                result = object.__new__(kind)
                memo[identity] = result
                for name, child in zip(names, children):
                    object.__setattr__(result, name, restore(child))
            elif type(value) in _ATOMIC:
                return value
            else:
                restored = tuple(restore(child) for child in value)
                result = (
                    value if all(a is b for a, b in zip(value, restored)) else restored
                )
            memo[identity] = result
            return result

        for value in self.nodes:
            restore(value)
        return memo

    def memo(self):
        # Derive ids afresh: persisted integer ids would be stale after pickle.
        return {id(value): value for value in self.nodes}


def _plain_runtime(value, static_ids, seen):
    """Fast path contains no custom copy hooks; unknowns use the full fallback."""
    kind, identity = type(value), id(value)
    if kind in _ATOMIC or identity in static_ids or identity in seen:
        return True
    seen.add(identity)
    if kind is Counter and vars(value):
        return False
    if kind in (dict, Counter):
        return all(
            _plain_runtime(k, static_ids, seen) and _plain_runtime(v, static_ids, seen)
            for k, v in value.items()
        )
    if kind in (list, tuple, set, frozenset):
        return all(_plain_runtime(v, static_ids, seen) for v in value)
    return False


def _seed_scalar_rows(state, memo):
    """Native jobs/machines/capacity rows have immutable scalar columns."""
    for name in ("jobs", "machine_state", "capacity"):
        mapping = state.get(name)
        if type(mapping) is not dict:
            continue
        for row in mapping.values():
            if (
                type(row) is dict
                and id(row) not in memo
                and all(
                    type(k) in _ATOMIC and type(v) in _ATOMIC for k, v in row.items()
                )
            ):
                memo[id(row)] = row.copy()


def _recover_owned_graph(value, memo):
    """Replace static preimages without running hooks installed after capture.

    Fast-path runtime containers were already detached at capture. Only these
    exact builtins and self-copy/immutable leaves can occur in the saved graph.
    Tuple memo handling retains cycles through a mutable container, like deepcopy.
    """
    identity = id(value)
    if identity in memo:
        return memo[identity]
    kind = type(value)
    if kind in (dict, Counter):
        result = dict.__new__(kind)
        memo[identity] = result
        for key, child in dict.items(value):
            dict.__setitem__(
                result,
                _recover_owned_graph(key, memo),
                _recover_owned_graph(child, memo),
            )
    elif kind is list:
        result = []
        memo[identity] = result
        result.extend(_recover_owned_graph(child, memo) for child in value)
    elif kind is tuple:
        children = [_recover_owned_graph(child, memo) for child in value]
        if identity in memo:
            return memo[identity]
        result = (
            value if all(a is b for a, b in zip(value, children)) else tuple(children)
        )
    elif kind is set:
        result = set()
        memo[identity] = result
        result.update(_recover_owned_graph(child, memo) for child in value)
    elif kind is frozenset:
        result = frozenset(_recover_owned_graph(child, memo) for child in value)
    else:
        # Captured scalar/self-copy leaves keep their original identity. Their
        # currently installed copy hooks have no role in rolling back a snapshot.
        return value
    memo[identity] = result
    return result


@dataclass(slots=True)
class RuntimeSnapshot:
    """Rollback state sharing immutable records and baseline self-copy leaves."""

    values: dict
    fast_path: bool
    shared_design_nodes: int = 0
    design: StaticDesign | None = None

    def restore_into(self, core):
        protocol = core.protocol
        values = self.values
        if self.design is not None and (
            not _native_copy_semantics() or not self.design.unchanged_nodes()
        ):
            values = _recover_owned_graph(values, self.design.recovery_memo())
        core.__dict__.clear()
        core.__dict__.update(values)
        core.protocol = protocol


class TransactionState:
    """Owns the reusable static plan, never a mutable runtime snapshot cache."""

    def __init__(self):
        self.design = None

    def capture(self, core):
        state = {k: v for k, v in core.__dict__.items() if k != "protocol"}
        if (
            any(type(key) is not str for key in state)
            or not state.keys() <= _KNOWN_FIELDS
            or not _native_copy_semantics()
        ):
            return RuntimeSnapshot(copy.deepcopy(state), False)
        if self.design is None or not self.design.current(state):
            self.design = StaticDesign.compile(state)
        if self.design is None:
            return RuntimeSnapshot(copy.deepcopy(state), False)
        memo = self.design.memo()
        seen = set()
        if not all(_plain_runtime(v, memo, seen) for v in state.values()):
            return RuntimeSnapshot(copy.deepcopy(state), False)
        _seed_scalar_rows(state, memo)
        saved = {}
        memo[id(state)] = saved
        # Single memo covers mutable lookup shells and runtime together. Iterate
        # original attribute order rather than rebuilding lookup tables from config.
        for name, value in state.items():
            saved[name] = copy.deepcopy(value, memo)
        return RuntimeSnapshot(saved, True, len(self.design.nodes), self.design)
