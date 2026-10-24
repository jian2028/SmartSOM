"""Fresh route descriptors for native immutable steps; extension-safe fallback."""

import copy
from dataclasses import asdict, fields

from smartsom.domain.production import ProductionStep

_NAMES = tuple(field.name for field in fields(ProductionStep))
_DESCRIPTORS = {name: vars(ProductionStep)[name] for name in _NAMES}
_MISSING = object()
_HOOKS = {
    name: vars(ProductionStep).get(name, _MISSING)
    for name in (
        "__getattribute__",
        "__getattr__",
        "__deepcopy__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
    )
}
_DEEPCOPY = copy.deepcopy
_DISPATCH = {
    kind: copy._deepcopy_dispatch[kind] for kind in (type(None), str, int, tuple)
}


def step_descriptor(step):
    """Avoid recursive dataclass reflection only for exact scalar native fields.

    No descriptor cache survives a call. Inspect current field content even when
    a frozen instance was mutated through object.__setattr__. New mutable dicts
    are returned each time; extension fields and changed hooks use asdict.
    """
    attributes = vars(ProductionStep)
    if (
        type(step) is not ProductionStep
        or ProductionStep.__bases__ != (object,)
        or any(
            attributes.get(name, _MISSING) is not hook for name, hook in _HOOKS.items()
        )
        or any(
            attributes.get(name) is not descriptor
            for name, descriptor in _DESCRIPTORS.items()
        )
        or tuple(field.name for field in fields(step)) != _NAMES
        or copy.deepcopy is not _DEEPCOPY
        or type(copy._deepcopy_dispatch) is not dict
        or type(copy.dispatch_table) is not dict
        or any(
            copy._deepcopy_dispatch.get(kind) is not handler
            or kind in copy.dispatch_table
            for kind, handler in _DISPATCH.items()
        )
    ):
        return asdict(step)
    operation_id = step.operation_id
    operation_type = step.operation_type
    nominal = step.nominal_ticks
    machine_ticks = step.machine_nominal_ticks
    reference = step.reference_ticks
    if (
        type(operation_id) is not str
        or type(operation_type) is not str
        or type(nominal) is not int
        or (reference is not None and type(reference) is not str)
        or type(machine_ticks) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not int
            for pair in machine_ticks
        )
    ):
        return asdict(step)
    return dict(
        operation_id=operation_id,
        operation_type=operation_type,
        nominal_ticks=nominal,
        machine_nominal_ticks=machine_ticks,
        reference_ticks=reference,
    )
