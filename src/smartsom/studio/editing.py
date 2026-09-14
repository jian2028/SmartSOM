"""Immutable authoring operations; no Qt or simulation state."""

import re
from dataclasses import fields, replace

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferSlotTarget,
    BufferTarget,
    Cell,
    ChargerDesign,
    ChargerTarget,
    Footprint,
    InspectionSlotTarget,
    InspectionStationDesign,
    MachineDesign,
    MachineTarget,
    PoolStorage,
    PortBinding,
    PortDesign,
    ScrapBinDesign,
    ScrapBinTarget,
    SlotDesign,
    SlotStorage,
    entity_id,
    iter_resources,
    target_cell,
    target_owner_id,
    validate_factory_design,
)

COLLECTIONS = {
    "machine": ("machines", MachineDesign, "machine_id"),
    "buffer": ("buffers", BufferDesign, "buffer_id"),
    "inspection_station": (
        "inspection_stations",
        InspectionStationDesign,
        "inspection_station_id",
    ),
    "scrap_bin": ("scrap_bins", ScrapBinDesign, "scrap_bin_id"),
    "charger": ("chargers", ChargerDesign, "charger_id"),
    "port": ("ports", PortDesign, "port_id"),
    "agv": ("agvs", AGVDesign, "agv_id"),
}


def checked(design):
    errors = [i for i in validate_factory_design(design) if i.severity == "error"]
    if errors:
        raise ValueError(
            "\n".join(f"{i.entity_id or 'Factory'}: {i.message}" for i in errors)
        )
    return design


def resource(design, identifier):
    return next(r for r in iter_resources(design) if entity_id(r) == identifier)


def kind_of(value):
    return next(k for k, (_, cls, _) in COLLECTIONS.items() if isinstance(value, cls))


def next_id(prefix, used):
    numbers = [
        int(m.group(1))
        for value in used
        if (m := re.fullmatch(re.escape(prefix) + r"_(\d+)", value))
    ]
    number = max(numbers, default=0) + 1
    while f"{prefix}_{number:03d}" in used:
        number += 1
    return f"{prefix}_{number:03d}"


def rect_of(value):
    if hasattr(value, "footprint"):
        f = value.footprint
        return f.x, f.y, f.width, f.height
    c = value.cell if isinstance(value, PortDesign) else value.initial_cell
    return c.x, c.y, 1, 1


def slots_of(value):
    if isinstance(value, InspectionStationDesign):
        return value.slots
    if isinstance(value, BufferDesign) and isinstance(value.storage, SlotStorage):
        return value.storage.slots
    return None


def with_slots(value, slots):
    if isinstance(value, InspectionStationDesign):
        return replace(value, slots=tuple(slots))
    return replace(value, storage=SlotStorage(tuple(slots)))


def replace_resources(design, replacements):
    return replace(
        design,
        **{
            collection: tuple(
                replacements.get(entity_id(r), r) for r in getattr(design, collection)
            )
            for collection, _, _ in COLLECTIONS.values()
        },
    )


def create_resource(design, kind, x, y, width=1, height=1):
    check_bounds(design, x, y, width, height)
    collection, cls, id_field = COLLECTIONS[kind]
    identifier = next_id(kind, {entity_id(r) for r in iter_resources(design)})
    name = f"{kind.replace('_', ' ').capitalize() if kind != 'agv' else 'AGV'} {int(identifier.rsplit('_', 1)[1])}"
    args = {id_field: identifier, "name": name}
    if kind == "agv":
        args["initial_cell"] = Cell(x, y)
    elif kind == "port":
        args["cell"] = Cell(x, y)
    else:
        args["footprint"] = Footprint(x, y, width, height)
    if kind in ("buffer", "inspection_station"):
        slots = tuple(
            SlotDesign(f"slot_{j * width + i + 1:03d}", Cell(i, j))
            for j in range(height)
            for i in range(width)
        )
        args["storage" if kind == "buffer" else "slots"] = (
            SlotStorage(slots) if kind == "buffer" else slots
        )
    value = cls(**args)
    return checked(
        replace(design, **{collection: (*getattr(design, collection), value)})
    ), identifier


def check_bounds(design, x, y, width, height):
    if (
        width < 1
        or height < 1
        or x < 0
        or y < 0
        or x + width > design.grid.width
        or y + height > design.grid.height
    ):
        raise ValueError("Resource footprint must fit inside the grid")


def relocate(value, x, y):
    if hasattr(value, "footprint"):
        return replace(value, footprint=replace(value.footprint, x=x, y=y))
    return replace(
        value,
        **{"cell" if isinstance(value, PortDesign) else "initial_cell": Cell(x, y)},
    )


def move(design, identifiers, dx, dy):
    replacements = {}
    for identifier in identifiers:
        r = resource(design, identifier)
        x, y, _, _ = rect_of(r)
        replacements[identifier] = relocate(r, x + dx, y + dy)
    return checked(replace_resources(design, replacements))


def rotate(design, identifiers):
    values = [resource(design, i) for i in identifiers]
    if not values:
        return design
    boxes = [rect_of(r) for r in values]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    height = max(b[1] + b[3] for b in boxes) - top
    replacements = {}
    headings = ("north", "east", "south", "west")
    for r, (x, y, w, h) in zip(values, boxes, strict=True):
        changed = relocate(r, left + height - (y - top) - h, top + x - left)
        if hasattr(r, "footprint"):
            changed = replace(
                changed,
                footprint=replace(
                    changed.footprint,
                    width=h,
                    height=w,
                    rotation=(r.footprint.rotation + 90) % 360,
                ),
            )
            if (slots := slots_of(r)) is not None:
                changed = with_slots(
                    changed,
                    [
                        replace(
                            s, local_cell=Cell(h - 1 - s.local_cell.y, s.local_cell.x)
                        )
                        for s in slots
                    ],
                )
        elif isinstance(r, AGVDesign):
            changed = replace(
                changed,
                initial_heading=headings[(headings.index(r.initial_heading) + 1) % 4],
            )
        elif isinstance(r, PortDesign):
            changed = replace(
                changed,
                allowed_headings=tuple(
                    headings[(headings.index(hd) + 1) % 4] for hd in r.allowed_headings
                ),
            )
        replacements[entity_id(r)] = changed
    return checked(replace_resources(design, replacements))


def prune_bindings(design):
    return replace(
        design,
        ports=tuple(
            replace(
                p,
                bindings=tuple(
                    b for b in p.bindings if target_cell(design, b.target) is not None
                ),
            )
            for p in design.ports
        ),
    )


def resize_resource(value, width, height):
    old = value.footprint
    result = replace(value, footprint=replace(old, width=width, height=height))
    if (slots := slots_of(value)) is not None:
        retained = [
            s for s in slots if s.local_cell.x < width and s.local_cell.y < height
        ]
        used = {s.slot_id for s in slots}
        for y in range(height):
            for x in range(width):
                if x >= old.width or y >= old.height:
                    identifier = next_id("slot", used)
                    retained.append(SlotDesign(identifier, Cell(x, y)))
                    used.add(identifier)
        result = with_slots(result, retained)
    return result


def update_resource(design, old, new):
    """Property edits reconcile slot geometry, retaining holes and identities."""
    check_bounds(design, *rect_of(new))
    if hasattr(old, "footprint") and slots_of(old) is not None:
        if (old.footprint.width, old.footprint.height) != (
            new.footprint.width,
            new.footprint.height,
        ):
            resized = resize_resource(old, new.footprint.width, new.footprint.height)
            new = with_slots(new, slots_of(resized))
    return checked(prune_bindings(replace_resources(design, {entity_id(old): new})))


def resize(design, identifier, width, height):
    old = resource(design, identifier)
    check_bounds(design, old.footprint.x, old.footprint.y, width, height)
    return checked(
        prune_bindings(
            replace_resources(design, {identifier: resize_resource(old, width, height)})
        )
    )


def delete(design, identifiers):
    identifiers = set(identifiers)
    candidate = replace(
        design,
        **{
            collection: tuple(
                r
                for r in getattr(design, collection)
                if entity_id(r) not in identifiers
            )
            for collection, _, _ in COLLECTIONS.values()
        },
    )
    candidate = replace(
        candidate,
        buffers=tuple(
            replace(b, machine_id=None) if b.machine_id in identifiers else b
            for b in candidate.buffers
        ),
    )
    return checked(prune_bindings(candidate))


def impact(before, after):
    """Human-readable destructive reference changes for one confirmation."""
    lines = []
    new = {entity_id(r): r for r in iter_resources(after)}
    for old in iter_resources(before):
        changed = new.get(entity_id(old))
        if changed is None:
            lines.append(f"Remove {old.name} ({entity_id(old)})")
        if slots_of(old) is not None:
            kept = (
                {s.slot_id for s in (slots_of(changed) or ())}
                if changed is not None
                else set()
            )
            removed = [s.slot_id for s in slots_of(old) if s.slot_id not in kept]
            if removed:
                lines.append(f"{entity_id(old)}: remove slots {', '.join(removed)}")
        if (
            isinstance(old, BufferDesign)
            and old.machine_id
            and changed is not None
            and changed.machine_id != old.machine_id
        ):
            lines.append(f"{old.buffer_id}: detach from {old.machine_id}")
        if isinstance(old, PortDesign):
            retained = (
                set(changed.bindings) if isinstance(changed, PortDesign) else set()
            )
            for binding in old.bindings:
                if binding not in retained:
                    target = binding.target
                    lines.append(
                        f"{old.port_id}: remove binding to {target_owner_id(target)}{('/' + target.slot_id) if hasattr(target, 'slot_id') else ''}"
                    )
    return lines


def rename(design, identifier, new_id):
    # None identifies the factory overview. A resource may legitimately share
    # the factory's ID because these are different identity namespaces.
    if identifier is None:
        return checked(replace(design, factory_id=new_id))
    old = resource(design, identifier)
    id_field = COLLECTIONS[kind_of(old)][2]
    replacements = {identifier: replace(old, **{id_field: new_id})}
    candidate = replace_resources(design, replacements)
    candidate = replace(
        candidate,
        buffers=tuple(
            replace(b, machine_id=new_id) if b.machine_id == identifier else b
            for b in candidate.buffers
        ),
    )
    ports = []
    for p in candidate.ports:
        bindings = []
        for b in p.bindings:
            t = b.target
            if target_owner_id(t) == identifier:
                target_key = next(
                    f.name
                    for f in fields(t)
                    if f.name.endswith("_id") and f.name != "slot_id"
                )
                t = replace(t, **{target_key: new_id})
            bindings.append(replace(b, target=t))
        ports.append(replace(p, bindings=tuple(bindings)))
    return checked(replace(candidate, ports=tuple(ports)))


def set_slots(design, identifier, slots, renames=None):
    old = resource(design, identifier)
    candidate = replace_resources(design, {identifier: with_slots(old, slots)})
    if renames:
        candidate = replace(
            candidate,
            ports=tuple(
                replace(
                    p,
                    bindings=tuple(
                        replace(
                            b,
                            target=replace(b.target, slot_id=renames[b.target.slot_id]),
                        )
                        if target_owner_id(b.target) == identifier
                        and getattr(b.target, "slot_id", None) in renames
                        else b
                        for b in p.bindings
                    ),
                )
                for p in candidate.ports
            ),
        )
    return checked(prune_bindings(candidate))


def convert_storage(design, identifier, mode, capacity):
    old = resource(design, identifier)
    if mode == "pool":
        storage = PoolStorage(capacity)
    else:
        f = old.footprint
        storage = SlotStorage(
            tuple(
                SlotDesign(f"slot_{y * f.width + x + 1:03d}", Cell(x, y), capacity)
                for y in range(f.height)
                for x in range(f.width)
            )
        )
    return checked(
        prune_bindings(
            replace_resources(design, {identifier: replace(old, storage=storage)})
        )
    )


def target_at(design, identifier, cell):
    value = resource(design, identifier)
    if (slots := slots_of(value)) is not None:
        local = Cell(cell.x - value.footprint.x, cell.y - value.footprint.y)
        slot = next((s for s in slots if s.local_cell == local), None)
        if slot is None:
            return None
        return (
            BufferSlotTarget(identifier, slot.slot_id)
            if isinstance(value, BufferDesign)
            else InspectionSlotTarget(identifier, slot.slot_id)
        )
    cls = {
        MachineDesign: MachineTarget,
        BufferDesign: BufferTarget,
        ScrapBinDesign: ScrapBinTarget,
        ChargerDesign: ChargerTarget,
    }.get(type(value))
    return cls(identifier) if cls else None


def default_binding(target):
    return PortBinding(
        target,
        ("charge",)
        if isinstance(target, ChargerTarget)
        else ("drop_off",)
        if isinstance(target, ScrapBinTarget)
        else ("pickup", "drop_off"),
    )


def copy_selection(design, identifiers):
    selected = set(identifiers)
    return tuple(r for r in iter_resources(design) if entity_id(r) in selected)


def paste(design, copied, x, y):
    if not copied:
        return design, ()
    left = min(rect_of(r)[0] for r in copied)
    top = min(rect_of(r)[1] for r in copied)
    used = {entity_id(r) for r in iter_resources(design)}
    mapping = {}
    for r in copied:
        new_id = next_id(kind_of(r), used)
        used.add(new_id)
        mapping[entity_id(r)] = new_id
    result = design
    for r in copied:
        kind = kind_of(r)
        collection, _, id_field = COLLECTIONS[kind]
        rx, ry, _, _ = rect_of(r)
        new = relocate(r, x + rx - left, y + ry - top)
        new = replace(
            new, **{id_field: mapping[entity_id(r)], "name": r.name + " copy"}
        )
        if isinstance(new, BufferDesign):
            new = replace(new, machine_id=mapping.get(r.machine_id))
        if isinstance(new, PortDesign):
            bindings = []
            for b in r.bindings:
                owner = target_owner_id(b.target)
                if owner not in mapping:
                    continue
                key = next(
                    f.name
                    for f in fields(b.target)
                    if f.name.endswith("_id") and f.name != "slot_id"
                )
                bindings.append(
                    replace(b, target=replace(b.target, **{key: mapping[owner]}))
                )
            new = replace(new, bindings=tuple(bindings))
        result = replace(result, **{collection: (*getattr(result, collection), new)})
    return checked(result), tuple(mapping.values())
