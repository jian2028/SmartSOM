"""Guarded derived lookups for native port records; extensions retain their scan."""

from dataclasses import asdict

from smartsom.domain.factory_design import (
    BufferSlotTarget,
    BufferTarget,
    ChargerTarget,
    InspectionSlotTarget,
    MachineTarget,
    PortBinding,
    PortDesign,
    ScrapBinTarget,
)


def _native_fields(target):
    # Exact identity checks also avoid hashing unknown extension classes/targets.
    kind = type(target)
    if kind is BufferSlotTarget:
        return ("buffer_id", "slot_id", "kind")
    if kind is InspectionSlotTarget:
        return ("inspection_station_id", "slot_id", "kind")
    if kind is BufferTarget:
        return ("buffer_id", "kind")
    if kind is MachineTarget:
        return ("machine_id", "kind")
    if kind is ScrapBinTarget:
        return ("scrap_bin_id", "kind")
    if kind is ChargerTarget:
        return ("charger_id", "kind")
    return ()


def decode_target(target):
    """Skip asdict only for exact native targets containing plain strings."""
    fields = _native_fields(target)
    if fields:
        values = tuple(getattr(target, name, None) for name in fields)
        if all(type(value) is str for value in values):
            return values[0], values[1] if len(fields) == 3 else "pool"
    # Subclasses can contain mutable/unhashable fields and custom deepcopy hooks.
    data = asdict(target)
    owner = next(v for k, v in data.items() if k.endswith("_id") and k != "slot_id")
    return owner, data.get("slot_id", "pool")


class PortLookup:
    """One current topology, guarded before use; never captures simulator state."""

    def __init__(self, ports, records, guards, index):
        self.ports, self.records = ports, records
        self.guards, self.index = guards, index

    def matches(self, ports):
        return (
            type(ports) is dict
            and len(ports) == len(self.ports)
            and all(a is b for a, b in zip(ports.values(), self.ports))
            and all(type(record) is kind for record, kind in self.records)
            and all(
                getattr(record, field, None) is value
                for record, field, value in self.guards
            )
        )

    @classmethod
    def build(cls, ports):
        """Unknown/mutable records decline the index without decoding them."""
        if type(ports) is not dict:
            return None
        ordered = tuple(ports.values())
        records, guards, index = [], [], {}
        for port in ordered:
            if type(port) is not PortDesign:
                return None
            records.append((port, PortDesign))
            bindings = getattr(port, "bindings", None)
            if type(bindings) is not tuple:
                return None
            guards.append((port, "bindings", bindings))
            seen = set()
            for binding in bindings:
                if type(binding) is not PortBinding:
                    return None
                records.append((binding, PortBinding))
                target = getattr(binding, "target", None)
                operations = getattr(binding, "operations", None)
                fields = _native_fields(target)
                if not fields or type(operations) is not tuple:
                    return None
                values = tuple(getattr(target, name, None) for name in fields)
                if any(type(value) is not str for value in values + operations):
                    return None
                records.append((target, type(target)))
                guards.extend(
                    ((binding, "target", target), (binding, "operations", operations))
                )
                guards.extend(zip((target,) * len(fields), fields, values))
                for operation in operations:
                    key = values[0], operation
                    if key not in seen:
                        index.setdefault(key, []).append(port)
                        seen.add(key)
        return cls(
            ordered,
            tuple(records),
            tuple(guards),
            {k: tuple(v) for k, v in index.items()},
        )
