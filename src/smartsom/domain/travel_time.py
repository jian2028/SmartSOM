"""Immutable directed travel times; no configuration or framework dependency."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TravelTimeMatrix:
    points: tuple[tuple[str, int, int], ...]
    times: tuple[tuple[str, str, int | None], ...]
    source: str = "manual"

    def __post_init__(self):
        ids = [p[0] for p in self.points]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("travel matrix point identities must be unique")
        pairs = set()
        for start, end, ticks in self.times:
            if start not in ids or end not in ids or (start, end) in pairs:
                raise ValueError("unknown or duplicate travel matrix point")
            if ticks is not None and (type(ticks) is not int or ticks < 0):
                raise ValueError("travel times require nonnegative integer ticks")
            if start == end and ticks != 0:
                raise ValueError("travel matrix diagonal must be zero")
            pairs.add((start, end))
        if len(pairs) != len(ids) ** 2:
            raise ValueError("travel matrix must cover all ordered point pairs")

    def __deepcopy__(self, memo):
        return self


def physical_contract(scenario):
    return {
        "transport": "travel-time-matrix/v1"
        if scenario.transport_matrix
        else "grid/v3",
        "processing_rounding": scenario.processing_rounding,
        "dispatch_semantics": "nonexclusive-intentions/3",
    }


def validate_model_contract(metadata, scenario=None):
    """Reject historical decision semantics on every model-loading entry point."""
    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )

    if (
        metadata.get("action_contract") != ACTION_CONTRACT
        or metadata.get("observation_contract") != OBSERVATION_CONTRACT
    ):
        raise ValueError("model decision contract is incompatible; retraining required")
    physical = metadata.get("physical_contract")
    if (
        not isinstance(physical, dict)
        or physical.get("dispatch_semantics") != "nonexclusive-intentions/3"
        or physical.get("transport") not in ("grid/v3", "travel-time-matrix/v1")
        or physical.get("processing_rounding") not in ("half_up", "ceil")
        or (scenario is not None and physical != physical_contract(scenario))
    ):
        raise ValueError(
            "model transport/processing contract is incompatible; retraining required"
        )


def validate_matrix_scenario(scenario):
    """Point identity and next-leg feasibility before a matrix episode allocates."""
    from dataclasses import asdict

    matrix = scenario.transport_matrix
    if matrix is None:
        return
    factory = scenario.factory
    expected = {p.port_id: (p.cell.x, p.cell.y) for p in factory.ports}
    expected.update(
        {
            "initial:" + a.agv_id: (a.initial_cell.x, a.initial_cell.y)
            for a in factory.agvs
        }
    )
    if {p: (x, y) for p, x, y in matrix.points} != expected:
        raise ValueError(
            "travel matrix points differ from factory ports/initial positions"
        )
    times = {(a, b): v for a, b, v in matrix.times}

    def ports(owner, operation):
        result = []
        for port in factory.ports:
            for binding in port.bindings:
                fields = asdict(binding.target)
                key = next(
                    v for k, v in fields.items() if k.endswith("_id") and k != "slot_id"
                )
                if key == owner and operation in binding.operations:
                    result.append(port.port_id)
                    break
        return result

    pre = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_pre"
    }
    post = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_post"
    }
    inputs = [b.buffer_id for b in factory.buffers if b.role == "system_input"]
    outputs = [b.buffer_id for b in factory.buffers if b.role == "system_output"]
    inspections = [s.inspection_station_id for s in factory.inspection_stations]

    def require_leg(sources, destinations):
        targets = [p for owner in destinations for p in ports(owner, "drop_off")]
        for owner in sources:
            origins = ports(owner, "pickup")
            if (
                not origins
                or not targets
                or any(
                    not any(times[(a, b)] is not None for b in targets) for a in origins
                )
            ):
                raise ValueError(
                    "travel matrix has no feasible next leg from a declared pickup port"
                )

    if scenario.demands:
        require_leg(inspections, outputs)
    for demand in scenario.demands:
        previous = [demand.input_id or inputs[0]]
        for step in demand.steps:
            capable = [
                m.machine_id
                for m in factory.machines
                if step.operation_type in m.operation_types
            ]
            require_leg(previous, [pre.get(m, m) for m in capable])
            previous = [post.get(m, m) for m in capable]
        require_leg(previous, inspections)
