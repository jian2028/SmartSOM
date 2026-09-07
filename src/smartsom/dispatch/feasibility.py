"""Pure construction of the current static decision view."""

from smartsom.dispatch.contracts import DecisionContext, Dispatch, DispatchCandidate
from smartsom.domain import MachineState, Operation, OperationState, OperationStatus


def build_decision(
    simulation_time: int,
    operations: tuple[Operation, ...],
    operation_states: tuple[OperationState, ...],
    machine_states: tuple[MachineState, ...],
) -> DecisionContext:
    states = {state.operation_id: state for state in operation_states}
    idle = {state.machine_id for state in machine_states if state.operation_id is None}
    candidates = []
    for operation in sorted(operations, key=lambda item: item.operation_id):
        if states[operation.operation_id].status != OperationStatus.PENDING:
            continue
        if any(
            states[predecessor].status != OperationStatus.COMPLETED
            for predecessor in operation.predecessor_ids
        ):
            continue
        for mode in sorted(operation.modes, key=lambda mode: mode.processing_mode_id):
            if mode.machine_id in idle:
                candidates.append(
                    DispatchCandidate(
                        Dispatch(operation.operation_id, mode.processing_mode_id),
                        mode.machine_id,
                        mode.nominal_ticks,
                    )
                )
    return DecisionContext(
        simulation_time,
        tuple(sorted(operation_states, key=lambda item: item.operation_id)),
        tuple(sorted(machine_states, key=lambda item: item.machine_id)),
        tuple(candidates),
    )
