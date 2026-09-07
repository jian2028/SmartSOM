"""Compatibility of typed algorithm requests with existing scenario inputs."""

from smartsom.config.codec import ConfigurationError
from smartsom.config.models import (
    AlgorithmFile,
    CPSatAlgorithm,
    RunBudget,
    RunSpec,
    ScenarioFile,
    ScriptedAlgorithm,
)
from smartsom.dispatch import Dispatch, Transfer, Transport
from smartsom.domain import FactorySpec, WorkloadInstance


def bind_algorithm(
    run: RunSpec, scenario: ScenarioFile, algorithm: AlgorithmFile
) -> RunSpec:
    if isinstance(algorithm.algorithm, CPSatAlgorithm):
        if scenario.buffers is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support buffers")
        if scenario.transport is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support transport")
        if scenario.machine_events is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support machine events")
        if scenario.arrivals is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support arrivals")
        if scenario.processing_time is not None:
            raise ConfigurationError(
                "pyjobshop.cp_sat does not support processing uncertainty"
            )
        if scenario.visibility != "full_static":
            raise ConfigurationError(
                "pyjobshop.cp_sat requires scenario visibility full_static"
            )
        if run.budget is None:
            return run.model_copy(update={"budget": RunBudget()})
    elif run.budget is not None:
        raise ConfigurationError("online providers do not accept a solver budget")
    return run


def validate_algorithm_references(
    algorithm: AlgorithmFile,
    workload: WorkloadInstance,
    factory: FactorySpec | None = None,
    *,
    transport_enabled: bool = False,
    buffers_enabled: bool = False,
) -> None:
    selected = algorithm.algorithm
    if not isinstance(selected, ScriptedAlgorithm):
        return
    modes = {
        op.operation_id: {mode.processing_mode_id for mode in op.modes}
        for op in workload.operations
    }
    for action in selected.parameters.actions:
        if isinstance(action, Transfer):
            if not buffers_enabled or transport_enabled or factory is None:
                raise ValueError(
                    "script transfer requires buffers enabled and AGV disabled"
                )
            if action.job_id not in {
                j.job_id for o in workload.orders for j in o.jobs
            } or (
                action.destination.kind == "machine"
                and action.destination.machine_id
                not in {m.machine_id for m in factory.machines}
            ):
                raise ValueError("script references unknown transfer job or machine")
        if isinstance(action, Transport):
            if not transport_enabled or factory is None or factory.transport is None:
                raise ValueError("script transport requires enabled factory transport")
            if (
                action.job_id not in {j.job_id for o in workload.orders for j in o.jobs}
                or action.agv_id not in {a.agv_id for a in factory.transport.agvs}
                or (
                    action.destination.kind == "machine"
                    and action.destination.machine_id
                    not in {m.machine_id for m in factory.machines}
                )
            ):
                raise ValueError(
                    "script references unknown transport job, AGV or machine"
                )
        if not isinstance(action, Dispatch):
            continue
        if (
            action.operation_id not in modes
            or action.processing_mode_id not in modes[action.operation_id]
        ):
            raise ValueError(f"script references unknown operation or mode: {action!r}")
