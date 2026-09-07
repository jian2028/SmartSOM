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
from smartsom.dispatch import Dispatch
from smartsom.domain import WorkloadInstance


def bind_algorithm(
    run: RunSpec, scenario: ScenarioFile, algorithm: AlgorithmFile
) -> RunSpec:
    if isinstance(algorithm.algorithm, CPSatAlgorithm):
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
    algorithm: AlgorithmFile, workload: WorkloadInstance
) -> None:
    selected = algorithm.algorithm
    if not isinstance(selected, ScriptedAlgorithm):
        return
    modes = {
        op.operation_id: {mode.processing_mode_id for mode in op.modes}
        for op in workload.operations
    }
    for action in selected.parameters.actions:
        if not isinstance(action, Dispatch):
            continue
        if (
            action.operation_id not in modes
            or action.processing_mode_id not in modes[action.operation_id]
        ):
            raise ValueError(f"script references unknown operation or mode: {action!r}")
