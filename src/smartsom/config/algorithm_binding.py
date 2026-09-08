"""Compatibility of typed algorithm requests with existing scenario inputs."""

from smartsom.config.codec import ConfigurationError
from smartsom.config.models import (
    AlgorithmFile,
    CPSatAlgorithm,
    EpisodeBudget,
    LearningAlgorithm,
    RunBudget,
    RunSpec,
    ScenarioFile,
    ScriptedAlgorithm,
)
from smartsom.dispatch import Dispatch, Transfer, Transport
from smartsom.domain import FactorySpec, WorkloadInstance
from smartsom.domain.quality import QualityPlan


def bind_algorithm(
    run: RunSpec, scenario: ScenarioFile, algorithm: AlgorithmFile
) -> RunSpec:
    if isinstance(algorithm.algorithm, CPSatAlgorithm):
        if isinstance(run.budget, EpisodeBudget):
            raise ConfigurationError("CP accepts only a solver budget")
        if scenario.holding_buffer is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support holding buffer")
        if scenario.quality is not None:
            raise ConfigurationError("pyjobshop.cp_sat does not support quality")
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
    elif isinstance(algorithm.algorithm, LearningAlgorithm):
        if scenario.visibility != "decision_context":
            raise ConfigurationError("learning requires decision_context visibility")
        if isinstance(run.budget, RunBudget):
            raise ConfigurationError("learning does not accept a solver budget")
        if run.budget is None:
            return run.model_copy(update={"budget": EpisodeBudget()})
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
    holding_buffer_enabled: bool = False,
    quality: QualityPlan | None = None,
) -> None:
    selected = algorithm.algorithm
    if not isinstance(selected, ScriptedAlgorithm):
        fixed = getattr(selected.parameters, "quality_mode", None)
        if fixed is not None:
            if quality is None:
                raise ValueError("fixed quality_mode requires enabled quality")
            groups = {}
            for row in quality.modes:
                groups.setdefault(
                    (row.operation_id, row.base_processing_mode_id), set()
                ).add(row.mode.quality_mode_id)
            if any(fixed not in labels for labels in groups.values()):
                raise ValueError(
                    f"all candidate base modes must support quality mode {fixed!r}"
                )
        return
    modes = {
        op.operation_id: {mode.processing_mode_id for mode in op.modes}
        for op in workload.operations
    }
    if quality is not None:
        modes = {}
        for row in quality.modes:
            modes.setdefault(row.operation_id, set()).add(row.processing_mode_id)
    for action in selected.parameters.actions:
        if (
            isinstance(action, (Transport, Transfer))
            and action.destination.kind == "holding"
        ):
            if (
                isinstance(action, Transfer)
                or not holding_buffer_enabled
                or factory is None
                or factory.holding_buffer is None
                or action.destination.buffer_id != factory.holding_buffer.buffer_id
            ):
                raise ValueError(
                    "script holding destination requires enabled known AGV holding buffer"
                )
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
