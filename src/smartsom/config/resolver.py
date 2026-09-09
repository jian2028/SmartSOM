"""Resolve references and materialize a complete immutable single-run input."""

from dataclasses import dataclass, replace
from pathlib import Path

from smartsom.config.algorithm_binding import (
    bind_algorithm,
    validate_algorithm_references,
)
from smartsom.config.arrivals import read_arrivals
from smartsom.config.codec import (
    ConfigurationError,
    digest,
    normalize_factory,
    read_model,
)
from smartsom.config.materialization import (
    materialize_arrivals,
    materialize_machine_events,
    materialize_processing_times,
    materialize_quality,
    materialize_workload,
)
from smartsom.config.models import (
    AlgorithmFile,
    ArrivalProvenance,
    FactoryFile,
    FixedArrivals,
    FixedMachineEvents,
    FixedProcessingTimes,
    FixedQuality,
    GeneratedArrivals,
    GeneratedMachineEvents,
    GeneratedProcessingTimes,
    GenerationProvenance,
    InstanceFile,
    MachineEventFile,
    MachineEventProvenance,
    ProcessingProvenance,
    ProcessingTimeFile,
    ProfileFile,
    QualityFile,
    QualityProvenance,
    RunSpec,
    ScenarioFile,
)
from smartsom.config.seeds import SEED_VERSION, NamedSeed, derive_seeds
from smartsom.domain import ArrivalPlan, FactorySpec, WorkloadInstance
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.quality import QualityPlan
from smartsom.workloads.fjs import ImportProvenance


@dataclass(frozen=True, slots=True)
class SourceFile:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class StudySeedOrigin:
    study_seed: int
    case_id: str
    replication: int
    algorithm_id: str
    world_seed: int
    algorithm_seed: int
    version: str = "smartsom.study-seeds/v1"


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    run: RunSpec
    scenario: ScenarioFile
    algorithm: AlgorithmFile
    factory: FactorySpec
    workload: WorkloadInstance
    profile: ProfileFile | None
    provenance: GenerationProvenance | ImportProvenance | None
    seeds: tuple[NamedSeed, ...]
    sources: tuple[SourceFile, ...]
    factory_sha256: str
    workload_sha256: str
    seed_version: str = SEED_VERSION
    arrivals: ArrivalPlan | None = None
    arrival_provenance: ArrivalProvenance | None = None
    arrivals_sha256: str | None = None
    processing_times: ProcessingTimePlan | None = None
    processing_provenance: ProcessingProvenance | None = None
    processing_times_sha256: str | None = None
    machine_events: MachineOutagePlan | None = None
    machine_event_provenance: MachineEventProvenance | None = None
    machine_events_sha256: str | None = None
    transport_enabled: bool = False
    transport_sha256: str | None = None
    buffers_enabled: bool = False
    buffers_sha256: str | None = None
    quality: QualityPlan | None = None
    quality_provenance: QualityProvenance | None = None
    quality_draws_sha256: str | None = None
    quality_modes_sha256: str | None = None
    study_seed_origin: StudySeedOrigin | None = None
    holding_buffer_enabled: bool = False
    holding_buffer_sha256: str | None = None


def _reference(owner: Path, value: str) -> Path:
    try:
        return (owner.parent / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(
            f"{owner}: invalid reference {value!r}: {exc}"
        ) from exc


def resolve_run(run_config_path: str | Path) -> ResolvedRun:
    path = Path(run_config_path).resolve()
    run, sha = read_model(path, RunSpec)
    return _resolve_run_spec(run, path, [SourceFile("run", path, sha)])


def _resolve_run_spec(
    run: RunSpec,
    path: Path,
    sources: list[SourceFile],
    *,
    algorithm_override: AlgorithmFile | None = None,
    scenario_override: ScenarioFile | None = None,
) -> ResolvedRun:
    """Shared preparation path; studies supply a run spec without authoring files."""

    def load(path, model, role):
        parsed, sha256 = read_model(path, model)
        sources.append(SourceFile(role, path, sha256))
        return parsed

    scenario_path = _reference(path, run.scenario)
    algorithm_path = _reference(path, run.algorithm)
    scenario = scenario_override or load(scenario_path, ScenarioFile, "scenario")
    algorithm = algorithm_override or load(algorithm_path, AlgorithmFile, "algorithm")
    from smartsom.learning.checkpoint import (
        resolve_checkpoint_reference,
        validate_checkpoint,
    )

    algorithm = resolve_checkpoint_reference(algorithm, algorithm_path)
    factory_path = _reference(scenario_path, scenario.factory)
    workload_path = _reference(scenario_path, scenario.workload.path)
    factory = normalize_factory(load(factory_path, FactoryFile, "factory").factory)
    transport_enabled = scenario.transport is not None
    buffers_enabled = scenario.buffers is not None
    holding_buffer_enabled = scenario.holding_buffer is not None
    if holding_buffer_enabled and factory.holding_buffer is None:
        raise ConfigurationError(
            "enabled holding buffer requires factory holding resource"
        )
    if transport_enabled and factory.transport is None:
        raise ConfigurationError(
            "enabled transport requires factory transport resources"
        )
    generated = scenario.workload.kind == "profile"
    run = bind_algorithm(run, scenario, algorithm)
    seeds = derive_seeds(
        run.seed,
        generated=generated,
        quality=scenario.quality is not None,
        solver=algorithm.algorithm.interface_kind == "offline_solver",
    )
    seed_values = {seed.domain: seed.value for seed in seeds}
    try:
        source = load(
            workload_path, ProfileFile if generated else InstanceFile, "workload"
        )
        inputs = materialize_workload(factory, source, seed_values["workload"])
        workload = inputs.workload
        arrival_source = None
        arrival_provenance = None
        if isinstance(scenario.arrivals, FixedArrivals):
            arrival_path = _reference(scenario_path, scenario.arrivals.path)
            arrival_source, arrival_provenance, raw_sha = read_arrivals(arrival_path)
            sources.append(SourceFile("arrivals", arrival_path, raw_sha))
            scenario = scenario.model_copy(
                update={
                    "arrivals": scenario.arrivals.model_copy(
                        update={"path": str(arrival_path)}
                    )
                }
            )
        elif isinstance(scenario.arrivals, GeneratedArrivals):
            arrival_source = scenario.arrivals.profile
        arrivals = materialize_arrivals(
            workload, arrival_source, seed_values["demand"], arrival_provenance
        )
        processing_source = None
        if isinstance(scenario.processing_time, FixedProcessingTimes):
            processing_path = _reference(scenario_path, scenario.processing_time.path)
            processing_source = load(
                processing_path, ProcessingTimeFile, "processing_times"
            )
            scenario = scenario.model_copy(
                update={
                    "processing_time": scenario.processing_time.model_copy(
                        update={"path": str(processing_path)}
                    )
                }
            )
        elif isinstance(scenario.processing_time, GeneratedProcessingTimes):
            processing_source = scenario.processing_time.profile
        processing = materialize_processing_times(
            workload, processing_source, seed_values["processing_time"]
        )
        machine_source = None
        if isinstance(scenario.machine_events, FixedMachineEvents):
            machine_path = _reference(scenario_path, scenario.machine_events.path)
            machine_source = load(machine_path, MachineEventFile, "machine_events")
            scenario = scenario.model_copy(
                update={
                    "machine_events": scenario.machine_events.model_copy(
                        update={"path": str(machine_path)}
                    )
                }
            )
        elif isinstance(scenario.machine_events, GeneratedMachineEvents):
            machine_source = scenario.machine_events.profile
        machine_events = materialize_machine_events(
            factory, machine_source, seed_values["machine_events"]
        )
        quality_source = scenario.quality
        if isinstance(quality_source, FixedQuality):
            quality_path = _reference(scenario_path, quality_source.path)
            quality_source = load(quality_path, QualityFile, "quality")
            scenario = scenario.model_copy(
                update={
                    "quality": scenario.quality.model_copy(
                        update={"path": str(quality_path)}
                    )
                }
            )
        quality = materialize_quality(
            factory,
            workload,
            processing.plan,
            quality_source,
            seed_values.get("quality"),
        )
        seeds = tuple(
            replace(seed, consumed=arrivals.seed_consumed)
            if seed.domain == "demand"
            else replace(seed, consumed=processing.seed_consumed)
            if seed.domain == "processing_time"
            else replace(seed, consumed=machine_events.seed_consumed)
            if seed.domain == "machine_events"
            else replace(seed, consumed=quality.seed_consumed)
            if seed.domain == "quality"
            else seed
            for seed in seeds
        )
        if isinstance(source, InstanceFile) and source.content_sha256 is not None:
            if source.content_sha256 != inputs.sha256:
                raise ValueError("instance content_sha256 does not match its workload")
        validate_algorithm_references(
            algorithm,
            workload,
            factory,
            transport_enabled=transport_enabled,
            buffers_enabled=buffers_enabled,
            holding_buffer_enabled=holding_buffer_enabled,
            quality=quality.plan,
        )
    except ValueError as exc:
        raise ConfigurationError(
            f"{path}: materialization/reference validation: {exc}"
        ) from exc
    resolved = ResolvedRun(
        run=run.model_copy(
            update={
                "scenario": str(scenario_path),
                "algorithm": str(algorithm_path),
                "output_root": str(_reference(path, run.output_root)),
            }
        ),
        scenario=scenario.model_copy(
            update={
                "factory": str(factory_path),
                "workload": scenario.workload.model_copy(
                    update={"path": str(workload_path)}
                ),
            }
        ),
        algorithm=algorithm,
        factory=factory,
        workload=workload,
        profile=inputs.profile,
        provenance=inputs.provenance,
        seeds=seeds,
        sources=tuple(sources),
        factory_sha256=digest(factory),
        quality=quality.plan,
        quality_provenance=quality.provenance,
        quality_draws_sha256=digest(quality.plan.draws)
        if quality.plan is not None
        else None,
        quality_modes_sha256=digest(quality.plan.modes)
        if quality.plan is not None
        else None,
        holding_buffer_enabled=holding_buffer_enabled,
        holding_buffer_sha256=digest(factory.holding_buffer)
        if holding_buffer_enabled
        else None,
        buffers_enabled=buffers_enabled,
        buffers_sha256=digest(factory.buffers) if buffers_enabled else None,
        transport_enabled=transport_enabled,
        transport_sha256=digest(factory.transport) if transport_enabled else None,
        workload_sha256=inputs.sha256,
        arrivals=arrivals.plan,
        arrival_provenance=arrivals.provenance,
        arrivals_sha256=digest(arrivals.plan) if arrivals.plan is not None else None,
        processing_times=processing.plan,
        processing_provenance=processing.provenance,
        processing_times_sha256=digest(processing.plan)
        if processing.plan is not None
        else None,
        machine_events=machine_events.plan,
        machine_event_provenance=machine_events.provenance,
        machine_events_sha256=digest(machine_events.plan)
        if machine_events.plan is not None
        else None,
    )
    validate_checkpoint(resolved)
    return resolved
