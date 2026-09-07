"""Resolve references and materialize a complete immutable single-run input."""

from dataclasses import dataclass, replace
from pathlib import Path

from smartsom.config.arrivals import read_arrivals
from smartsom.config.codec import (
    ConfigurationError,
    digest,
    normalize_factory,
    normalize_workload,
    read_model,
)
from smartsom.config.models import (
    AlgorithmFile,
    ArrivalProvenance,
    CPSatAlgorithm,
    FactoryFile,
    FixedArrivals,
    FixedProcessingTimes,
    GeneratedArrivals,
    GeneratedProcessingTimes,
    GenerationProvenance,
    InstanceFile,
    ProcessingProvenance,
    ProcessingTimeFile,
    ProfileFile,
    RunBudget,
    RunSpec,
    ScenarioFile,
    ScriptedAlgorithm,
)
from smartsom.config.seeds import SEED_VERSION, NamedSeed, derive_seeds
from smartsom.dispatch import Dispatch
from smartsom.domain import ArrivalPlan, FactorySpec, WorkloadInstance, validate_problem
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.workloads import generate, generate_fjsp
from smartsom.workloads.arrivals import generate_arrivals
from smartsom.workloads.fjs import ImportProvenance
from smartsom.workloads.processing_times import generate_processing_times


@dataclass(frozen=True, slots=True)
class SourceFile:
    role: str
    path: Path
    sha256: str


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


def _reference(owner: Path, value: str) -> Path:
    try:
        return (owner.parent / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(
            f"{owner}: invalid reference {value!r}: {exc}"
        ) from exc


def resolve_run(run_config_path: str | Path) -> ResolvedRun:
    sources = []

    def load(path, model, role):
        parsed, sha256 = read_model(path, model)
        sources.append(SourceFile(role, path, sha256))
        return parsed

    path = Path(run_config_path).resolve()
    run = load(path, RunSpec, "run")
    scenario_path = _reference(path, run.scenario)
    algorithm_path = _reference(path, run.algorithm)
    scenario = load(scenario_path, ScenarioFile, "scenario")
    algorithm = load(algorithm_path, AlgorithmFile, "algorithm")
    factory_path = _reference(scenario_path, scenario.factory)
    workload_path = _reference(scenario_path, scenario.workload.path)
    factory = normalize_factory(load(factory_path, FactoryFile, "factory").factory)
    generated = scenario.workload.kind == "profile"
    offline = isinstance(algorithm.algorithm, CPSatAlgorithm)
    if offline and scenario.arrivals is not None:
        raise ConfigurationError("pyjobshop.cp_sat does not support arrivals")
    if offline and scenario.processing_time is not None:
        raise ConfigurationError(
            "pyjobshop.cp_sat does not support processing uncertainty"
        )
    if offline:
        if scenario.visibility != "full_static":
            raise ConfigurationError(
                "pyjobshop.cp_sat requires scenario visibility full_static"
            )
        if run.budget is None:
            run = run.model_copy(update={"budget": RunBudget()})
    elif run.budget is not None:
        raise ConfigurationError("online providers do not accept a solver budget")
    seeds = derive_seeds(run.seed, generated=generated, solver=offline)
    profile = None
    provenance = None
    arrivals = None
    arrival_provenance = None
    processing_times = None
    processing_provenance = None
    try:
        if generated:
            profile = load(workload_path, ProfileFile, "workload")
            workload_seed = next(
                seed.value for seed in seeds if seed.domain == "workload"
            )
            materialize = (
                generate if profile.generator == "static_jsp_v1" else generate_fjsp
            )
            workload = materialize(factory, profile.profile, workload_seed)
            provenance = GenerationProvenance(
                generator=profile.generator,
                generator_version="1",
                profile_sha256=digest(profile),
                effective_seed=workload_seed,
            )
        else:
            instance = load(workload_path, InstanceFile, "workload")
            workload = instance.workload
            provenance = instance.provenance
        workload = normalize_workload(workload)
        validate_problem(factory, workload)
        workload_sha256 = digest(workload)
        if isinstance(scenario.arrivals, FixedArrivals):
            arrival_path = _reference(scenario_path, scenario.arrivals.path)
            arrivals, arrival_provenance, raw_sha = read_arrivals(arrival_path)
            sources.append(SourceFile("arrivals", arrival_path, raw_sha))
            scenario = scenario.model_copy(
                update={
                    "arrivals": scenario.arrivals.model_copy(
                        update={"path": str(arrival_path)}
                    )
                }
            )
        elif isinstance(scenario.arrivals, GeneratedArrivals):
            arrival_profile = scenario.arrivals.profile
            demand_seed = next(seed.value for seed in seeds if seed.domain == "demand")
            arrivals = generate_arrivals(workload, arrival_profile, demand_seed)
            arrival_provenance = ArrivalProvenance(
                profile_sha256=digest(arrival_profile), effective_seed=demand_seed
            )
            seeds = derive_seeds(
                run.seed,
                generated=generated,
                solver=offline,
                demand=arrival_profile.initial_job_count < len(arrivals.jobs),
            )
        if arrivals is not None:
            arrivals.validate(workload)
        if isinstance(scenario.processing_time, FixedProcessingTimes):
            processing_path = _reference(scenario_path, scenario.processing_time.path)
            processing = load(processing_path, ProcessingTimeFile, "processing_times")
            processing_times, processing_provenance = (
                processing.processing_times,
                processing.provenance,
            )
            scenario = scenario.model_copy(
                update={
                    "processing_time": scenario.processing_time.model_copy(
                        update={"path": str(processing_path)}
                    )
                }
            )
        elif isinstance(scenario.processing_time, GeneratedProcessingTimes):
            processing_profile = scenario.processing_time.profile
            processing_seed = next(
                seed.value for seed in seeds if seed.domain == "processing_time"
            )
            processing_times, draws = generate_processing_times(
                workload, processing_profile, processing_seed
            )
            processing_provenance = ProcessingProvenance(
                profile=processing_profile,
                profile_sha256=digest(processing_profile),
                effective_seed=processing_seed,
                draws=draws,
            )
            seeds = tuple(
                replace(
                    seed, consumed=processing_profile.low != processing_profile.high
                )
                if seed.domain == "processing_time"
                else seed
                for seed in seeds
            )
        if processing_times is not None:
            processing_times.validate(workload)
        if not generated and instance.content_sha256 is not None:
            if instance.content_sha256 != workload_sha256:
                raise ValueError("instance content_sha256 does not match its workload")
        selected = algorithm.algorithm
        if isinstance(selected, ScriptedAlgorithm):
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
                    raise ValueError(
                        f"script references unknown operation or mode: {action!r}"
                    )
    except ValueError as exc:
        raise ConfigurationError(
            f"{path}: materialization/reference validation: {exc}"
        ) from exc
    return ResolvedRun(
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
        profile=profile,
        provenance=provenance,
        seeds=seeds,
        sources=tuple(sources),
        factory_sha256=digest(factory),
        workload_sha256=workload_sha256,
        arrivals=arrivals,
        arrival_provenance=arrival_provenance,
        arrivals_sha256=digest(arrivals) if arrivals is not None else None,
        processing_times=processing_times,
        processing_provenance=processing_provenance,
        processing_times_sha256=digest(processing_times)
        if processing_times is not None
        else None,
    )
