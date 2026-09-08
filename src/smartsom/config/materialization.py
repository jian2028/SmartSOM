"""Materialize already parsed inputs without algorithms, paths or run output."""

from dataclasses import dataclass

from smartsom.config.codec import digest, normalize_workload
from smartsom.config.models import (
    ArrivalProvenance,
    GeneratedQuality,
    GenerationProvenance,
    InstanceFile,
    MachineEventFile,
    MachineEventProvenance,
    ProcessingProvenance,
    ProcessingTimeFile,
    ProfileFile,
    QualityFile,
    QualityProvenance,
)
from smartsom.domain import ArrivalPlan, FactorySpec, WorkloadInstance, validate_problem
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.quality import QualityPlan
from smartsom.modules.quality import prepare_quality
from smartsom.workloads import generate, generate_fjsp
from smartsom.workloads.arrivals import UniformReleaseProfile, generate_arrivals
from smartsom.workloads.fjs import ImportProvenance
from smartsom.workloads.machine_events import (
    MachineEventProfile,
    generate_machine_events,
)
from smartsom.workloads.processing_times import (
    UniformMultiplierProfile,
    generate_processing_times,
)
from smartsom.workloads.quality import generate_quality


@dataclass(frozen=True, slots=True)
class WorkloadInputs:
    workload: WorkloadInstance
    profile: ProfileFile | None
    provenance: GenerationProvenance | ImportProvenance | None
    sha256: str


@dataclass(frozen=True, slots=True)
class ArrivalInputs:
    plan: ArrivalPlan | None
    provenance: ArrivalProvenance | None
    seed_consumed: bool


@dataclass(frozen=True, slots=True)
class ProcessingInputs:
    plan: ProcessingTimePlan | None
    provenance: ProcessingProvenance | None
    seed_consumed: bool


@dataclass(frozen=True, slots=True)
class MachineEventInputs:
    plan: MachineOutagePlan | None
    provenance: MachineEventProvenance | None
    seed_consumed: bool


def materialize_machine_events(
    factory: FactorySpec,
    source: MachineEventFile | MachineEventProfile | None,
    seed: int,
) -> MachineEventInputs:
    plan = provenance = None
    consumed = False
    if isinstance(source, MachineEventProfile):
        plan = generate_machine_events(factory, source, seed)
        provenance = MachineEventProvenance(
            profile=source, profile_sha256=digest(source), effective_seed=seed
        )
        consumed = True
    elif source is not None:
        plan, provenance = source.machine_events, source.provenance
    if plan is not None:
        plan.validate(factory)
    return MachineEventInputs(plan, provenance, consumed)


def materialize_workload(
    factory: FactorySpec, source: ProfileFile | InstanceFile, seed: int
) -> WorkloadInputs:
    profile = None
    if isinstance(source, ProfileFile):
        profile = source
        generator = generate if source.generator == "static_jsp_v1" else generate_fjsp
        workload = generator(factory, source.profile, seed)
        provenance = GenerationProvenance(
            generator=source.generator,
            generator_version="1",
            profile_sha256=digest(source),
            effective_seed=seed,
        )
    else:
        workload, provenance = source.workload, source.provenance
    workload = normalize_workload(workload)
    validate_problem(factory, workload)
    return WorkloadInputs(workload, profile, provenance, digest(workload))


def materialize_arrivals(
    workload: WorkloadInstance,
    source: ArrivalPlan | UniformReleaseProfile | None,
    seed: int,
    provenance: ArrivalProvenance | None = None,
) -> ArrivalInputs:
    consumed = False
    if isinstance(source, UniformReleaseProfile):
        plan = generate_arrivals(workload, source, seed)
        provenance = ArrivalProvenance(
            profile_sha256=digest(source), effective_seed=seed
        )
        consumed = source.initial_job_count < len(plan.jobs)
    else:
        plan = source
    if plan is not None:
        plan.validate(workload)
    return ArrivalInputs(plan, provenance, consumed)


def materialize_processing_times(
    workload: WorkloadInstance,
    source: ProcessingTimeFile | UniformMultiplierProfile | None,
    seed: int,
) -> ProcessingInputs:
    plan = None
    provenance = None
    consumed = False
    if isinstance(source, UniformMultiplierProfile):
        plan, draws = generate_processing_times(workload, source, seed)
        provenance = ProcessingProvenance(
            profile=source,
            profile_sha256=digest(source),
            effective_seed=seed,
            draws=draws,
        )
        consumed = source.low != source.high
    elif source is not None:
        plan, provenance = source.processing_times, source.provenance
    if plan is not None:
        plan.validate(workload)
    return ProcessingInputs(plan, provenance, consumed)


@dataclass(frozen=True, slots=True)
class QualityInputs:
    plan: QualityPlan | None
    provenance: QualityProvenance | None
    seed_consumed: bool


def materialize_quality(
    factory: FactorySpec,
    workload: WorkloadInstance,
    processing_times: ProcessingTimePlan | None,
    source: GeneratedQuality | QualityFile | None,
    seed: int | None,
) -> QualityInputs:
    if source is None:
        return QualityInputs(None, None, False)
    if isinstance(source, GeneratedQuality):
        draws = generate_quality(workload, seed)
        provenance = QualityProvenance(effective_seed=seed)
    elif isinstance(source, QualityFile):
        draws, provenance = source.draws, source.provenance
    else:
        raise ValueError("unsupported quality source")
    return QualityInputs(
        prepare_quality(factory, workload, draws, processing_times=processing_times),
        provenance,
        isinstance(source, GeneratedQuality),
    )
