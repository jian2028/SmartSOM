"""Resolve references and materialize a complete immutable single-run input."""

from dataclasses import dataclass
from pathlib import Path

from smartsom.config.codec import (
    ConfigurationError,
    digest,
    normalize_factory,
    normalize_workload,
    read_model,
)
from smartsom.config.models import (
    AlgorithmFile,
    CPSatAlgorithm,
    FactoryFile,
    GenerationProvenance,
    InstanceFile,
    ProfileFile,
    RunBudget,
    RunSpec,
    ScenarioFile,
    ScriptedAlgorithm,
)
from smartsom.config.seeds import SEED_VERSION, NamedSeed, derive_seeds
from smartsom.dispatch import Dispatch
from smartsom.domain import FactorySpec, WorkloadInstance, validate_problem
from smartsom.workloads import generate
from smartsom.workloads.static_jsp import GENERATOR, GENERATOR_VERSION


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
    provenance: GenerationProvenance | None
    seeds: tuple[NamedSeed, ...]
    sources: tuple[SourceFile, ...]
    factory_sha256: str
    workload_sha256: str
    seed_version: str = SEED_VERSION


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
    try:
        if generated:
            profile = load(workload_path, ProfileFile, "workload")
            workload_seed = next(
                seed.value for seed in seeds if seed.domain == "workload"
            )
            workload = generate(factory, profile.profile, workload_seed)
            provenance = GenerationProvenance(
                generator=GENERATOR,
                generator_version=GENERATOR_VERSION,
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
    )
