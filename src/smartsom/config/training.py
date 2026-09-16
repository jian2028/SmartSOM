"""Training preparation and backend-independent episode materialization."""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from smartsom.config.codec import ConfigurationError, canonical_json, digest
from smartsom.config.materialization import (
    materialize_arrivals,
    materialize_machine_events,
    materialize_processing_times,
    materialize_quality,
)
from smartsom.config.models import (
    AlgorithmFile,
    ArrivalProvenance,
    GeneratedArrivals,
    GeneratedMachineEvents,
    GeneratedProcessingTimes,
    GeneratedQuality,
    LearningAlgorithm,
    MachineEventFile,
    MachineEventProvenance,
    ProcessingProvenance,
    ProcessingTimeFile,
    QualityFile,
    QualityProvenance,
    RunSpec,
    TrainingRunSpec,
)
from smartsom.config.resolver import (
    ResolvedRun,
    SourceFile,
    _reference,
    _resolve_run_spec,
)
from smartsom.config.seeds import NamedSeed, derive_seeds
from smartsom.learning.episode import EpisodeInput
from smartsom.learning.projection import validate_capacity

EPISODE_SEED_VERSION = "smartsom.training-episode/v1"


def episode_root(root: int, index: int) -> int:
    if (
        type(root) is not int
        or not 0 <= root < 2**64
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("invalid training root or episode index")
    return int.from_bytes(
        hashlib.sha256(
            canonical_json([EPISODE_SEED_VERSION, root, index]).encode()
        ).digest()[:8],
        "big",
    )


def framework_seed(root: int, provider: str) -> int:
    return (
        int.from_bytes(
            hashlib.sha256(
                canonical_json(["smartsom.learner-rng/v1", root, provider]).encode()
            ).digest()[:8],
            "big",
        )
        % 2**31
    )


def episode_input(resolved: ResolvedRun) -> EpisodeInput:
    return EpisodeInput(
        resolved.factory,
        resolved.workload,
        resolved.arrivals,
        resolved.scenario.decision_trigger,
        resolved.processing_times,
        resolved.machine_events,
        resolved.transport_enabled,
        resolved.buffers_enabled,
        resolved.holding_buffer_enabled,
        resolved.quality,
        resolved.scenario.quality.probability_visibility
        if resolved.scenario.quality
        else "public",
    )


@dataclass(frozen=True, slots=True)
class TrainingEpisode:
    index: int
    root_seed: int
    seeds: tuple[NamedSeed, ...]
    input: EpisodeInput
    arrival_provenance: ArrivalProvenance | None = None
    processing_provenance: ProcessingProvenance | None = None
    machine_event_provenance: MachineEventProvenance | None = None
    quality_provenance: QualityProvenance | None = None

    @property
    def input_sha256(self):
        return digest(self.input)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRun:
    run: TrainingRunSpec
    algorithm: AlgorithmFile
    base: ResolvedRun
    sources: tuple[SourceFile, ...]
    framework_seed: int
    episode_seed_version: str = EPISODE_SEED_VERSION

    def episode(self, index: int, *, root_seed: int | None = None) -> TrainingEpisode:
        """Pure materialization; never reread files or regenerate the base workload."""
        base, scenario = self.base, self.base.scenario
        root = episode_root(self.run.seed, index) if root_seed is None else root_seed
        seeds = derive_seeds(root, generated=False, quality=base.quality is not None)
        values = {s.domain: s.value for s in seeds}
        arrivals = materialize_arrivals(
            base.workload,
            scenario.arrivals.profile
            if isinstance(scenario.arrivals, GeneratedArrivals)
            else base.arrivals,
            values["demand"],
            base.arrival_provenance,
        )
        processing = materialize_processing_times(
            base.workload,
            scenario.processing_time.profile
            if isinstance(scenario.processing_time, GeneratedProcessingTimes)
            else ProcessingTimeFile(
                schema="smartsom.processing-times/v1",
                processing_times=base.processing_times,
                provenance=base.processing_provenance,
            )
            if base.processing_times
            else None,
            values["processing_time"],
        )
        machines = materialize_machine_events(
            base.factory,
            scenario.machine_events.profile
            if isinstance(scenario.machine_events, GeneratedMachineEvents)
            else MachineEventFile(
                schema="smartsom.machine-events/v1",
                machine_events=base.machine_events,
                provenance=base.machine_event_provenance,
            )
            if base.machine_events is not None
            else None,
            values["machine_events"],
        )
        quality = materialize_quality(
            base.factory,
            base.workload,
            processing.plan,
            scenario.quality
            if isinstance(scenario.quality, GeneratedQuality)
            else QualityFile(
                schema="smartsom.quality-draws/v1",
                draws=base.quality.draws,
                provenance=base.quality_provenance,
            )
            if base.quality
            else None,
            values.get("quality"),
        )
        consumed = {
            "demand": arrivals.seed_consumed,
            "processing_time": processing.seed_consumed,
            "machine_events": machines.seed_consumed,
            "quality": quality.seed_consumed,
        }
        seeds = tuple(replace(s, consumed=consumed.get(s.domain, False)) for s in seeds)
        inputs = replace(
            episode_input(base),
            arrivals=arrivals.plan,
            processing_times=processing.plan,
            machine_events=machines.plan,
            quality=quality.plan,
        )
        return TrainingEpisode(
            index,
            root,
            seeds,
            inputs,
            arrivals.provenance,
            processing.provenance,
            machines.provenance,
            quality.provenance,
        )


def resolve_training_run(path: str | Path):
    from smartsom.config.experiment import load_config, prepare

    return prepare(load_config(path), training=True)


def resolve_training_spec(
    run: TrainingRunSpec,
    path: Path,
    algorithm: AlgorithmFile,
    *,
    sources: tuple[SourceFile, ...] = (),
    algorithm_source: SourceFile | None = None,
    scenario_override=None,
    require_dependencies: bool = True,
) -> ResolvedTrainingRun:
    """Prepare a typed experiment without writing an intermediate authoring file."""
    algorithm_path = _reference(path, run.algorithm)
    selected = algorithm.algorithm
    if (
        not isinstance(selected, LearningAlgorithm)
        or selected.checkpoint is not None
        or selected.checkpoint_sha256 is not None
    ):
        raise ConfigurationError(
            "train requires a learning provider without a checkpoint"
        )
    if run.budget.environment_steps % selected.parameters.n_steps:
        raise ConfigurationError(
            "training budget must contain whole PPO rollouts; no implicit overshoot"
        )
    neutral = AlgorithmFile.model_validate_json(
        '{"schema":"smartsom.algorithm/v1","algorithm":{"provider":"builtin.first_feasible"}}'
    )
    base = _resolve_run_spec(
        RunSpec(
            schema="smartsom.run/v1",
            scenario=run.scenario,
            algorithm=run.algorithm,
            seed=run.seed,
            output_root=run.output_root,
        ),
        path,
        list(sources),
        algorithm_override=neutral,
        scenario_override=scenario_override,
    )
    if base.scenario.visibility != "decision_context":
        raise ConfigurationError("training requires public decision_context visibility")
    validate_capacity(selected.projection, base.workload, base.quality)
    from smartsom.learning.checkpoint import require_backend

    if require_dependencies:
        require_backend(selected.provider)
    return ResolvedTrainingRun(
        run.model_copy(
            update={
                "scenario": base.run.scenario,
                "algorithm": str(algorithm_path),
                "output_root": base.run.output_root,
            }
        ),
        algorithm,
        base,
        (*base.sources, *((algorithm_source,) if algorithm_source else ())),
        framework_seed(run.seed, selected.provider),
    )
