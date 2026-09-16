"""Versioned Cartesian studies; paired worlds are materialized before binding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.codec import (
    ConfigurationError,
    canonical_json,
    digest,
    primitive,
    read_model,
)
from smartsom.config.models import (
    AlgorithmFile,
    EpisodeBudget,
    RecordingSpec,
    Reference,
    RunBudget,
    Seed,
    StrictModel,
)
from smartsom.config.production import prepare_experiment, recipe_identity
from smartsom.config.resolver import (
    ResolvedRun,
    SourceFile,
    _reference,
)

if TYPE_CHECKING:
    from smartsom.config.experiment import PreparedExperiment


class StudyCase(StrictModel):
    id: Reference
    scenario: Reference


class StudyAlgorithm(StrictModel):
    id: Reference
    config: Reference
    budget: RunBudget | EpisodeBudget | None = None


Module = Literal[
    "arrivals",
    "processing_time",
    "machine_events",
    "transport",
    "buffers",
    "quality",
    "holding_buffer",
]


class StudyVariant(StrictModel):
    id: Reference
    disable: tuple[Module, ...] = ()

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.disable)) != len(self.disable):
            raise ValueError("duplicate disabled module")
        return self


class StudySpec(StrictModel):
    schema_id: Literal["smartsom.study/v1"] = Field(alias="schema")
    seed: Seed
    replications: Annotated[int, Field(gt=0)]
    cases: Annotated[tuple[StudyCase, ...], Field(min_length=1)]
    algorithms: Annotated[tuple[StudyAlgorithm, ...], Field(min_length=1)]
    variants: Annotated[tuple[StudyVariant, ...], Field(min_length=1)] = (
        StudyVariant(id="control"),
    )
    objective: Literal["makespan"] = "makespan"
    output_root: Reference
    recording: RecordingSpec = RecordingSpec(observations="hash")

    @model_validator(mode="after")
    def unique_ids(self):
        for name in ("cases", "algorithms", "variants"):
            values = getattr(self, name)
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"duplicate {name} ID")
        return self


@dataclass(frozen=True, slots=True)
class PlanEntry:
    entry_id: str
    case_id: str
    algorithm_id: str
    replication: int
    variant_id: str
    disabled: tuple[str, ...]
    resolved: PreparedExperiment


@dataclass(frozen=True, slots=True)
class ResolvedStudy:
    spec: StudySpec
    sources: tuple[SourceFile, ...]
    entries: tuple[PlanEntry, ...]
    plan_sha256: str


def study_roots(
    root: int, case_id: str, replication: int, algorithm_id: str
) -> tuple[int, int]:
    def hashed(parts):
        return int.from_bytes(
            hashlib.sha256(canonical_json(parts).encode()).digest()[:8], "big"
        )

    world = hashed(["smartsom.study-world/v1", root, case_id, replication])
    algorithm = hashed(["smartsom.study-algorithm/v1", world, algorithm_id])
    return world, algorithm


def semantic_run(resolved: ResolvedRun) -> dict:
    """Exclude provenance locations and recording policy from scientific identity."""
    from smartsom.config.experiment import ExperimentConfig, PreparedExperiment

    if isinstance(resolved, PreparedExperiment):
        config = ExperimentConfig.model_validate_json(resolved.config_json)
        return recipe_identity(resolved.resolved, config, resolved.validation_json)
    scenario = primitive(resolved.scenario)
    scenario.pop("factory")
    scenario["workload"].pop("path")
    for field in ("arrivals", "processing_time", "machine_events", "quality"):
        if scenario.get(field):
            scenario[field].pop("path", None)
    return {
        "scenario": scenario,
        "algorithm": primitive(resolved.algorithm),
        "objective": resolved.run.objective,
        "budget": primitive(resolved.run.budget),
        "seeds": primitive(resolved.seeds),
        "seed_origin": primitive(resolved.study_seed_origin),
        **{
            key: getattr(resolved, key)
            for key in (
                "factory_sha256",
                "workload_sha256",
                "arrivals_sha256",
                "processing_times_sha256",
                "machine_events_sha256",
                "quality_draws_sha256",
                "quality_modes_sha256",
            )
        },
    }


def _ablate(prepared, variant):
    """Ablate stochastic inputs; physical facilities require an explicit factory case."""
    import json

    from smartsom.config.experiment import ExperimentConfig

    recipe = prepared.resolved
    settings = json.loads(recipe.settings_json)
    updates = {}
    for name in variant.disable:
        if name == "arrivals" and settings.get("arrivals"):
            updates["arrivals"] = None
        elif name == "processing_time" and (
            settings["processing_low"] != 1
            or settings["processing_high"] != 1
            or settings.get("processing_samples")
        ):
            updates.update(processing_low=1, processing_high=1, processing_samples=[])
        elif name == "machine_events" and (
            settings["outages"] or settings["outage_profiles"]
        ):
            updates.update(outages=[], outage_profiles=[])
        elif name in {"transport", "buffers", "holding_buffer", "quality"}:
            raise ConfigurationError(
                f"{name} is a physical facility in the grid model; provide an explicit factory/scenario case instead of a legacy module toggle"
            )
        else:
            raise ConfigurationError(f"ablation disables an inactive module: {name}")
    if not updates:
        return prepared
    settings.update(updates)
    from dataclasses import replace as domain_replace

    from smartsom.config.production import ScenarioFile, WorkloadFile, materialize

    workload = WorkloadFile.model_validate_json(recipe.workload_json)
    if "arrivals" in variant.disable:
        workload = workload.model_copy(
            update={
                "demands": tuple(
                    domain_replace(d, release_at=0, reveal_at=0)
                    for d in workload.demands
                )
            }
        )
    scenario = materialize(
        recipe.scenario.factory,
        workload,
        ScenarioFile.model_validate_json(canonical_json(settings)),
        recipe.scenario.seed,
    )
    recipe = replace(
        recipe,
        scenario_json=canonical_json(scenario),
        settings_json=canonical_json(settings),
        workload_json=canonical_json(workload),
    )
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    return replace(
        prepared,
        resolved=recipe,
        scientific_sha256=digest(recipe_identity(recipe, config)),
    )


def resolve_study(path: str | Path) -> ResolvedStudy:
    """Prepare paired grid worlds using the same experiment boundary as run/train."""
    from smartsom.config.experiment import ExperimentConfig

    path = Path(path).resolve()
    spec, sha = read_model(path, StudySpec)
    output = str(_reference(path, spec.output_root))
    sources = [SourceFile("study", path, sha)]
    entries = []
    try:
        for case in sorted(spec.cases, key=lambda x: x.id):
            for replication in range(spec.replications):
                world, _ = study_roots(spec.seed, case.id, replication, "")
                shared_world = None
                variants = {}
                for row in sorted(spec.algorithms, key=lambda x: x.id):
                    algorithm_path = _reference(path, row.config)
                    _, algorithm_sha = read_model(algorithm_path, AlgorithmFile)
                    sources.append(
                        SourceFile("algorithm", algorithm_path, algorithm_sha)
                    )
                    config = ExperimentConfig.model_validate(
                        {
                            "scenario": str(_reference(path, case.scenario)),
                            "algorithm": {"source": str(algorithm_path)},
                        }
                    )
                    config.seed = world
                    config.output.root = output
                    config.output.name = f"{case.id}-{row.id}-{replication + 1}"
                    config.logging.verbose = False
                    config.logging.debug = spec.recording.debug
                    config.logging.observations = spec.recording.observations
                    config.validation.enabled = False
                    if row.budget is not None:
                        for key, value in primitive(row.budget).items():
                            if value is None:
                                continue
                            if key == "solver_time_limit_seconds":
                                raise ConfigurationError(
                                    "CP-SAT has no grid production adapter"
                                )
                            if key in ("max_decisions", "max_ticks"):
                                setattr(config.training, key, value)
                    prepared = prepare_experiment(
                        config, training=False, frozen_world=shared_world
                    )
                    if shared_world is None:
                        shared_world = prepared.resolved
                        variants = {
                            variant.id: _ablate(prepared, variant).resolved
                            for variant in spec.variants
                        }
                    _, policy_seed = study_roots(
                        spec.seed, case.id, replication, row.id
                    )
                    recipe = replace(prepared.resolved, algorithm_seed=policy_seed)
                    prepared = replace(
                        prepared,
                        resolved=recipe,
                        scientific_sha256=digest(recipe_identity(recipe, config)),
                    )
                    for variant in sorted(spec.variants, key=lambda x: x.id):
                        variant_world = variants[variant.id]
                        variant_recipe = replace(
                            recipe,
                            scenario_json=variant_world.scenario_json,
                            settings_json=variant_world.settings_json,
                            workload_json=variant_world.workload_json,
                        )
                        resolved = replace(
                            prepared,
                            resolved=variant_recipe,
                            scientific_sha256=digest(
                                recipe_identity(variant_recipe, config)
                            ),
                        )
                        identity = digest(
                            [
                                case.id,
                                replication,
                                variant.id,
                                row.id,
                                semantic_run(resolved),
                            ]
                        )
                        entries.append(
                            PlanEntry(
                                identity,
                                case.id,
                                row.id,
                                replication,
                                variant.id,
                                tuple(sorted(variant.disable)),
                                resolved,
                            )
                        )
    except ValueError as exc:
        raise ConfigurationError(f"{path}: study preparation: {exc}") from exc
    entries.sort(key=lambda e: (e.case_id, e.replication, e.variant_id, e.algorithm_id))
    return ResolvedStudy(
        spec.model_copy(update={"output_root": output}),
        tuple(sorted(set(sources), key=lambda source: (source.role, str(source.path)))),
        tuple(entries),
        digest([entry.entry_id for entry in entries]),
    )
