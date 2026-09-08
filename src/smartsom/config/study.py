"""Versioned Cartesian studies; paired worlds are materialized before binding."""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.algorithm_binding import (
    bind_algorithm,
    validate_algorithm_references,
)
from smartsom.config.codec import (
    ConfigurationError,
    canonical_json,
    digest,
    primitive,
    read_model,
)
from smartsom.config.models import (
    AlgorithmFile,
    DispatchRuleAlgorithm,
    EpisodeBudget,
    RecordingSpec,
    Reference,
    RunBudget,
    RunSpec,
    ScenarioFile,
    Seed,
    StrictModel,
)
from smartsom.config.resolver import (
    ResolvedRun,
    SourceFile,
    StudySeedOrigin,
    _reference,
    _resolve_run_spec,
)
from smartsom.config.seeds import derive_seeds
from smartsom.modules.quality import prepare_quality


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
    resolved: ResolvedRun


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


def _ablate(base: ResolvedRun, variant: StudyVariant) -> ResolvedRun:
    scenario = primitive(base.scenario)
    updates = {}
    attributes = {
        "arrivals": ("arrivals", "arrival_provenance", "arrivals_sha256"),
        "processing_time": (
            "processing_times",
            "processing_provenance",
            "processing_times_sha256",
        ),
        "machine_events": (
            "machine_events",
            "machine_event_provenance",
            "machine_events_sha256",
        ),
        "quality": (
            "quality",
            "quality_provenance",
            "quality_draws_sha256",
            "quality_modes_sha256",
        ),
    }
    for name in variant.disable:
        if scenario.get(name) is None:
            raise ValueError(f"ablation disables an inactive module: {name}")
        scenario[name] = None
        if name in ("transport", "buffers", "holding_buffer"):
            updates[f"{name}_enabled"] = False
            updates[f"{name}_sha256"] = None
        else:
            updates.update(dict.fromkeys(attributes[name]))
    scenario = ScenarioFile.model_validate_json(canonical_json(scenario))
    if (
        "processing_time" in variant.disable
        and base.quality
        and "quality" not in variant.disable
    ):
        quality = prepare_quality(base.factory, base.workload, base.quality.draws)
        updates.update(quality=quality, quality_modes_sha256=digest(quality.modes))
    domains = {
        "arrivals": "demand",
        "processing_time": "processing_time",
        "machine_events": "machine_events",
        "quality": "quality",
    }
    disabled_domains = {domains[x] for x in variant.disable if x in domains}
    return replace(
        base,
        scenario=scenario,
        seeds=tuple(
            replace(s, consumed=False) if s.domain in disabled_domains else s
            for s in base.seeds
        ),
        **updates,
    )


def resolve_study(path: str | Path) -> ResolvedStudy:
    path = Path(path).resolve()
    spec, sha = read_model(path, StudySpec)
    output = str(_reference(path, spec.output_root))
    sources = [SourceFile("study", path, sha)]
    algorithms = {}
    for row in spec.algorithms:
        target = _reference(path, row.config)
        algorithm, sha = read_model(target, AlgorithmFile)
        from smartsom.learning.checkpoint import resolve_checkpoint_reference

        algorithm = resolve_checkpoint_reference(algorithm, target)
        source = SourceFile("algorithm", target, sha)
        algorithms[row.id] = (algorithm, source, row.budget)
        sources.append(source)
    entries = []
    neutral = AlgorithmFile(
        schema="smartsom.algorithm/v1",
        algorithm=DispatchRuleAlgorithm(provider="builtin.first_feasible"),
    )
    try:
        for case in sorted(spec.cases, key=lambda x: x.id):
            for replication in range(spec.replications):
                world, _ = study_roots(spec.seed, case.id, replication, "")
                run = RunSpec(
                    schema="smartsom.run/v1",
                    scenario=str(_reference(path, case.scenario)),
                    algorithm="__materialization__",
                    seed=world,
                    output_root=output,
                    recording=spec.recording,
                )
                base = _resolve_run_spec(run, path, [], algorithm_override=neutral)
                sources.extend(base.sources)
                for variant in sorted(spec.variants, key=lambda x: x.id):
                    variant_input = _ablate(base, variant)
                    for algorithm_id, (algorithm, source, budget) in sorted(
                        algorithms.items()
                    ):
                        _, algorithm_root = study_roots(
                            spec.seed, case.id, replication, algorithm_id
                        )
                        algorithm_seeds = {
                            s.domain: s
                            for s in derive_seeds(
                                algorithm_root,
                                generated=False,
                                solver=algorithm.algorithm.interface_kind
                                == "offline_solver",
                            )
                        }
                        seeds = tuple(
                            algorithm_seeds[s.domain]
                            if s.domain in ("algorithm", "solver")
                            else s
                            for s in variant_input.seeds
                        )
                        run = bind_algorithm(
                            variant_input.run.model_copy(
                                update={"algorithm": str(source.path), "budget": budget}
                            ),
                            variant_input.scenario,
                            algorithm,
                        )
                        resolved = replace(
                            variant_input,
                            run=run,
                            algorithm=algorithm,
                            seeds=seeds,
                            sources=(*variant_input.sources, source),
                            study_seed_origin=StudySeedOrigin(
                                spec.seed,
                                case.id,
                                replication,
                                algorithm_id,
                                world,
                                algorithm_root,
                            ),
                        )
                        validate_algorithm_references(
                            algorithm,
                            resolved.workload,
                            resolved.factory,
                            transport_enabled=resolved.transport_enabled,
                            buffers_enabled=resolved.buffers_enabled,
                            holding_buffer_enabled=resolved.holding_buffer_enabled,
                            quality=resolved.quality,
                        )
                        from smartsom.learning.checkpoint import validate_checkpoint

                        validate_checkpoint(resolved)
                        identity = digest(
                            [
                                case.id,
                                replication,
                                variant.id,
                                algorithm_id,
                                semantic_run(resolved),
                            ]
                        )
                        entries.append(
                            PlanEntry(
                                identity,
                                case.id,
                                algorithm_id,
                                replication,
                                variant.id,
                                tuple(sorted(variant.disable)),
                                resolved,
                            )
                        )
    except ValueError as exc:
        raise ConfigurationError(f"{path}: study preparation: {exc}") from exc
    return ResolvedStudy(
        spec.model_copy(update={"output_root": output}),
        tuple(sorted(set(sources), key=lambda s: (s.role, str(s.path)))),
        tuple(entries),
        digest([e.entry_id for e in entries]),
    )
