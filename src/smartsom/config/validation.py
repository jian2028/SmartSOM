"""Materialize explicit validation cases once, without starting a simulator."""

import hashlib
from pathlib import Path

from pydantic import TypeAdapter

from smartsom.config.codec import ConfigurationError, canonical_json, digest
from smartsom.config.models import (
    AlgorithmFile,
    DispatchRuleAlgorithm,
    RunSpec,
    StrictModel,
)
from smartsom.config.resolver import _resolve_run_spec
from smartsom.config.study import semantic_run, study_roots
from smartsom.config.training import episode_input
from smartsom.learning.checkpoint import structural_identity
from smartsom.learning.episode import EpisodeInput


class ValidationInput(StrictModel):
    input_id: str
    case_id: str
    world_seed: int
    replication: int
    episode: EpisodeInput
    source: str
    source_sha256: str
    reference: str


def read_validation_inputs(value):
    inputs = TypeAdapter(tuple[ValidationInput, ...]).validate_json(value)
    if not inputs or len({item.input_id for item in inputs}) != len(inputs):
        raise ConfigurationError("validation inputs are empty or duplicated")
    for item in inputs:
        if item.input_id != digest(
            [item.case_id, item.world_seed, item.replication, item.episode]
        ):
            raise ConfigurationError("validation input identity mismatch")
    return tuple(
        {name: getattr(item, name) for name in type(item).model_fields}
        for item in inputs
    )


def validate_frozen_cases(value, options):
    inputs = read_validation_inputs(value)
    if len(inputs) != len(options.scenarios) * options.replications:
        raise ConfigurationError("validation snapshot input coverage mismatch")
    coverage = set()
    for item in inputs:
        key = (item["reference"], item["replication"])
        if (
            key in coverage
            or item["reference"] not in options.scenarios
            or item["replication"] not in range(options.replications)
        ):
            raise ConfigurationError("validation snapshot case/replication mismatch")
        coverage.add(key)
        if (
            study_roots(options.seed, item["case_id"], item["replication"], "")[0]
            != item["world_seed"]
        ):
            raise ConfigurationError("validation snapshot seed authority mismatch")
    return inputs


def freeze_validation_cases(resolved, options):
    """External cases must obey the model's existing structural compatibility."""
    entries, cases = [], set()
    expected = structural_identity(
        resolved.base, resolved.algorithm.algorithm.projection
    )

    def resolve(path, seed):
        return _resolve_run_spec(
            RunSpec(
                schema="smartsom.run/v1",
                scenario=str(path),
                algorithm="__validation_world__",
                seed=seed,
                output_root="runs",
            ),
            path,
            [],
            algorithm_override=AlgorithmFile(
                schema="smartsom.algorithm/v1",
                algorithm=DispatchRuleAlgorithm(provider="builtin.spt"),
            ),
        )

    for reference in options.scenarios:
        path = Path(reference).expanduser().resolve()
        if path.is_dir():
            path /= "scenario.yaml"
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            case_id = f"scenario-{digest(semantic_run(resolve(path, 0)))[:16]}"
            if case_id in cases:
                raise ValueError("duplicate validation scenario")
            cases.add(case_id)
            for replication in range(options.replications):
                world, _ = study_roots(options.seed, case_id, replication, "")
                case = resolve(path, world)
                if (
                    structural_identity(case, resolved.algorithm.algorithm.projection)
                    != expected
                ):
                    raise ValueError(
                        "validation scenario is incompatible with the model structure"
                    )
                episode = episode_input(case)
                entries.append(
                    ValidationInput(
                        input_id=digest([case_id, world, replication, episode]),
                        case_id=case_id,
                        world_seed=world,
                        replication=replication,
                        episode=episode,
                        source=str(path),
                        source_sha256=sha,
                        reference=reference,
                    )
                )
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"validation.scenarios {path}: {exc}") from exc
    value = canonical_json(entries)
    read_validation_inputs(value)
    return value
