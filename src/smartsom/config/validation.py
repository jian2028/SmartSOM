"""Freeze explicit validation worlds independently of training and source locations."""

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter, field_serializer, field_validator

from smartsom.config.codec import ConfigurationError, canonical_json, digest, primitive
from smartsom.config.models import StrictModel
from smartsom.config.production import scenario_from_snapshot
from smartsom.config.study import study_roots
from smartsom.domain.production import ProductionScenario
from smartsom.learning.production_contract import factory_identity


class ValidationInput(StrictModel):
    input_id: str
    case_id: str
    world_seed: int
    replication: int
    episode: ProductionScenario
    source: str
    source_sha256: str
    reference: str

    @field_validator("episode", mode="before")
    @classmethod
    def decode_episode(cls, value):
        return scenario_from_snapshot(value) if isinstance(value, dict) else value

    @field_serializer("episode")
    def encode_episode(self, value):
        return primitive(value)


def read_validation_inputs(value):
    try:
        inputs = TypeAdapter(tuple[ValidationInput, ...]).validate_json(value)
        if not inputs or len({item.input_id for item in inputs}) != len(inputs):
            raise ValueError("validation inputs are empty or duplicated")
        for item in inputs:
            if item.input_id != digest(
                [item.case_id, item.world_seed, item.replication, item.episode]
            ):
                raise ValueError("validation input identity mismatch")
            if item.episode.seed != item.world_seed:
                raise ValueError("validation episode seed identity mismatch")
        return tuple(
            {name: getattr(item, name) for name in type(item).model_fields}
            for item in inputs
        )
    except ValueError as exc:
        raise ConfigurationError(f"invalid validation input identity: {exc}") from exc


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


def freeze_validation_cases(recipe, options):
    """Generate each case once, then freeze one world per paired validation seed."""
    from smartsom.config.experiment import ExperimentConfig
    from smartsom.config.production import prepare_experiment

    entries, cases = [], set()
    expected = factory_identity(recipe.scenario.factory)
    for reference in options.scenarios:
        path = Path(reference).expanduser().resolve()
        if path.is_dir():
            path /= "scenario.yaml"
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            config = ExperimentConfig(
                scenario=str(path),
                seed=0,
                algorithm={"source": "__frozen_validation_algorithm__"},
            )
            config.validation.enabled = False
            base = prepare_experiment(
                config, training=False, frozen_algorithm=recipe.algorithm
            ).resolved
            settings = json.loads(base.settings_json)
            settings.pop("factory", None)
            settings.pop("workload", None)
            case_id = f"scenario-{digest([base.scenario, settings])[:16]}"
            if case_id in cases:
                raise ValueError("duplicate validation scenario")
            cases.add(case_id)
            if (
                factory_identity(base.scenario.factory) != expected
                or base.scenario.quality_probability_visibility
                != recipe.scenario.quality_probability_visibility
            ):
                raise ValueError(
                    "validation scenario is incompatible with the model structure"
                )
            for replication in range(options.replications):
                world, _ = study_roots(options.seed, case_id, replication, "")
                episode = base.episode(world)
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
    validate_frozen_cases(value, options)
    return value
