"""Resolve reusable production recipes into detached engine inputs."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from smartsom.config.codec import primitive
from smartsom.config.extensions import ExtensionSpec
from smartsom.config.factory_design import load_factory_design_file
from smartsom.domain.production import (
    Demand,
    JointCommand,
    Outage,
    ProcessingSample,
    ProductionScenario,
    ProductionStep,
    QualitySample,
    validate_production_scenario,
)
from smartsom.workloads.fjs import ImportProvenance


class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class AlgorithmConfig(StrictModel):
    provider: Literal[
        "builtin.greedy",
        "builtin.random",
        "builtin.spt",
        "builtin.first_feasible",
        "builtin.scripted",
        "sb3.maskable_ppo",
        "rllib.ppo",
        "rllib.resource_ppo",
    ] = "builtin.greedy"
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = Field(default=0.0003, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    max_jobs: int = Field(default=64, gt=0)
    max_operations_per_job: int = Field(default=64, gt=0)
    max_modes_per_operation: int = Field(default=64, gt=0)
    time_scale: float = Field(default=100.0, gt=0)
    count_scale: float = Field(default=100.0, gt=0)
    checkpoint: str | None = None
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, lt=1)
    entropy_coefficient: float = Field(default=0.01, ge=0)
    batch_size: int = Field(default=64, gt=1)
    n_epochs: int = Field(default=4, gt=0)
    learner_reward_scale: float = Field(default=1.0, gt=0)
    activation: Literal["tanh"] = "tanh"
    extensions: ExtensionSpec | None = None
    quality_mode: str | None = None
    commands: tuple[JointCommand, ...] | None = None

    @model_validator(mode="after")
    def valid_network(self):
        if (self.provider == "builtin.scripted") != (self.commands is not None):
            raise ValueError("commands are required exclusively for builtin.scripted")
        if not self.hidden_sizes or any(n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        return self


class WorkloadProfile(StrictModel):
    jobs: int = Field(default=4, ge=0)
    route: tuple[str, ...] = ()
    operation_types: tuple[str, ...] = ()
    min_operations: int = Field(default=1, gt=0)
    max_operations: int = Field(default=1, gt=0)
    machine_duration_variation: bool = False
    nominal_min: int = Field(default=2, gt=0)
    nominal_max: int = Field(default=5, gt=0)
    due_at: int = Field(default=100, ge=0)
    priority: int = Field(default=1, gt=0)
    input_id: str | None = None

    @model_validator(mode="after")
    def route_source(self):
        if bool(self.route) == bool(self.operation_types):
            raise ValueError(
                "provide a fixed route or a catalog of sampled operation_types"
            )
        if self.min_operations > self.max_operations:
            raise ValueError("operation count bounds are reversed")
        if self.route and (self.min_operations, self.max_operations) != (1, 1):
            raise ValueError("operation count bounds apply only to sampled routes")
        if len(set(self.operation_types)) != len(self.operation_types):
            raise ValueError("duplicate sampled operation type")
        return self


class WorkloadFile(StrictModel):
    schema_id: Literal["smartsom.workload/v2"] = Field(alias="schema")
    demands: tuple[Demand, ...] | None = None
    profile: WorkloadProfile | None = None
    provenance: ImportProvenance | None = None

    @field_validator("demands", mode="before")
    @classmethod
    def duration_tables(cls, data):
        if (
            data is None
            or isinstance(data, tuple)
            and all(isinstance(d, Demand) for d in data)
        ):
            return data
        normalized = normalize_duration_tables({"demands": data})["demands"]
        return TypeAdapter(
            tuple[Demand, ...], config=ConfigDict(extra="forbid")
        ).validate_json(json.dumps(normalized), strict=True)

    @field_serializer("demands")
    def encode_demands(self, value):
        return primitive(value)

    @model_validator(mode="after")
    def one_source(self):
        if (self.demands is None) == (self.profile is None):
            raise ValueError("provide demands or profile, exactly one")
        if self.profile and self.profile.nominal_min > self.profile.nominal_max:
            raise ValueError("nominal_min exceeds nominal_max")
        return self


class ArrivalConfig(StrictModel):
    initial_jobs: int = Field(default=0, ge=0)
    release_min: int = Field(default=0, ge=0)
    release_max: int = Field(default=100, ge=0)
    notice_ticks: int = Field(default=0, ge=0)


class OutageProfile(StrictModel):
    machine_id: str
    mean_uptime_ticks: float = Field(gt=0, allow_inf_nan=False)
    repair_min: int = Field(default=1, gt=0)
    repair_max: int = Field(default=5, gt=0)
    until_tick: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def bounds(self):
        if self.repair_min > self.repair_max:
            raise ValueError("repair bounds are reversed")
        return self


class ScenarioFile(StrictModel):
    schema_id: Literal["smartsom.scenario/v2"] = Field(alias="schema")
    factory: str
    workload: str
    mode: Literal["static", "dynamic"] = "static"
    tick_limit: int = Field(default=1000, gt=0)
    arrivals: ArrivalConfig | None = None
    outages: tuple[Outage, ...] = ()
    outage_profiles: tuple[OutageProfile, ...] = ()
    processing_low: float = Field(default=1, gt=0)
    processing_high: float = Field(default=1, gt=0)
    reward_time_scale: int = Field(default=100, gt=0)
    processing_samples: tuple[ProcessingSample, ...] = ()
    quality_samples: tuple[QualitySample, ...] = ()
    quality_probability_visibility: Literal["public", "hidden"] = "public"


def read_file(path, model):
    import yaml

    from smartsom.config.codec import ConfigurationError, _UniqueLoader

    path = Path(path).expanduser().resolve()
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
        return model.model_validate_json(json.dumps(data))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def named_seed(root, name):
    return int.from_bytes(
        hashlib.sha256(f"production/{root}/{name}".encode()).digest()[:8], "big"
    )


def materialize(factory, workload, raw, seed):
    """Pure seeded materialization from frozen data; never reopen source paths."""
    from dataclasses import replace

    demands = workload.demands
    if demands is None:
        profile = workload.profile
        for kind in (*profile.route, *profile.operation_types):
            if kind not in factory.operation_types:
                raise ValueError(
                    f"workload profile references unknown operation type {kind}"
                )
        rng = random.Random(named_seed(seed, "workload"))

        def steps():
            route = profile.route or tuple(
                rng.choice(profile.operation_types)
                for _ in range(
                    rng.randint(profile.min_operations, profile.max_operations)
                )
            )
            result = []
            for index, kind in enumerate(route):
                nominal = rng.randint(profile.nominal_min, profile.nominal_max)
                overrides = (
                    tuple(
                        (
                            machine.machine_id,
                            rng.randint(profile.nominal_min, profile.nominal_max),
                        )
                        for machine in factory.machines
                        if kind in machine.operation_types
                    )
                    if profile.machine_duration_variation
                    else ()
                )
                result.append(
                    ProductionStep(f"op_{index + 1:03d}", kind, nominal, overrides)
                )
            return tuple(result)

        demands = tuple(
            Demand(
                f"demand_{i + 1:04d}",
                steps(),
                due_at=profile.due_at,
                priority=profile.priority,
                input_id=profile.input_id,
            )
            for i in range(profile.jobs)
        )
    if raw.arrivals is not None:
        if raw.mode != "dynamic":
            raise ValueError("generated arrivals require dynamic mode")
        spec = raw.arrivals
        if spec.release_min > spec.release_max or spec.initial_jobs > len(demands):
            raise ValueError("invalid arrival range or initial_jobs")
        rng = random.Random(named_seed(seed, "arrival"))
        rows = []
        for i, d in enumerate(sorted(demands, key=lambda d: d.demand_id)):
            release = (
                0
                if i < spec.initial_jobs
                else rng.randint(spec.release_min, spec.release_max)
            )
            rows.append(
                replace(
                    d, release_at=release, reveal_at=max(0, release - spec.notice_ticks)
                )
            )
        demands = tuple(rows)
    outages = list(raw.outages)
    if raw.outage_profiles:
        import math

        if raw.outages:
            raise ValueError("choose explicit outages or generated outage_profiles")
        ids = [p.machine_id for p in raw.outage_profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate machine in outage_profiles")
        for profile in sorted(raw.outage_profiles, key=lambda p: p.machine_id):
            if profile.machine_id not in {m.machine_id for m in factory.machines}:
                raise ValueError("outage profile references an unknown machine")
            rng = random.Random(named_seed(seed, f"outage/{profile.machine_id}"))
            repaired = 0
            generation_limit = min(raw.tick_limit, profile.until_tick or raw.tick_limit)
            while repaired < generation_limit:
                start = repaired + max(
                    1, math.ceil(rng.expovariate(1 / profile.mean_uptime_ticks))
                )
                if start >= generation_limit:
                    break
                repaired = start + rng.randint(profile.repair_min, profile.repair_max)
                outages.append(Outage(profile.machine_id, start, repaired))
    scenario = ProductionScenario(
        factory,
        demands,
        raw.mode,
        raw.tick_limit,
        seed,
        tuple(outages),
        raw.processing_low,
        raw.processing_high,
        raw.reward_time_scale,
        raw.processing_samples,
        raw.quality_samples,
        raw.quality_probability_visibility,
    )
    return scenario


def scenario_from_snapshot(data):
    return TypeAdapter(
        tuple[ProductionScenario, ...], config=ConfigDict(extra="forbid")
    ).validate_json(json.dumps([normalize_duration_tables(data)]), strict=True)[0]


def normalize_duration_tables(data):
    """Decode YAML/JSON mappings into the immutable domain representation."""
    import copy

    data = copy.deepcopy(data)
    for demand in data.get("demands") or ():
        if not isinstance(demand, dict):
            continue
        for step in demand.get("steps", ()):
            if not isinstance(step, dict):
                continue
            table = step.get("machine_nominal_ticks", ())
            if isinstance(table, dict):
                step["machine_nominal_ticks"] = list(table.items())
    return data


def frozen_inputs(scenario, algorithm):
    return {"scenario": primitive(scenario), "algorithm": primitive(algorithm)}


@dataclass(frozen=True)
class ProductionRecipe:
    """Frozen physical recipe shared by training, validation and evaluation."""

    scenario_json: str
    settings_json: str
    workload_json: str
    algorithm_json: str
    training_json: str = "{}"
    algorithm_seed: int | None = None
    workload_source_json: str | None = None

    @property
    def scenario(self):
        return scenario_from_snapshot(json.loads(self.scenario_json))

    @property
    def algorithm(self):
        return AlgorithmConfig.model_validate_json(self.algorithm_json)

    def episode(self, seed):
        return materialize(
            self.scenario.factory,
            WorkloadFile.model_validate_json(self.workload_json),
            ScenarioFile.model_validate_json(self.settings_json),
            seed,
        )


def compile_algorithm(config, *, training, require_dependencies=False):
    from smartsom.config.codec import canonical_json, read_model
    from smartsom.config.experiment import bind_training_algorithm
    from smartsom.config.models import AlgorithmFile, LearningAlgorithm

    authored, _ = read_model(Path(config.algorithm.source), AlgorithmFile)
    if authored.algorithm.provider == "pyjobshop.cp_sat":
        raise ValueError("CP-SAT has no grid production adapter")
    if training:
        authored = bind_training_algorithm(config, authored)
    selected = authored.algorithm
    if training and getattr(selected, "checkpoint", None):
        raise ValueError(
            "training requires an algorithm without a checkpoint; use initialize_from for a new experiment"
        )
    if selected.provider == "pyjobshop.cp_sat":
        raise ValueError("CP-SAT has no grid production adapter")
    data = {"provider": selected.provider}
    if selected.provider == "builtin.scripted":
        if selected.parameters.commands is None:
            raise ValueError(
                "historical scripted actions require migration to explicit grid commands; "
                "operation selections do not specify AGV movement or resource actions"
            )
        data["commands"] = primitive(selected.parameters.commands)
    if hasattr(selected.parameters, "quality_mode"):
        data["quality_mode"] = selected.parameters.quality_mode
    if isinstance(selected, LearningAlgorithm):
        if require_dependencies:
            from smartsom.learning.checkpoint import require_backend

            require_backend(selected.provider)
        data.update(
            {k: v for k, v in primitive(selected.parameters).items() if k != "n_steps"}
        )
        data.update(
            max_jobs=selected.projection.max_jobs, checkpoint=selected.checkpoint
        )
        data.update(
            {
                key: getattr(selected.projection, key)
                for key in (
                    "max_operations_per_job",
                    "max_modes_per_operation",
                    "time_scale",
                    "count_scale",
                )
            }
        )
        from smartsom.learning.extensions import bind_extensions

        data["extensions"] = primitive(
            bind_extensions(selected.extensions, selected.provider)
        )
    algorithm = AlgorithmConfig.model_validate_json(canonical_json(data))
    if algorithm.checkpoint is not None:
        from smartsom.experiments.packaging import (
            _checkpoint_files,
            locate_reference,
            model_locator,
        )
        from smartsom.learning.checkpoint import file_hash

        checkpoint = model_locator(
            locate_reference(config.algorithm.source, algorithm.checkpoint)
        )
        manifest = _checkpoint_files(checkpoint)
        if (
            selected.checkpoint_sha256
            and file_hash(checkpoint / "checkpoint.json") != selected.checkpoint_sha256
        ):
            raise ValueError("checkpoint manifest digest mismatch")
        if manifest.get("schema") != "smartsom.production-checkpoint/v1":
            raise ValueError(
                "checkpoint uses incompatible actions/observations; retrain with the grid core"
            )
        recorded = AlgorithmConfig.model_validate_json(
            canonical_json(manifest["algorithm"])
        )
        if recorded.model_copy(update={"checkpoint": None}) != algorithm.model_copy(
            update={"checkpoint": None}
        ):
            raise ValueError(
                "algorithm settings differ from saved checkpoint; use its exported algorithm document"
            )
        algorithm.checkpoint = str(checkpoint)
    return algorithm


def prepare_experiment(
    config,
    *,
    training,
    require_dependencies=False,
    frozen_world=None,
    frozen_algorithm=None,
):
    from smartsom.config.codec import canonical_json, digest
    from smartsom.config.experiment import ExperimentConfig, PreparedExperiment, merge

    origins = config.origins()
    config = ExperimentConfig.model_validate_json(canonical_json(config))
    algorithm = frozen_algorithm or compile_algorithm(
        config, training=training, require_dependencies=require_dependencies
    )
    path = Path(config.scenario).resolve()
    if frozen_world is None:
        raw = read_file(path, ScenarioFile)
        raw = ScenarioFile.model_validate_json(
            canonical_json(merge(primitive(raw), config.scenario_overrides))
        )
        factory_file, _ = load_factory_design_file(path.parent / raw.factory)
        factory = factory_file.factory
        workload = read_file(path.parent / raw.workload, WorkloadFile)
    else:
        if (
            training
            or config.scenario_overrides
            or config.seed != frozen_world.scenario.seed
        ):
            raise ValueError(
                "a paired world requires unchanged scenario controls and seed"
            )
        raw = ScenarioFile.model_validate_json(frozen_world.settings_json)
        factory = frozen_world.scenario.factory
        workload = WorkloadFile.model_validate_json(frozen_world.workload_json)
    if algorithm.quality_mode is not None and any(
        algorithm.quality_mode
        not in {mode.quality_mode_id for mode in machine.quality_modes}
        for machine in factory.machines
    ):
        raise ValueError("every machine must support the selected fixed quality_mode")
    scenario = (
        materialize(factory, workload, raw, config.seed)
        if frozen_world is None
        else frozen_world.scenario
    )
    if not algorithm.provider.startswith("builtin.") and (
        any(
            len(demand.steps) > algorithm.max_operations_per_job
            for demand in scenario.demands
        )
        or any(
            len(machine.quality_modes) > algorithm.max_modes_per_operation
            for machine in scenario.factory.machines
        )
    ):
        raise ValueError(
            "factory/workload exceeds configured learning projection capacity"
        )
    validate_production_scenario(scenario)
    if algorithm.checkpoint is not None:
        from smartsom.learning.production_contract import validate_checkpoint_manifest

        metadata = json.loads(
            (Path(algorithm.checkpoint) / "checkpoint.json").read_text()
        )
        validate_checkpoint_manifest(metadata, scenario, algorithm)
    if algorithm.commands is not None:
        groups = {
            "agvs": {v.agv_id for v in scenario.factory.agvs},
            "machines": {v.machine_id for v in scenario.factory.machines},
            "quality": {
                v.inspection_station_id for v in scenario.factory.inspection_stations
            },
            "rankings": {
                v.buffer_id
                for v in scenario.factory.buffers
                if v.role != "system_output"
            }
            | {v.inspection_station_id for v in scenario.factory.inspection_stations},
        }
        for command in algorithm.commands:
            for group, known in groups.items():
                unknown = {key for key, _ in getattr(command, group)} - known
                if unknown:
                    raise ValueError(
                        f"script references unknown {group}: {sorted(unknown)}"
                    )
    # Keep the generated base jobs fixed; episode seeds vary disturbances only.
    frozen_workload = primitive(
        workload.model_copy(update={"demands": scenario.demands, "profile": None})
    )
    recipe = ProductionRecipe(
        canonical_json(scenario),
        canonical_json(raw),
        canonical_json(frozen_workload),
        canonical_json(algorithm),
        canonical_json(config.training),
        workload_source_json=(
            frozen_world.workload_source_json
            if frozen_world
            else canonical_json(workload)
        ),
    )
    validation_json = None
    if config.validation.enabled and config.validation.scenarios:
        from smartsom.config.validation import freeze_validation_cases

        validation_json = freeze_validation_cases(recipe, config.validation)
    return PreparedExperiment(
        canonical_json(config),
        canonical_json(origins),
        recipe,
        digest(recipe_identity(recipe, config, validation_json)),
        validation_json,
    )


def recipe_identity(recipe, config, validation_json=None):
    from smartsom.config.codec import primitive

    settings = json.loads(recipe.settings_json)
    settings.pop("factory", None)
    settings.pop("workload", None)
    return {
        "scenario": json.loads(recipe.scenario_json),
        "settings": settings,
        "workload": json.loads(recipe.workload_json),
        "workload_source": json.loads(recipe.workload_source_json)
        if recipe.workload_source_json
        else None,
        "algorithm": json.loads(recipe.algorithm_json),
        "algorithm_seed": recipe.algorithm_seed,
        "training": json.loads(recipe.training_json),
        "runtime": primitive(config.runtime),
        "validation": [
            {
                key: value
                for key, value in row.items()
                if key not in {"reference", "source", "source_sha256"}
            }
            for row in json.loads(validation_json)
        ]
        if validation_json
        else None,
    }


def prepared_from_data(data):
    """Read and verify a detached grid experiment snapshot."""
    from smartsom.config.codec import digest
    from smartsom.config.experiment import ExperimentConfig, PreparedExperiment

    payload = dict(data)
    if payload.pop("schema", None) != "smartsom.prepared-grid-experiment/v1":
        raise ValueError("expected a frozen grid experiment snapshot")
    payload["resolved"] = ProductionRecipe(**payload["resolved"])
    prepared = PreparedExperiment(**payload)
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    if (
        digest(recipe_identity(prepared.resolved, config, prepared.validation_json))
        != prepared.scientific_sha256
    ):
        raise ValueError("frozen run scientific identity mismatch")
    return prepared


def prepared_from_run_record(data):
    """Reconstruct detached public run inputs from the two-file record format."""
    from smartsom.config.codec import canonical_json
    from smartsom.config.experiment import ExperimentConfig

    if data.get("schema") != "smartsom.production-run/v1":
        raise ValueError("expected a grid run record")
    experiment = data.get("experiment")
    if not experiment:
        raise ValueError("run has no frozen public experiment configuration")
    config = ExperimentConfig.model_validate_json(canonical_json(experiment["config"]))
    metadata = data["input_metadata"]
    source = WorkloadFile.model_validate_json(
        canonical_json(metadata["workload_authoring"])
    )
    scenario = scenario_from_snapshot(data["inputs"]["scenario"])
    workload = source.model_copy(update={"demands": scenario.demands, "profile": None})
    recipe = ProductionRecipe(
        canonical_json(scenario),
        canonical_json(metadata["scenario_authoring"]),
        canonical_json(workload),
        canonical_json(data["inputs"]["algorithm"]),
        canonical_json(config.training),
        experiment.get("algorithm_seed"),
        canonical_json(source),
    )
    return prepared_from_data(
        {
            "schema": "smartsom.prepared-grid-experiment/v1",
            "config_json": canonical_json(config),
            "origins_json": canonical_json(experiment["origins"]),
            "resolved": primitive(recipe),
            "scientific_sha256": experiment["scientific_sha256"],
            "validation_json": canonical_json(experiment["validation_inputs"])
            if experiment.get("validation_inputs") is not None
            else None,
        }
    )


def checkpoint_algorithm_document(metadata, checkpoint, manifest_sha256):
    """Preserve the reusable Algorithm YAML/JSON envelope for saved grid weights."""
    from smartsom.config.codec import canonical_json
    from smartsom.config.models import (
        AlgorithmFile,
        PPOParameters,
        ResourcePPOParameters,
    )

    selected = AlgorithmConfig.model_validate_json(
        canonical_json(metadata["algorithm"])
    )
    parameter_type = (
        ResourcePPOParameters
        if selected.provider == "rllib.resource_ppo"
        else PPOParameters
    )
    parameters = {
        key: getattr(selected, key)
        for key in parameter_type.model_fields
        if key != "n_steps"
    }
    parameters["n_steps"] = max(256, selected.batch_size)
    parameters["n_steps"] -= parameters["n_steps"] % selected.batch_size
    document = {
        "schema": "smartsom.algorithm/v1",
        "algorithm": {
            "provider": selected.provider,
            "projection": {
                key: getattr(selected, key)
                for key in (
                    "max_jobs",
                    "max_operations_per_job",
                    "max_modes_per_operation",
                    "time_scale",
                    "count_scale",
                )
            },
            "parameters": parameters,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": manifest_sha256,
            "extensions": primitive(selected.extensions),
        },
    }
    return primitive(AlgorithmFile.model_validate_json(canonical_json(document)))
