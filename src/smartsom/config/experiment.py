"""Editable experiment recipes resolved through the existing scientific boundary."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from smartsom.config.codec import (
    ConfigurationError,
    _UniqueLoader,
    canonical_json,
    digest,
    primitive,
    read_model,
)
from smartsom.config.extensions import ExtensionSpec
from smartsom.config.models import (
    AlgorithmFile,
    CPSatAlgorithm,
    EpisodeBudget,
    LearningAlgorithm,
    RecordingSpec,
    RunBudget,
    RunSpec,
    ScenarioFile,
    TrainingBudget,
    TrainingRunSpec,
)
from smartsom.config.resolver import SourceFile, _resolve_run_spec
from smartsom.config.study import semantic_run
from smartsom.config.training import ResolvedTrainingRun, resolve_training_spec

Positive = Annotated[int, Field(gt=0)]
Seed = Annotated[int, Field(ge=0, lt=2**64)]
PRESET_ROOT = Path(__file__).with_name("presets")
PRESETS = {
    "marl_micro": ("learning_marl", "Resource MARL: shared machine and AGV PPO"),
    "rllib_micro": ("learning_rllib", "Centralized RLlib PPO"),
    "sb3_micro": ("learning_sb3", "Centralized SB3 MaskablePPO"),
    "competition": ("competition", "Small hand-checkable scheduling example"),
    "ft06_cp": ("ft06_cp", "Static FT06 with optional CP-SAT"),
}


class EditableModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class TrainingOptions(EditableModel):
    total_steps: Positive = 4096
    steps_per_update: Annotated[int, Field(gt=1)] = 256
    max_decisions: Positive = 1024
    max_ticks: Positive = 10000


class RuntimeOptions(EditableModel):
    num_envs: Positive = 1
    sampling_processes: Annotated[int, Field(ge=0)] = 0
    numerical_threads: Positive = 1
    device: Literal["cpu", "cuda"] = "cpu"
    max_concurrent: Positive = 1


class SolverOptions(EditableModel):
    time_limit_seconds: Annotated[float, Field(gt=0)] = 60.0


class CategoricalSpace(EditableModel):
    type: Literal["categorical"] = "categorical"
    choices: Annotated[tuple[Any, ...], Field(min_length=1)]


class FloatSpace(EditableModel):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def valid_range(self):
        if self.low > self.high or self.log and self.low <= 0:
            raise ValueError(
                "search range requires low <= high and positive log bounds"
            )
        return self


class IntSpace(EditableModel):
    type: Literal["int"] = "int"
    low: int
    high: int
    step: Positive = 1
    log: bool = False

    @model_validator(mode="after")
    def valid_range(self):
        if self.low > self.high or self.log and (self.low <= 0 or self.step != 1):
            raise ValueError("invalid integer search range or logarithmic step")
        if (self.high - self.low) % self.step:
            raise ValueError("integer search bounds must contain whole steps")
        return self


class SearchOptions(EditableModel):
    method: Literal["grid", "random", "optuna"] = "grid"
    space: dict[
        str,
        Annotated[
            CategoricalSpace | FloatSpace | IntSpace, Field(discriminator="type")
        ],
    ] = Field(default_factory=dict)
    trials: Positive | None = None
    seed: Seed = 0
    seeds: tuple[Seed, ...] = (101,)
    objective: Literal["makespan", "return", "passing_rate"] | None = None
    direction: Literal["min", "max"] = "min"
    failure_policy: Literal["all_complete"] | None = None
    pruning: bool = False


class AlgorithmOptions(EditableModel):
    source: str
    extensions: ExtensionSpec | None = None
    learning_rate: Annotated[float, Field(gt=0)] = 0.0003
    gamma: Annotated[float, Field(gt=0, le=1)] = 1.0
    gae_lambda: Annotated[float, Field(gt=0, le=1)] = 0.95
    clip_range: Annotated[float, Field(gt=0, lt=1)] = 0.2
    entropy_coefficient: Annotated[float, Field(ge=0)] = 0.0
    batch_size: Annotated[int, Field(gt=1)] = 64
    n_epochs: Positive = 10
    hidden_sizes: tuple[Positive, ...] = (64, 64)
    activation: Literal["tanh"] = "tanh"
    learner_reward_scale: Annotated[float, Field(gt=0)] = 1.0


class ValidationOptions(EditableModel):
    enabled: bool = True
    every_updates: Positive = 4
    seed: Seed = 303
    replications: Positive = 5
    scenarios: tuple[str, ...] = ()
    deterministic: bool = True
    full_replay: bool = False
    best_mode: Literal["completion_first", "all_complete", "custom"] = (
        "completion_first"
    )
    metric: str = "makespan"
    direction: Literal["min", "max"] = "min"
    failure_policy: str | None = None
    patience: Positive | None = None
    min_delta: Annotated[float, Field(ge=0)] = 0.0


class EvaluationOptions(EditableModel):
    seed: Seed = 202
    replications: Positive = 5
    deterministic: bool = True
    full_replay: bool = True
    checkpoint: Literal["last", "best"] = "last"
    baselines: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()


class CheckpointOptions(EditableModel):
    every_updates: Positive | None = 4
    keep_last: Positive = 2
    save_last: bool = True
    save_best: bool = True


class LoggingOptions(EditableModel):
    progress: Literal["auto", "on", "off"] = "auto"
    verbose: Literal[0, 1, 2] = 1
    format: Literal["text", "json"] = "text"
    every_seconds: Annotated[float, Field(gt=0)] = 1.0
    observations: Literal["hash", "full"] = "hash"
    debug: bool = False
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str | None = None
    wandb_mode: Literal["online", "offline"] = "online"


class OutputOptions(EditableModel):
    root: str = "runs"
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")] = "experiment"
    tags: tuple[str, ...] = ()


class ExperimentConfig(EditableModel):
    schema_id: Literal["smartsom.experiment-config/v2"] = Field(
        default="smartsom.experiment-config/v2", alias="schema"
    )
    scenario: str
    scenario_overrides: dict[str, Any] = Field(default_factory=dict)
    seed: Seed = 101
    algorithm: AlgorithmOptions
    training: TrainingOptions = Field(default_factory=TrainingOptions)
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
    solver: SolverOptions = Field(default_factory=SolverOptions)
    search: SearchOptions = Field(default_factory=SearchOptions)
    validation: ValidationOptions = Field(default_factory=ValidationOptions)
    evaluation: EvaluationOptions = Field(default_factory=EvaluationOptions)
    checkpointing: CheckpointOptions = Field(default_factory=CheckpointOptions)
    logging: LoggingOptions = Field(default_factory=LoggingOptions)
    output: OutputOptions = Field(default_factory=OutputOptions)
    _origins: dict[str, str] = PrivateAttr(default_factory=dict)
    _baseline: dict = PrivateAttr(default_factory=dict)
    _owner: Path = PrivateAttr(default_factory=lambda: Path.cwd() / "experiment.yaml")

    @model_validator(mode="after")
    def consistent_budgets(self):
        total, batch = self.training.total_steps, self.training.steps_per_update
        if total % batch:
            raise ValueError(
                "training.total_steps must contain whole updates; no implicit overshoot"
            )
        if batch % self.runtime.num_envs:
            raise ValueError(
                "training.steps_per_update must divide evenly across runtime.num_envs"
            )
        if batch % self.algorithm.batch_size:
            raise ValueError(
                "training.steps_per_update must contain whole algorithm.batch_size minibatches"
            )
        if self.runtime.sampling_processes > self.runtime.num_envs:
            raise ValueError(
                "runtime.sampling_processes cannot exceed runtime.num_envs"
            )
        if not self.algorithm.hidden_sizes:
            raise ValueError("algorithm.hidden_sizes cannot be empty")
        if self.validation.enabled and self.validation.seed == self.evaluation.seed:
            raise ValueError("validation.seed and evaluation.seed must be distinct")
        return self

    def origins(self) -> dict[str, str]:
        current = flatten(primitive(self))
        return {
            key: self._origins.get(key, "default")
            if key in self._baseline and value == self._baseline[key]
            else "python"
            for key, value in current.items()
        }


def flatten(data: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def merge(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _remember(config: ExperimentConfig, origin: str) -> ExperimentConfig:
    config._baseline = flatten(primitive(config))
    config._origins = dict.fromkeys(config._baseline, origin)
    return config


def from_legacy(path: str | Path) -> ExperimentConfig:
    """Read a v1 recipe without executing it or requiring optional backends."""
    path = Path(path).resolve()
    data = yaml.load(path.read_text(), Loader=_UniqueLoader)
    training = data.get("schema") == "smartsom.training-run/v1"
    run, _ = read_model(path, TrainingRunSpec if training else RunSpec)
    algorithm_path = (path.parent / run.algorithm).resolve()
    algorithm, _ = read_model(algorithm_path, AlgorithmFile)
    params = primitive(algorithm.algorithm.parameters)
    values = {k: v for k, v in params.items() if k in AlgorithmOptions.model_fields}
    values["source"] = str(algorithm_path)
    if isinstance(algorithm.algorithm, LearningAlgorithm):
        values["extensions"] = primitive(algorithm.algorithm.extensions)
    budget = primitive(run.budget) if run.budget is not None else {}
    config = ExperimentConfig.model_validate_json(
        canonical_json(
            {
                "scenario": str((path.parent / run.scenario).resolve()),
                "seed": run.seed,
                "algorithm": values,
                "training": {
                    "total_steps": budget.get("environment_steps", 4096),
                    "steps_per_update": params.get("n_steps", 256),
                    "max_decisions": budget.get("max_decisions", 1024),
                    "max_ticks": budget.get("max_ticks", 10000),
                },
                "logging": {
                    "observations": (
                        run.recording.observations if run.recording else "hash"
                    ),
                    "debug": bool(run.recording and run.recording.debug),
                    "tensorboard": False,
                },
                "validation": {"enabled": False},
                "checkpointing": {"every_updates": None, "save_best": False},
                "solver": {
                    "time_limit_seconds": budget.get("solver_time_limit_seconds", 60.0)
                },
                "output": {
                    "root": str((path.parent / run.output_root).resolve()),
                    "name": path.stem,
                },
            }
        )
    )
    config._owner = path
    return _remember(config, str(path))


def load_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise ConfigurationError(
            f"unknown preset {name!r}; choose {', '.join(PRESETS)}"
        )
    config = from_legacy(PRESET_ROOT / "configs" / "runs" / f"{PRESETS[name][0]}.yaml")
    config.output.root = str(Path.cwd() / "runs")
    config.output.name = name
    if name.endswith("_micro"):
        config.training.total_steps = 4096
        config.validation.enabled = True
        config.checkpointing.every_updates = 4
        config.checkpointing.save_best = True
        config.logging.tensorboard = True
    return _remember(config, f"preset:{name}")


def load_config(path: str | Path, *, preset: str | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    try:
        data = yaml.load(path.read_text(), Loader=_UniqueLoader)
        if not isinstance(data, dict):
            raise ValueError("expected a mapping")
        if data.get("schema") in {"smartsom.run/v1", "smartsom.training-run/v1"}:
            if preset:
                raise ValueError("legacy recipes cannot also select a preset")
            return from_legacy(path)
        selected = data.pop("preset", None)
        if preset and selected and selected != preset:
            raise ValueError("conflicting preset selections")
        selected = preset or selected
        base = load_preset(selected) if selected else None
        for field in ("scenario",):
            if field in data:
                data[field] = str((path.parent / data[field]).resolve())
        for section, field in (("algorithm", "source"), ("output", "root")):
            if field in data.get(section, {}):
                data[section][field] = str(
                    (path.parent / data[section][field]).resolve()
                )
        overrides = data.get("scenario_overrides", {})
        if "factory" in overrides:
            overrides["factory"] = str((path.parent / overrides["factory"]).resolve())
        for section in (
            "workload",
            "arrivals",
            "processing_time",
            "machine_events",
            "quality",
        ):
            if (
                isinstance(overrides.get(section), dict)
                and "path" in overrides[section]
            ):
                overrides[section]["path"] = str(
                    (path.parent / overrides[section]["path"]).resolve()
                )
        for section in ("validation", "evaluation"):
            if "scenarios" in data.get(section, {}):
                data[section]["scenarios"] = [
                    str((path.parent / p).resolve()) for p in data[section]["scenarios"]
                ]
        merged = merge(primitive(base), data) if base else data
        config = ExperimentConfig.model_validate_json(canonical_json(merged))
        config._owner = path
        _remember(config, str(path))
        if base:
            config._origins = {
                **base._origins,
                **dict.fromkeys(flatten(data), str(path)),
            }
        return config
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def apply_overrides(
    config: ExperimentConfig, overrides: list[tuple[str, Any]]
) -> ExperimentConfig:
    data = primitive(config)
    touched = set()
    for path, value in overrides:
        if any(
            path == old or path.startswith(old + ".") or old.startswith(path + ".")
            for old in touched
        ):
            raise ConfigurationError(f"duplicate/overlapping override: {path}")
        touched.add(path)
        parts = path.split(".")
        node = data
        for part in parts[:-1]:
            if path.startswith("scenario_overrides.") and part not in node:
                node[part] = {}
            if part not in node or not isinstance(node[part], dict):
                raise ConfigurationError(f"unknown configuration field: {path}")
            node = node[part]
        if parts[-1] not in node and not path.startswith("scenario_overrides."):
            raise ConfigurationError(f"unknown configuration field: {path}")
        node[parts[-1]] = value
    try:
        updated = ExperimentConfig.model_validate_json(canonical_json(data))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    updated._owner = config._owner
    _remember(updated, "default")
    updated._origins = config.origins()
    for field in updated._baseline:
        if any(field == p or field.startswith(p + ".") for p in touched):
            updated._origins[field] = "cli"
    return updated


@dataclass(frozen=True)
class PreparedExperiment:
    config_json: str
    origins_json: str
    resolved: Any
    scientific_sha256: str
    validation_json: str | None = None


def training_identity(
    resolved: ResolvedTrainingRun, runtime: RuntimeOptions, validation_json=None
) -> dict:
    identity = {
        "base": semantic_run(resolved.base),
        "algorithm": primitive(resolved.algorithm),
        "budget": primitive(resolved.run.budget),
        "seed": resolved.run.seed,
        "runtime": primitive(runtime),
    }
    if validation_json is not None:
        from smartsom.config.validation import read_validation_inputs

        identity["validation_inputs"] = [
            {
                key: value
                for key, value in item.items()
                if key not in {"source", "source_sha256", "reference"}
            }
            for item in read_validation_inputs(validation_json)
        ]
    return identity


def bind_training_algorithm(
    config: ExperimentConfig, algorithm: AlgorithmFile
) -> AlgorithmFile:
    selected = algorithm.algorithm
    if not isinstance(selected, LearningAlgorithm):
        raise ConfigurationError(
            "train requires a learning preset; use run for rules and solvers"
        )
    params = primitive(selected.parameters)
    values = primitive(config.algorithm)
    for key in params:
        if key in values:
            params[key] = values[key]
    if selected.provider == "rllib.resource_ppo":
        params["learner_reward_scale"] = config.algorithm.learner_reward_scale
    elif config.algorithm.learner_reward_scale != 1.0:
        raise ConfigurationError(
            "learner_reward_scale currently belongs to resource PPO"
        )
    params["n_steps"] = config.training.steps_per_update
    payload = primitive(algorithm)
    payload["algorithm"]["parameters"] = params
    from smartsom.learning.extensions import bind_extensions

    extensions = bind_extensions(config.algorithm.extensions, selected.provider)
    if extensions is None:
        payload["algorithm"].pop("extensions", None)
    else:
        payload["algorithm"]["extensions"] = primitive(extensions)
    return AlgorithmFile.model_validate_json(canonical_json(payload))


def prepare_frozen(
    config: ExperimentConfig, template: PreparedExperiment
) -> PreparedExperiment:
    """Bind a candidate to an already materialized training world without files."""
    if not isinstance(template.resolved, ResolvedTrainingRun):
        raise ConfigurationError("a frozen training template is required")
    original = ExperimentConfig.model_validate_json(template.config_json)
    if (
        digest(
            training_identity(
                template.resolved, original.runtime, template.validation_json
            )
        )
        != template.scientific_sha256
    ):
        raise ConfigurationError("frozen template scientific identity mismatch")
    origins = config.origins()
    config = ExperimentConfig.model_validate_json(canonical_json(config))
    if (
        any(
            getattr(config, name) != getattr(original, name)
            for name in ("seed", "scenario", "scenario_overrides", "validation")
        )
        or config.algorithm.source != original.algorithm.source
    ):
        raise ConfigurationError(
            "frozen candidates must retain the template seed, scenario and provider source"
        )
    algorithm = bind_training_algorithm(config, template.resolved.algorithm)
    run = template.resolved.run.model_copy(
        update={
            "budget": TrainingBudget(
                environment_steps=config.training.total_steps,
                max_decisions=config.training.max_decisions,
                max_ticks=config.training.max_ticks,
            ),
            "recording": RecordingSpec(
                observations=config.logging.observations, debug=config.logging.debug
            ),
            "output_root": str(Path(config.output.root).resolve()),
        }
    )
    resolved = replace(template.resolved, algorithm=algorithm, run=run)
    return PreparedExperiment(
        canonical_json(config),
        canonical_json(origins),
        resolved,
        digest(training_identity(resolved, config.runtime, template.validation_json)),
        template.validation_json,
    )


def prepare(
    config: ExperimentConfig,
    *,
    training: bool = True,
    require_dependencies: bool = False,
) -> PreparedExperiment:
    """Freeze and validate all authoring values before allocating any run directory."""
    origins = config.origins()
    config = ExperimentConfig.model_validate_json(canonical_json(config))
    algorithm_path = Path(config.algorithm.source).resolve()
    algorithm, algorithm_sha = read_model(algorithm_path, AlgorithmFile)
    selected = algorithm.algorithm
    scenario_path = Path(config.scenario).resolve()
    scenario, scenario_sha = read_model(scenario_path, ScenarioFile)
    scenario = ScenarioFile.model_validate_json(
        canonical_json(merge(primitive(scenario), config.scenario_overrides))
    )
    sources = (SourceFile("scenario", scenario_path, scenario_sha),)
    algorithm_source = SourceFile("algorithm", algorithm_path, algorithm_sha)
    recording = RecordingSpec(
        observations=config.logging.observations, debug=config.logging.debug
    )
    if training:
        algorithm = bind_training_algorithm(config, algorithm)
        run = TrainingRunSpec(
            schema="smartsom.training-run/v1",
            scenario=str(scenario_path),
            algorithm=str(algorithm_path),
            seed=config.seed,
            output_root=str(Path(config.output.root).resolve()),
            budget=TrainingBudget(
                environment_steps=config.training.total_steps,
                max_decisions=config.training.max_decisions,
                max_ticks=config.training.max_ticks,
            ),
            recording=recording,
        )
        resolved = resolve_training_spec(
            run,
            scenario_path,
            algorithm,
            sources=sources,
            algorithm_source=algorithm_source,
            scenario_override=scenario,
            require_dependencies=require_dependencies,
        )
        validation_json = None
        if config.validation.enabled and config.validation.scenarios:
            from smartsom.config.validation import freeze_validation_cases

            validation_json = freeze_validation_cases(resolved, config.validation)
        scientific = training_identity(resolved, config.runtime, validation_json)
    else:
        validation_json = None
        budget = (
            RunBudget(solver_time_limit_seconds=config.solver.time_limit_seconds)
            if isinstance(selected, CPSatAlgorithm)
            else EpisodeBudget(
                max_decisions=config.training.max_decisions,
                max_ticks=config.training.max_ticks,
            )
            if isinstance(selected, LearningAlgorithm)
            else None
        )
        run = RunSpec(
            schema="smartsom.run/v1",
            scenario=str(scenario_path),
            algorithm=str(algorithm_path),
            seed=config.seed,
            output_root=str(Path(config.output.root).resolve()),
            recording=recording,
            budget=budget,
        )
        resolved = _resolve_run_spec(
            run,
            scenario_path,
            list((*sources, algorithm_source)),
            algorithm_override=algorithm,
            scenario_override=scenario,
        )
        scientific = semantic_run(resolved)
    return PreparedExperiment(
        canonical_json(config),
        canonical_json(origins),
        resolved,
        digest(scientific),
        validation_json,
    )


def preview(config: ExperimentConfig) -> dict:
    algorithm, _ = read_model(Path(config.algorithm.source), AlgorithmFile)
    prepared = prepare(
        config, training=isinstance(algorithm.algorithm, LearningAlgorithm)
    )
    resolved = prepared.resolved
    base = resolved.base if isinstance(resolved, ResolvedTrainingRun) else resolved
    return {
        "config": primitive(config),
        "origins": config.origins(),
        "scientific_sha256": prepared.scientific_sha256,
        "provider": algorithm.algorithm.provider,
        "inputs": {
            "machines": len(base.factory.machines),
            "agvs": len(base.factory.transport.agvs) if base.factory.transport else 0,
            "jobs": sum(len(o.jobs) for o in base.workload.orders),
            "operations": len(base.workload.operations),
            "workload_sha256": base.workload_sha256,
        },
        "scenario": primitive(base.scenario),
        "sources": primitive(base.sources),
        "sampling": {
            "total_steps": config.training.total_steps,
            "updates": config.training.total_steps // config.training.steps_per_update,
            "steps_per_env_per_update": config.training.steps_per_update
            // config.runtime.num_envs,
        },
        "evaluation": {
            "replications": config.evaluation.replications,
            "algorithms": 1 + len(config.evaluation.baselines),
            "scenarios": max(1, len(config.evaluation.scenarios)),
            "runs": config.evaluation.replications
            * max(1, len(config.evaluation.scenarios))
            * (1 + len(config.evaluation.baselines)),
        },
        "validation": {
            "enabled": config.validation.enabled,
            "scenarios": max(1, len(config.validation.scenarios)),
            "inputs_per_validation": config.validation.replications
            * max(1, len(config.validation.scenarios))
            if config.validation.enabled
            else 0,
            "frozen_external_inputs": prepared.validation_json is not None,
        },
        "platform_qualification": "macOS verification required; Linux/CUDA not yet verified",
    }
