"""Resolve and freeze classified composable experiment files without frameworks."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, PrivateAttr, model_validator

from smartsom._filesystem import native_path
from smartsom.config.codec import ConfigurationError, canonical_json, digest, primitive
from smartsom.config.compositions import CompositionFile
from smartsom.config.experiment import (
    CheckpointOptions,
    EditableModel,
    EvaluationOptions,
    LoggingOptions,
    OutputOptions,
    RuntimeOptions,
    ValidationOptions,
)
from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionModel,
    ExtensionSpec,
    RewardSpec,
)
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.policies import PolicyFile, QNetwork
from smartsom.config.production import (
    ScenarioFile,
    WorkloadFile,
    materialize,
    named_seed,
    read_file,
)
from smartsom.config.travel_time import freeze_transport
from smartsom.domain.production_decisions import ACTION_CONTRACT, OBSERVATION_CONTRACT
from smartsom.domain.travel_time import physical_contract, validate_model_contract
from smartsom.experiments.checkpoint_retention import leased
from smartsom.learning.production_contract import factory_identity


class PPOParameters(ExtensionModel):
    learning_rate: Annotated[float, Field(gt=0)] = 0.0003
    gae_lambda: Annotated[float, Field(gt=0, le=1)] = 0.95
    clip_range: Annotated[float, Field(gt=0, lt=1)] = 0.2
    entropy_coefficient: Annotated[float, Field(ge=0)] = 0.01
    batch_size: Annotated[int, Field(gt=1)] = 64
    n_epochs: Annotated[int, Field(gt=0)] = 4
    value_coefficient: Annotated[float, Field(ge=0)] = 0.5
    max_grad_norm: Annotated[float, Field(gt=0)] = 0.5


class DQNParameters(ExtensionModel):
    learning_rate: Annotated[float, Field(gt=0)] = 0.0005
    replay_capacity: Annotated[int, Field(gt=0)] = 50000
    batch_size: Annotated[int, Field(gt=1)] = 32
    warmup_ticks: Annotated[int, Field(ge=0)] = 1000
    target_update_ticks: Annotated[int, Field(gt=0)] = 500
    epsilon_start: Annotated[float, Field(ge=0, le=1)] = 1.0
    epsilon_end: Annotated[float, Field(ge=0, le=1)] = 0.05
    epsilon_ticks: Annotated[int, Field(gt=0)] = 10000
    train_every_ticks: Annotated[int, Field(gt=0)] = 1
    gradient_steps: Annotated[int, Field(gt=0)] = 1
    double_q: Literal[True] = True
    dueling: Literal[False] = False
    n_step: Literal[1] = 1
    num_atoms: Literal[1] = 1


class TrainingOptionsV3(EditableModel):
    mode: Literal["resource", "central"] = "resource"
    backend: Literal["rllib", "sb3"] = "rllib"
    algorithm: Literal["ppo", "dqn"] = "ppo"
    parameters: str | None = None
    gamma: Annotated[float, Field(gt=0, le=1)] = 0.99
    reward: RewardSpec = Field(default_factory=RewardSpec)
    groups: tuple[str, ...]
    record_initial: bool = False
    total_ticks: Annotated[int, Field(gt=0)] = 4096
    ticks_per_update: Annotated[int, Field(gt=0)] = 256
    max_ticks: Annotated[int, Field(gt=0)] = 10000

    @model_validator(mode="after")
    def supported_matrix(self):
        if self.backend == "sb3" and (
            self.mode != "central" or self.algorithm != "ppo"
        ):
            raise ValueError("SB3 supports only central MaskablePPO")
        if self.mode == "central" and self.algorithm != "ppo":
            raise ValueError("central DQN is outside the v3 contract")
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError("training.groups must be nonempty and unique")
        return self


class EvaluationOptionsV3(EvaluationOptions):
    checkpoint: str = "last"


class ComposableExperimentConfig(EditableModel):
    schema_id: Literal["smartsom.experiment-config/v3"] = Field(alias="schema")
    interface_contract: Literal["smartsom.configurable-experiment/v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scenario: str
    composition: str
    scenario_overrides: dict = Field(default_factory=dict)
    seed: Annotated[int, Field(ge=0, lt=2**64)] = 101
    training: TrainingOptionsV3 | None = None
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
    validation: ValidationOptions = Field(default_factory=ValidationOptions)
    evaluation: EvaluationOptionsV3 = Field(default_factory=EvaluationOptionsV3)
    checkpointing: CheckpointOptions = Field(default_factory=CheckpointOptions)
    logging: LoggingOptions = Field(default_factory=LoggingOptions)
    output: OutputOptions = Field(default_factory=OutputOptions)
    _owner: Path = PrivateAttr(default_factory=lambda: Path.cwd() / "experiment.yaml")

    @model_validator(mode="after")
    def independent_inputs(self):
        if (
            self.training
            and self.validation.enabled
            and self.validation.seed == self.evaluation.seed
        ):
            raise ValueError("validation and held-out evaluation seeds must differ")
        return self

    def origins(self):
        return {"root": str(self._owner)}


class ExecutionConfig(ComposableExperimentConfig):
    """Detached execution recipe: author inputs are compiled, never reopened."""

    schema_id: Literal[
        "smartsom.execution-config/v1", "smartsom.execution-config/v2"
    ] = Field(alias="schema")
    scenario: None = None
    composition: None = None


@dataclass(frozen=True)
class CompositionInputs:
    """Typed in-memory boundary shared with the four-file authoring compiler."""

    scenario: object
    settings: ScenarioFile
    workload: WorkloadFile
    composition: CompositionFile
    matching: dict
    policies: dict[str, PolicyFile]
    policy_origins: dict[str, Path]
    parameters: dict
    validation: list
    evaluation: list
    origins: dict
    training_metadata: dict


@dataclass(frozen=True)
class PreparedComposition:
    config_json: str
    scenario_json: str
    composition_json: str
    policies_json: str
    parameters_json: str
    validation_json: str
    evaluation_json: str
    origins_json: str
    scientific_sha256: str
    training_inputs_json: str = "{}"

    @property
    def config(self):
        cls = (
            ExecutionConfig
            if json.loads(self.config_json).get("schema")
            in {"smartsom.execution-config/v1", "smartsom.execution-config/v2"}
            else ComposableExperimentConfig
        )
        return cls.model_validate_json(self.config_json)

    @property
    def scenario(self):
        from smartsom.config.production import scenario_from_snapshot

        return scenario_from_snapshot(json.loads(self.scenario_json))


def read_document(path):
    import yaml

    from smartsom.config.experiment import _UniqueLoader

    path = Path(path).expanduser().resolve()
    try:
        result = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
        if not isinstance(result, dict):
            raise ValueError("expected a YAML mapping")
        return result
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def load_v3(path, data=None):
    path = Path(path).resolve()
    data = copy.deepcopy(data if data is not None else read_document(path))
    for key in ("scenario", "composition"):
        if key in data:
            data[key] = str((path.parent / data[key]).resolve())
    if data.get("training", {}).get("parameters"):
        data["training"]["parameters"] = str(
            (path.parent / data["training"]["parameters"]).resolve()
        )
    if "root" in data.get("output", {}):
        data["output"]["root"] = str((path.parent / data["output"]["root"]).resolve())
    else:
        data.setdefault("output", {})["root"] = str(path.parent / "runs")
    for section in ("validation", "evaluation"):
        if data.get(section, {}).get("scenarios"):
            data[section]["scenarios"] = [
                str((path.parent / p).resolve()) for p in data[section]["scenarios"]
            ]
    try:
        config = ComposableExperimentConfig.model_validate_json(canonical_json(data))
    except ValueError as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    config._owner = path
    return config


def world(path, seed, overrides=None):
    from smartsom.config.experiment import merge

    path = Path(path)
    data = merge(read_document(path), overrides or {})
    settings = freeze_transport(
        ScenarioFile.model_validate_json(canonical_json(data)), path
    )
    factory_file, _ = load_factory_design_file(path.parent / settings.factory)
    workload = read_file(path.parent / settings.workload, WorkloadFile)
    return materialize(factory_file.factory, workload, settings, seed)


@leased
def model_location(selector):
    """Metadata-only resolution. ZIP is not extracted by show-config."""
    source = Path(selector.source).resolve()
    if native_path(source).is_file():
        import zipfile

        if selector.checkpoint is not None:
            raise ValueError("a fixed component ZIP cannot select another checkpoint")
        with zipfile.ZipFile(native_path(source)) as archive:
            names = [n for n in archive.namelist() if n == "model.json"]
            if len(names) != 1:
                raise ValueError("ZIP is not a v3 component/central model package")
            if set(archive.namelist()) - {"model.json", "weights.pt", "encoder.json"}:
                raise ValueError("unexpected component archive member")
            metadata = json.loads(archive.read("model.json"))
            validate_model_contract(metadata)
            if hashlib.sha256(archive.read("weights.pt")).hexdigest() != metadata.get(
                "weights_sha256"
            ):
                raise ValueError("model weights hash mismatch")
            if hashlib.sha256(archive.read("encoder.json")).hexdigest() != metadata.get(
                "encoder_sha256"
            ):
                raise ValueError("model encoder hash mismatch")
        return {
            "source": str(source),
            "metadata": metadata,
            "package_sha256": hashlib.sha256(
                native_path(source).read_bytes()
            ).hexdigest(),
        }
    selected = source
    if (native_path(source / "run.json")).is_file():
        checkpoint = selector.checkpoint or "last"
        if checkpoint in ("best", "last"):
            from smartsom.experiments.checkpoint_retention import resolve

            retained = resolve(source, checkpoint)
            alias = source / "checkpoints" / f"{checkpoint}.json"
            if retained is None and not native_path(alias).is_file():
                raise ValueError(f"{checkpoint} checkpoint does not exist")
            if retained is not None:
                selected = retained
            else:
                pointer = json.loads(native_path(alias).read_text(encoding="utf-8"))
                selected = source / "checkpoints" / pointer["checkpoint"]
        elif checkpoint.startswith("update-") and checkpoint[7:].isdigit():
            selected = source / "checkpoints" / checkpoint
        else:
            raise ValueError("checkpoint must be best, last or update-NNNNNN")
    elif selector.checkpoint is not None:
        raise ValueError("explicit update/component paths are already fixed")
    if not (native_path(selected / "model.json")).is_file():
        if selector.group is None:
            central = selected / "controllers/central"
            if (native_path(central / "model.json")).is_file():
                selected = central
            else:
                raise ValueError("select the source strategy group explicitly")
        else:
            selected = selected / "groups" / selector.group
    metadata_path = selected / "model.json"
    if not native_path(metadata_path).is_file():
        raise ValueError("no v3 component model; legacy weights require retraining")
    metadata = json.loads(native_path(metadata_path).read_text(encoding="utf-8"))
    validate_model_contract(metadata)
    weights = selected / metadata.get("weights_file", "weights.pt")
    if not native_path(weights).is_file():
        raise ValueError("model weights missing")
    actual_hash = hashlib.sha256(native_path(weights).read_bytes()).hexdigest()
    if metadata.get("weights_sha256") != actual_hash:
        raise ValueError("model weights hash mismatch")
    encoder = selected / "encoder.json"
    if not native_path(encoder).is_file() or hashlib.sha256(
        native_path(encoder).read_bytes()
    ).hexdigest() != metadata.get("encoder_sha256"):
        raise ValueError("model encoder hash mismatch")
    return {
        "source": str(selected),
        "metadata": metadata,
        "weights_sha256": actual_hash,
    }


def prepare_detached(config, *, inputs, training=None, require_dependencies=False):
    """Prepare already-validated, detached inputs without opening author files."""
    if inputs is None:
        raise TypeError("detached preparation requires compiled inputs")
    return _prepare_composition(
        config,
        training=training,
        require_dependencies=require_dependencies,
        inputs=inputs,
    )


def prepare_v3(config, *, training=None, require_dependencies=False, inputs=None):
    """Historical v3 author entry; new authors use detached preparation."""
    return _prepare_composition(
        config,
        training=training,
        require_dependencies=require_dependencies,
        inputs=inputs,
    )


def _prepare_composition(
    config, *, training=None, require_dependencies=False, inputs=None
):
    if training is None:
        training = config.training is not None
    if training and config.training is None:
        raise ConfigurationError("train requires training settings")
    if (
        training
        and config.evaluation.no_eligible_best is not None
        and (not config.validation.enabled or not config.checkpointing.save_best)
    ):
        raise ConfigurationError("no_eligible_best requires validation and save_best")
    try:
        if training and config.validation.enabled:
            from smartsom.experiments.training_controls import ValidationControls

            ValidationControls(
                **{
                    k: getattr(config.validation, k)
                    for k in type(config.validation).model_fields
                    if k != "enabled"
                }
            )
        if inputs is None:
            scenario = world(config.scenario, config.seed, config.scenario_overrides)
            scenario_path = Path(config.scenario)
            from smartsom.config.experiment import merge

            settings = ScenarioFile.model_validate_json(
                canonical_json(
                    merge(read_document(scenario_path), config.scenario_overrides)
                )
            )
            settings = freeze_transport(settings, scenario_path)
            workload = read_file(scenario_path.parent / settings.workload, WorkloadFile)
            path = Path(config.composition)
            composition = CompositionFile.model_validate_json(
                canonical_json(read_document(path))
            )
            matching_path = (path.parent / composition.pickup_matching).resolve()
            matching = read_document(matching_path)
        else:
            scenario, settings, workload = (
                inputs.scenario,
                inputs.settings,
                inputs.workload,
            )
            composition, matching = inputs.composition, inputs.matching
        training_inputs = {
            "settings": primitive(settings),
            "workload": primitive(workload),
        }
        if inputs is not None:
            training_inputs.update(inputs.training_metadata)
        if matching.get("schema") not in {
            "smartsom.pickup-rule/v1",
            "smartsom.pickup-matching/v2",
        } or matching.get("name") not in (
            "first_arrival",
            "global_optimal",
            "priority_greedy",
        ):
            raise ValueError("invalid pickup matching configuration")
        matching = dict(matching, name="first_arrival")
        central = composition.controller is not None
        if training and (config.training.mode == "central") != central:
            raise ValueError("training mode does not match composition")
        groups = {"central": composition.controller} if central else composition.groups
        train_groups = set(config.training.groups) if training else set()
        if train_groups - groups.keys():
            raise ValueError("unknown training group")
        declarations = {}
        origins = (
            {str(config._owner): digest(primitive(config))}
            if inputs is None
            else dict(inputs.origins)
        )
        for group, ref in groups.items():
            if not group or "/" in group or "\\" in group or group in (".", ".."):
                raise ValueError("group must be a safe semantic identifier")
            if inputs is None:
                policy_path = (path.parent / ref.policy).resolve()
                policy = PolicyFile.model_validate_json(
                    canonical_json(read_document(policy_path))
                )
                origins[str(policy_path)] = digest(primitive(policy))
            else:
                policy_path = inputs.policy_origins[group]
                policy = inputs.policies[group]
            declaration = primitive(policy)
            impl = policy.implementation
            if impl.kind == "new_model":
                if group not in train_groups:
                    raise ValueError(
                        f"fresh group {group} has no executable weights; select it for training"
                    )
                from smartsom.learning.extensions import bind_extensions, registration

                provider = (
                    "sb3.maskable_ppo"
                    if config.training.backend == "sb3"
                    else (
                        "rllib.ppo"
                        if central
                        else "rllib.resource_" + config.training.algorithm
                    )
                )
                from smartsom.config.policy_contracts import (
                    PHYSICAL_OBSERVATION,
                    pin_contract,
                    validate_factory,
                )

                observation, network_implementation = pin_contract(
                    impl.extensions.observation, impl.extensions.network_implementation
                )
                validate_factory(scenario.factory, observation)
                if observation is None or observation.name != PHYSICAL_OBSERVATION:
                    observation = bind_extensions(
                        ExtensionSpec(observation=observation), provider
                    ).observation
                if network_implementation is not None:
                    declaration["implementation"]["extensions"][
                        "network_implementation"
                    ] = primitive(network_implementation)
                declaration["implementation"]["extensions"]["observation"] = primitive(
                    observation
                )
                network = impl.extensions.network or (
                    QNetwork()
                    if config.training.algorithm == "dqn"
                    else ActorCriticSpec()
                )
                declaration["implementation"]["extensions"]["network"] = primitive(
                    network
                )
                if network is not None:
                    branch_names = (
                        ("q",) if isinstance(network, QNetwork) else ("actor", "critic")
                    )
                    for name in branch_names:
                        ref = getattr(network, name).encoder
                        entry = registration("torch_encoder", ref, provider)
                        declaration["implementation"]["extensions"]["network"][name][
                            "encoder"
                        ] = primitive(
                            ref.model_copy(update={"code_sha256": entry.code_sha256})
                        )
                if network is not None and isinstance(network, QNetwork) != (
                    config.training.algorithm == "dqn"
                ):
                    raise ValueError("network branches do not match PPO/DQN")
            elif impl.kind == "rule":
                if group in train_groups:
                    raise ValueError("rules cannot be trained")
            else:
                selector = impl.model.model_copy(
                    update={
                        "source": str(
                            (policy_path.parent / impl.model.source).resolve()
                        )
                    }
                )
                model = model_location(selector)
                metadata = model["metadata"]
                if (
                    metadata.get("action_contract") != ACTION_CONTRACT
                    or metadata.get("observation_contract") != OBSERVATION_CONTRACT
                ):
                    raise ValueError("incompatible model contract; retraining required")
                if metadata.get("factory_identity") != factory_identity(
                    scenario.factory
                ):
                    raise ValueError("model factory structure is incompatible")
                if metadata.get(
                    "physical_contract",
                    {"transport": "grid/v3", "processing_rounding": "half_up"},
                ) != physical_contract(scenario):
                    raise ValueError(
                        "model transport/processing contract is incompatible; retraining required"
                    )
                if metadata.get("role") != policy.role:
                    raise ValueError("model role mismatch")
                if group in train_groups and (
                    metadata.get("algorithm") != config.training.algorithm
                    or metadata.get("backend") != config.training.backend
                ):
                    raise ValueError(
                        "cross-algorithm/backend weight initialization is unsupported"
                    )
                declaration["resolved_model"] = model
            if central and policy.role != "central":
                raise ValueError("central composition requires a whole controller")
            declarations[group] = declaration
        if not central:
            machine_ids = {m.machine_id for m in scenario.factory.machines}
            vehicle_ids = {a.agv_id for a in scenario.factory.agvs}
            sources = {
                b.buffer_id
                for b in scenario.factory.buffers
                if b.role not in ("machine_pre", "system_output")
            } | {s.inspection_station_id for s in scenario.factory.inspection_stations}
            post_machines = {
                b.machine_id
                for b in scenario.factory.buffers
                if b.role == "machine_post"
            }
            sources |= machine_ids - post_machines
            known = {
                "machine": machine_ids,
                "buffer": sources,
                "dispatcher": vehicle_ids,
                "mover": vehicle_ids,
            }
            for role, binding in composition.bindings.items():
                if set(binding.overrides) - known[role]:
                    raise ValueError(f"unknown {role} override owner")
                for group in {binding.default, *binding.overrides.values()}:
                    if declarations[group]["role"] != role:
                        raise ValueError("strategy group role mismatch")
        if not central:
            binding = composition.bindings["mover"]
            for group in {binding.default, *binding.overrides.values()}:
                impl = declarations[group]["implementation"]
                automatic = (
                    impl["kind"] == "rule" and impl["name"] == "automatic_travel"
                )
                if automatic != bool(scenario.transport_matrix):
                    raise ValueError(
                        "matrix transport requires automatic_travel Mover; grid transport requires a movement policy"
                    )
        parameters = {}
        if training:
            data = (
                {
                    "schema": "smartsom.algorithm-parameters/v1",
                    "parameters": inputs.parameters,
                }
                if inputs is not None
                else (
                    read_document(config.training.parameters)
                    if config.training.parameters
                    else {
                        "schema": "smartsom.algorithm-parameters/v1",
                        "parameters": {},
                    }
                )
            )
            if data.get("schema") != "smartsom.algorithm-parameters/v1":
                raise ValueError("invalid algorithm parameter schema")
            cls = PPOParameters if config.training.algorithm == "ppo" else DQNParameters
            parameters = primitive(
                cls.model_validate_json(canonical_json(data.get("parameters", {})))
            )

        def cases(options, label):
            result = []
            for case_index, source in enumerate(
                options.scenarios or (config.scenario,)
            ):
                for replication in range(options.replications):
                    seed = named_seed(
                        options.seed, f"{label}:{case_index}:{replication}"
                    )
                    case = world(
                        source,
                        seed,
                        config.scenario_overrides
                        if source == config.scenario
                        else None,
                    )
                    from smartsom.config.experiment import merge

                    case_path = Path(source)
                    case_settings = ScenarioFile.model_validate_json(
                        canonical_json(
                            merge(
                                read_document(case_path),
                                config.scenario_overrides
                                if source == config.scenario
                                else {},
                            )
                        )
                    )
                    case_settings = freeze_transport(case_settings, case_path)
                    case_workload = read_file(
                        case_path.parent / case_settings.workload, WorkloadFile
                    )
                    result.append(
                        {
                            "case": str(case_index),
                            "replication": replication,
                            "seed": seed,
                            "scenario": primitive(case),
                            "recipe": {
                                "settings": primitive(case_settings),
                                "workload": primitive(case_workload),
                            },
                        }
                    )
            return result

        validation = (
            inputs.validation
            if inputs is not None
            else (
                cases(config.validation, "validation")
                if training and config.validation.enabled
                else []
            )
        )
        evaluation = (
            inputs.evaluation
            if inputs is not None
            else cases(config.evaluation, "evaluation")
        )
        from smartsom.config.production import scenario_from_snapshot

        if any(
            factory_identity(scenario_from_snapshot(case["scenario"]).factory)
            != factory_identity(scenario.factory)
            for case in validation + evaluation
        ):
            raise ValueError("validation/evaluation factory structure is incompatible")
        if any(
            physical_contract(scenario_from_snapshot(case["scenario"]))
            != physical_contract(scenario)
            for case in validation + evaluation
        ):
            raise ValueError("validation/evaluation physical contract differs")
        if central:
            impl = declarations["central"]["implementation"]
            max_jobs = (
                impl["projection"]["max_jobs"]
                if impl["kind"] == "new_model"
                else declarations["central"]["resolved_model"]["metadata"][
                    "projection"
                ]["max_jobs"]
            )
            if any(
                len(s["demands"]) > max_jobs
                for s in [
                    primitive(scenario),
                    *[c["scenario"] for c in validation + evaluation],
                ]
            ):
                raise ValueError(
                    "central projection capacity exceeded before run allocation"
                )
        resolved = primitive(composition)
        resolved["matching"] = matching
        frozen_config = primitive(config)
        if (
            config.validation.updates is not None
            or config.evaluation.no_eligible_best is not None
            or config.checkpointing.retention is not None
            or any(
                declaration["implementation"]
                .get("extensions", {})
                .get("network_implementation")
                or declaration.get("resolved_model", {})
                .get("metadata", {})
                .get("network_implementation")
                for declaration in declarations.values()
            )
        ):
            frozen_config["interface_contract"] = "smartsom.configurable-experiment/v1"
        if training:
            from smartsom.learning.extensions import bind_extensions

            provider = (
                "sb3.maskable_ppo"
                if config.training.backend == "sb3"
                else (
                    "rllib.ppo"
                    if central
                    else "rllib.resource_" + config.training.algorithm
                )
            )
            frozen_config["training"]["reward"] = primitive(
                bind_extensions(
                    ExtensionSpec(reward=config.training.reward), provider
                ).reward
            )
        # Titles are presentation only. Keep them in the archived configuration,
        # but exclude these new display fields from scientific comparisons.
        scientific_config = {
            **frozen_config,
            "logging": {
                key: value
                for key, value in frozen_config["logging"].items()
                if key not in {"title", "task_title"}
            },
        }
        payload = {
            "config": scientific_config,
            "scenario": primitive(scenario),
            "composition": resolved,
            "policies": declarations,
            "parameters": parameters,
            "validation": validation,
            "evaluation": evaluation,
            "training_inputs": training_inputs,
        }
        if require_dependencies:
            import importlib.util

            if config.runtime.device == "cuda":
                import torch

                if not torch.cuda.is_available():
                    raise ValueError(
                        "runtime.device=cuda requires available CUDA before allocation"
                    )

            if any(
                d["implementation"]["kind"] == "model" for d in declarations.values()
            ):
                if importlib.util.find_spec("torch") is None:
                    raise ValueError(
                        "component inference requires the locked Torch extra"
                    )
                import torch

                from smartsom.learning.policy_factory import (
                    encoder as make_encoder,
                )
                from smartsom.learning.policy_factory import (
                    network_class,
                    validate_metadata,
                )
                from smartsom.learning.production_inference import read_package

                for declaration in declarations.values():
                    model = declaration.get("resolved_model")
                    if model is None:
                        continue
                    metadata, weights, state = read_package(
                        model["source"], metadata_validator=validate_metadata
                    )
                    encoder = make_encoder(scenario.factory, metadata, training=False)
                    encoder.load_state_dict(state)
                    if (
                        metadata.get("candidate_width") != encoder.candidate_width
                        or metadata.get("context_size") != encoder.context_size
                    ):
                        raise ValueError(
                            "component encoding dimensions are incompatible"
                        )
                    with torch.random.fork_rng():
                        network = network_class(metadata.get("network_implementation"))(
                            encoder.context_size,
                            metadata["network"],
                            metadata["provider"],
                            metadata["algorithm"],
                            central=metadata["role"] == "central",
                            central_private_end=metadata["central_private_end"],
                            candidate_width=encoder.candidate_width,
                        )
                        network.load_state_dict(weights, strict=True)
            if training:
                module = "ray" if config.training.backend == "rllib" else "sb3_contrib"
                if importlib.util.find_spec(module) is None:
                    raise ValueError(
                        f"{config.training.backend} requires its locked learning extra"
                    )
        prepared = PreparedComposition(
            canonical_json(frozen_config),
            canonical_json(scenario),
            canonical_json(resolved),
            canonical_json(declarations),
            canonical_json(parameters),
            canonical_json(validation),
            canonical_json(evaluation),
            canonical_json(origins),
            digest(payload),
            canonical_json(training_inputs),
        )
        if frozen_config.get("interface_contract"):
            from dataclasses import replace

            from smartsom.config.scientific_contract import identity

            prepared = replace(prepared, scientific_sha256=identity(prepared))
        return prepared
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc


def preview_v3(config):
    prepared = prepare_v3(config)
    composition = json.loads(prepared.composition_json)
    declarations = json.loads(prepared.policies_json)
    bindings = primitive(composition.get("bindings", {}))
    details = {}
    for group, policy in declarations.items():
        metadata = policy.get("resolved_model", {}).get("metadata", {})
        details[group] = {
            "role": policy["role"],
            "kind": policy["implementation"]["kind"],
            "training": bool(config.training and group in config.training.groups),
            "source": policy.get("resolved_model", {}).get("source"),
            "model_identity": metadata,
        }
    return {
        "schema": config.schema_id,
        "provider": "composable."
        + (
            f"{config.training.backend}.{config.training.algorithm}"
            if config.training
            else "evaluation"
        ),
        "bindings": bindings,
        "groups": details,
        "compatible": True,
        "matching": composition["matching"],
        "training": primitive(config.training),
        "seeds": {
            "training": config.seed,
            "validation": config.validation.seed,
            "evaluation": config.evaluation.seed,
        },
        "inputs": {
            "workload_sha256": digest(json.loads(prepared.scenario_json)["demands"])
        },
        "scientific_sha256": prepared.scientific_sha256,
    }


def apply_overrides_v3(config, overrides):
    data, touched = primitive(config), set()
    for key, value in overrides:
        if any(
            key == old or key.startswith(old + ".") or old.startswith(key + ".")
            for old in touched
        ):
            raise ConfigurationError("duplicate/overlapping override")
        touched.add(key)
        parts, node = key.split("."), data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigurationError(f"unknown v3 configuration field: {key}")
            node = node[part]
        node[parts[-1]] = value
    try:
        result = ComposableExperimentConfig.model_validate_json(canonical_json(data))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    result._owner = config._owner
    return result
