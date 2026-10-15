"""Opt-in detached production worlds and directly seeded evaluation cases."""

from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.codec import ConfigurationError, canonical_json, digest, primitive
from smartsom.config.experiment import EditableModel, Seed
from smartsom.config.production import (
    ScenarioFile,
    WorkloadFile,
    scenario_from_snapshot,
)
from smartsom.config.workload_v3 import ArrivalSlots, DueFunction, WorkloadV3

SCHEMA = "smartsom.frozen-experiment-inputs/v1"


def case_streams(case, *, environment_seed):
    """Expose actual streams for new cases; historical result fields stay intact."""
    contract = case.get("recipe", {}).get("frozen_input_contract")
    if contract is None:
        return {}
    fields = {
        "seed_semantics": "data_seed",
        "data_seed": contract["data_seed"],
        "environment_seed": environment_seed,
        "policy_root": contract["policy_root"],
    }
    if (
        case["seed"] != fields["data_seed"]
        or environment_seed != contract["environment_seed"]
        or case["case"] != contract["case"]
        or any(case.get(key) != value for key, value in fields.items())
    ):
        raise ValueError("frozen case stream identity is inconsistent")
    return fields


class FrozenWorldFile(EditableModel):
    schema_id: Literal["smartsom.frozen-production-world/v1"] = Field(alias="schema")
    scenario: dict
    provenance: dict = Field(default_factory=dict)


class FrozenWorldTiming(EditableModel):
    arrivals: ArrivalSlots = Field(default_factory=ArrivalSlots)
    due: DueFunction = Field(default_factory=DueFunction)

    @model_validator(mode="after")
    def explicit_timing(self):
        if not self.arrivals.continuous and self.due.nominal_work_factor is None:
            raise ValueError(
                "frozen world timing requires continuous arrivals or nominal_work_factor"
            )
        return self


class FrozenTrainingInput(EditableModel):
    world: Annotated[str, Field(min_length=1)]
    data_seed: Seed
    environment_seed: Seed
    timing: FrozenWorldTiming | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class FrozenCase(FrozenTrainingInput):
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
    policy_seed: Seed


class FrozenExperimentInputs(EditableModel):
    schema_id: Literal["smartsom.frozen-experiment-inputs/v1"] = Field(alias="schema")
    training: FrozenTrainingInput | None = None
    validation: tuple[FrozenCase, ...] = ()
    evaluation: Annotated[tuple[FrozenCase, ...], Field(min_length=1)]
    timing: FrozenWorldTiming | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def unique_cases(self):
        for name in ("validation", "evaluation"):
            ids = [case.id for case in getattr(self, name)]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate frozen {name} case identity")
        return self


def resolve_world(
    reference,
    *,
    owner,
    factory,
    factory_path,
    split,
    read_source,
    source_hashes,
    timing=None,
):
    """Read/hash one snapshot; preserve every physical field without resampling."""
    path = (owner.parent / reference.world).resolve()
    raw, sha256 = read_source(path)
    source_hashes[path] = sha256
    try:
        document = FrozenWorldFile.model_validate_json(canonical_json(raw))
        scenario = scenario_from_snapshot(document.scenario)
        from smartsom.domain.production import validate_production_scenario

        validate_production_scenario(scenario)
    except ValueError as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    if digest(primitive(scenario.factory)) != digest(primitive(factory)):
        raise ConfigurationError(
            "frozen world factory differs from the selected factory"
        )
    if scenario.seed != reference.environment_seed:
        raise ConfigurationError("frozen world seed differs from environment_seed")
    timing = reference.timing or timing
    timing_evidence = None
    if timing is not None:
        from smartsom.workloads.workload_v3 import apply_workload_timing

        original_demands = scenario.demands
        config = WorkloadV3(
            schema="smartsom.workload/v3",
            demands=original_demands,
            mode=scenario.mode,
            arrivals=timing.arrivals,
            due=timing.due,
        )
        try:
            demands, timing_evidence = apply_workload_timing(
                config, original_demands, factory, reference.data_seed, split
            )
            scenario = replace(scenario, demands=demands)
            validate_production_scenario(scenario)
        except ValueError as exc:
            raise ConfigurationError(f"{path}: {exc}") from exc
        timing_evidence["original_demands_sha256"] = digest(original_demands)
    transport = {"mode": "grid"}
    if scenario.transport_matrix is not None:
        overrides = {}
        for source, target, ticks in scenario.transport_matrix.times:
            overrides.setdefault(source, {})[target] = ticks
        transport = {
            "mode": "travel_time_matrix",
            "resolved": {
                "schema": "smartsom.travel-time-matrix/v1",
                "source": "manual",
                "overrides": overrides,
            },
        }
    settings = ScenarioFile.model_validate_json(
        canonical_json(
            {
                "schema": "smartsom.frozen-scenario/v1",
                "factory": str(factory_path),
                "workload": str(path),
                **{
                    name: primitive(getattr(scenario, name))
                    for name in (
                        "mode",
                        "tick_limit",
                        "outages",
                        "processing_low",
                        "processing_high",
                        "reward_time_scale",
                        "processing_samples",
                        "quality_samples",
                        "quality_probability_visibility",
                        "processing_rounding",
                    )
                },
                "processing_low": float(scenario.processing_low),
                "processing_high": float(scenario.processing_high),
                "transport": transport,
            }
        )
    )
    workload = WorkloadFile.model_validate_json(
        canonical_json(
            {
                "schema": "smartsom.frozen-workload/v1",
                "demands": primitive(scenario.demands),
            }
        )
    )
    provenance = {
        "kind": "frozen_world",
        "data_seed": reference.data_seed,
        "split": split,
        "source": document.provenance,
        **(
            {
                "timing": timing_evidence,
                "source_scope": "original frozen world before timing overlay",
            }
            if timing_evidence is not None
            else {}
        ),
    }
    contract = {
        "schema": SCHEMA,
        "world_sha256": sha256,
        "data_seed": reference.data_seed,
        "environment_seed": reference.environment_seed,
        "provenance": document.provenance,
        **(
            {
                "timing": timing_evidence,
                "materialized_world_sha256": digest(scenario),
            }
            if timing_evidence is not None
            else {}
        ),
    }
    if isinstance(reference, FrozenCase):
        contract.update(case=reference.id, policy_root=reference.policy_seed)
    return (
        scenario,
        settings,
        workload,
        provenance,
        reference.data_seed,
        {
            "frozen_world": primitive(scenario),
            "frozen_input_contract": contract,
        },
    )
