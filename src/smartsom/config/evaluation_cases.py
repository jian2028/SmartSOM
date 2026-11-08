"""Portable fixed-case inventories, including exact imported physical worlds."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from smartsom.config.codec import StrictModel, digest, primitive
from smartsom.config.scenario_snapshot import scenario_from_snapshot


class FixedCaseSeed(StrictModel):
    """A portable world recipe; the optional hash pins an existing physical world."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)
    id: str = Field(min_length=1)
    condition: Literal["normal", "low", "medium", "high"]
    seed: int = Field(ge=0, strict=True)
    action_seeds: tuple[int, ...] = Field(min_length=1)
    scenario_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def explicit_action_seeds(cls, data):
        seeds = data.get("action_seeds", ()) if isinstance(data, dict) else ()
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("action_seeds require nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("action_seeds must be unique")
        return data


class FixedTestPanel(StrictModel):
    """Authored inside Experiment; runtime inventories are generated artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)
    conditions: tuple[Literal["normal", "low", "medium", "high"], ...]
    cases_per_condition: int = Field(gt=0, strict=True)
    cases: tuple[FixedCaseSeed, ...]

    @model_validator(mode="after")
    def balanced_unique_cases(self):
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("test conditions must be nonempty and unique")
        if Counter(c.condition for c in self.cases) != {
            c: self.cases_per_condition for c in self.conditions
        }:
            raise ValueError("test cases must match conditions and cases_per_condition")
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("test case IDs must be unique")
        if len({(c.condition, c.seed) for c in self.cases}) != len(self.cases):
            raise ValueError("test condition/seed pairs must be unique")
        return self


def inline_test_cases(panel, *, factory, workload, condition_inputs, drain_ticks):
    data = {
        "schema": "smartsom.case-recipe/v1",
        "split": "test",
        "factory_sha256": digest(primitive(factory)),
        "workload_sha256": digest(workload.model_dump(mode="json")),
        "condition_inputs_sha256": condition_input_identity(condition_inputs),
        "drain_ticks": drain_ticks,
        "cases": [c.model_dump(mode="json", exclude_none=True) for c in panel.cases],
    }
    return validate_cases(
        data,
        split="test",
        factory=factory,
        workload=workload,
        condition_inputs=condition_inputs,
        allow_unpinned=True,
    )


def experiment_panels(experiment, directory, factory, workload, *, drain_ticks=None):
    if drain_ticks is None:
        drain_ticks = experiment.drain_ticks
        if experiment.evaluation.mode == "completion":
            drain_ticks = math.ceil(
                drain_ticks * experiment.evaluation.completion_drain_multiplier
            )
    panels = {
        split: load_cases(
            Path(directory) / reference,
            split=split,
            factory=factory,
            workload=workload,
            condition_inputs=experiment._condition_inputs,
        )
        for split, reference in experiment.case_sets.items()
    }
    if experiment.test is not None:
        panels["test"] = inline_test_cases(
            experiment.test,
            factory=factory,
            workload=workload,
            condition_inputs=experiment._condition_inputs,
            drain_ticks=drain_ticks,
        )
    return panels


def condition_input_identity(inputs):
    return {
        condition: {
            "factory_sha256": digest(primitive(factory)),
            "workload_sha256": digest(workload.model_dump(mode="json")),
        }
        for condition, (factory, workload) in inputs.items()
    }


def load_cases(path, *, split, factory, workload, condition_inputs=None):
    return validate_cases(
        json.loads(Path(path).read_text()),
        split=split,
        factory=factory,
        workload=workload,
        condition_inputs=condition_inputs,
    )


def validate_cases(
    data, *, split, factory, workload, condition_inputs=None, allow_unpinned=False
):
    if (
        data.get("schema") not in {"smartsom.fixed-cases/v1", "smartsom.case-recipe/v1"}
        or data.get("split") != split
    ):
        raise ValueError("fixed cases require the matching split and schema")
    if data.get("factory_sha256") != digest(primitive(factory)) or data.get(
        "workload_sha256"
    ) != digest(workload.model_dump(mode="json")):
        raise ValueError("fixed cases differ from the configured factory/workload")
    condition_inputs = condition_inputs or {}
    if data.get("condition_inputs_sha256", {}) != condition_input_identity(
        condition_inputs
    ):
        raise ValueError("fixed cases differ from the configured condition scenarios")
    cases = [dict(case) for case in data.get("cases", [])]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("fixed cases require nonempty unique case IDs")
    identities = set()
    keys = set()
    for case in cases:
        if case["condition"] not in {"normal", "low", "medium", "high"}:
            raise ValueError("unknown fixed case condition")
        selected_factory, selected_workload = condition_inputs.get(
            case["condition"], (factory, workload)
        )
        if type(case["seed"]) is not int:
            raise ValueError("fixed case seed must be an integer")
        key = (case["condition"], case["seed"])
        if key in keys:
            raise ValueError("duplicate condition/seed in fixed cases")
        keys.add(key)
        if data["schema"] == "smartsom.case-recipe/v1":
            from smartsom.config.base_diagnostic import generate_scenario

            case["scenario"] = primitive(
                generate_scenario(
                    selected_factory,
                    selected_workload,
                    case["seed"],
                    case["condition"],
                    data["drain_ticks"],
                )
            )
            actual_hash = digest(case["scenario"])
            if (
                case.get("scenario_sha256", actual_hash if allow_unpinned else None)
                != actual_hash
            ):
                raise ValueError(
                    "case recipe no longer reproduces its frozen historical world"
                )
        scenario = scenario_from_snapshot(case["scenario"])
        if primitive(scenario.factory) != primitive(selected_factory):
            raise ValueError("imported scenario factory mismatch")
        if condition_inputs and data["schema"] == "smartsom.fixed-cases/v1":
            from smartsom.config.base_diagnostic import generate_scenario

            drain = scenario.tick_limit - max(d.release_at for d in scenario.demands)
            expected = generate_scenario(
                selected_factory,
                selected_workload,
                case["seed"],
                case["condition"],
                drain,
            )
            if primitive(expected) != primitive(scenario):
                raise ValueError(
                    "fixed case world differs from its condition scenario/seed"
                )
        identity = digest(primitive(scenario))
        if identity in identities:
            raise ValueError("duplicate physical world in fixed case split")
        identities.add(identity)
        seeds = case.get("action_seeds")
        if (
            not seeds
            or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("fixed cases require unique explicit action_seeds")
    return cases


def validate_panel_separation(panels, experiment, factory, workload, drain):
    """Compare actual worlds even when only one split is explicitly imported."""
    from smartsom.config.base_diagnostic import generate_scenario, seed_for

    if not panels:
        return
    identities = {}
    for split, namespace in (("periodic_evaluation", 80000), ("test", 90000)):
        if split in panels:
            identities[split] = {digest(c["scenario"]) for c in panels[split]}
        else:
            identities[split] = {
                digest(
                    primitive(
                        generate_scenario(
                            *experiment.inputs_for(
                                ("normal", "low", "medium", "high")[index % 4],
                                factory,
                                workload,
                            ),
                            seed_for(namespace, index // 4) % (2**31),
                            ("normal", "low", "medium", "high")[index % 4],
                            drain,
                        )
                    )
                )
                for index in range(getattr(experiment, f"{split}_cases"))
            }
    if identities["periodic_evaluation"] & identities["test"]:
        raise ValueError("periodic_evaluation and test contain the same physical world")
