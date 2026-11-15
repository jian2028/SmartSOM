"""Compile scientific case inventories for resident RLlib evaluation workers.

No framework is imported here. The caller supplies the policy resolved from the
official AlgorithmConfig; this compiler never chooses an exploration override.
"""

import json
from math import ceil

from smartsom.config.base_diagnostic import generate_scenario, seed_for
from smartsom.config.codec import digest
from smartsom.config.evaluation_cases import (
    experiment_panels,
    validate_panel_separation,
)
from smartsom.config.scenario_snapshot import scenario_from_snapshot


def compile_panels(experiment, directory, factory, workload, *, drain, mode):
    """Preserve frozen worlds and every prescribed action-seed replicate."""
    if mode not in {"sample", "argmax"}:
        raise ValueError("evaluation policy must resolve to sample or argmax")
    protocol = experiment.evaluation
    actual_drain = (
        ceil(drain * protocol.completion_drain_multiplier)
        if protocol.mode == "completion"
        else drain
    )
    inventories = experiment_panels(
        experiment, directory, factory, workload, drain_ticks=actual_drain
    )
    validate_panel_separation(inventories, experiment, factory, workload, actual_drain)
    panels = {}
    for split, namespace in (("periodic_evaluation", 80000), ("test", 90000)):
        count = getattr(experiment, split + "_cases")
        imported = inventories.get(split)
        if imported is not None:
            expected = experiment.case_set_sha256.get(split)
            if expected is not None and digest(imported) != expected:
                raise ValueError(
                    "fixed case inventory digest differs from frozen inputs"
                )
            if count != len(imported):
                raise ValueError("evaluation count differs from frozen case inventory")
        elif count % 4:
            raise ValueError("evaluation cases must be balanced across four conditions")
        rows = []
        for index in range(count):
            case = imported[index] if imported is not None else None
            condition = (
                case["condition"]
                if case is not None
                else ("normal", "low", "medium", "high")[index % 4]
            )
            world_seed = (
                case["seed"]
                if case is not None
                else seed_for(namespace, index // 4) % (2**31)
            )
            scenario = (
                scenario_from_snapshot(case["scenario"])
                if case is not None
                else generate_scenario(
                    *experiment.inputs_for(condition, factory, workload),
                    world_seed,
                    condition,
                    actual_drain,
                )
            )
            if (
                case is not None
                and protocol.mode == "completion"
                and scenario.tick_limit
                != max(d.release_at for d in scenario.demands) + actual_drain
            ):
                raise ValueError(
                    "completion evaluation requires cases frozen through protection horizon"
                )
            seeds = (
                case["action_seeds"]
                if case is not None
                else range(experiment.periodic_evaluation.sample_action_seeds)
            )
            if mode == "argmax":
                seeds = list(seeds)[:1]
            identity = case["id"] if case is not None else str(index)
            for replicate in seeds:
                # Legacy generated panels name replicates by index, but seed
                # policy sampling from the world plus that index. Imported
                # panels already carry explicit effective sampling seeds.
                action_seed = (
                    replicate
                    if case is not None
                    else seed_for(world_seed, replicate) % (2**31)
                )
                rows.append(
                    {
                        "case_id": json.dumps([split, identity, replicate]),
                        "world_id": identity,
                        "condition": condition,
                        "world_seed": world_seed,
                        "action_seed": action_seed,
                        "action_seed_identity": replicate,
                        "action_seed_explicit": case is not None,
                        "mode": mode,
                        "scenario": scenario,
                    }
                )
        panels[split] = rows
    return panels
