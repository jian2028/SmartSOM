"""Core guarantees migrated from matrix dispatch to explicit grid commands."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest
from test_production_runtime import act, loaded_machine, small_scenario

from smartsom.domain.factory_design import AGVDesign, Cell, GridDesign
from smartsom.domain.production import (
    Demand,
    JointCommand,
    MachineCommand,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator


def hand_commands():
    return (
        JointCommand(agvs=(("agv", "INTERACT"),)),
        JointCommand(agvs=(("agv", "RIGHT"),)),
        JointCommand(agvs=(("agv", "INTERACT"),)),
        JointCommand(
            machines=(("machine", MachineCommand("demand/attempt/1", "normal")),)
        ),
        JointCommand(),
        JointCommand(agvs=(("agv", "INTERACT"),)),
        JointCommand(agvs=(("agv", "RIGHT"),)),
        JointCommand(agvs=(("agv", "INTERACT"),)),
    )


def run_commands(scenario):
    sim = ProductionSimulator(scenario)
    return sim, [sim.step(command) for command in hand_commands()]


def test_entire_hand_timeline_and_repeat_execution():
    sim, records = run_commands(small_scenario())
    assert [
        (r["tick"], r["state"]["jobs"]["demand/attempt/1"]["location"]) for r in records
    ] == [
        (1, "agv"),
        (2, "agv"),
        (3, "machine"),
        (4, "machine"),
        (5, "machine"),
        (6, "agv"),
        (7, "agv"),
        (8, "output"),
    ]
    events = [e for row in records for e in row["events"]]
    assert [
        (e["kind"], e["tick"])
        for e in events
        if e["kind"] in ("processing_started", "processing_completed")
    ] == [
        ("processing_started", 3),
        ("processing_completed", 5),
    ]
    assert sim.status == "completed" and sim.completed == {"demand"}
    assert run_commands(small_scenario())[1] == records


@pytest.mark.parametrize(
    "command",
    [
        JointCommand(agvs=(("missing", "WAIT"),)),
        JointCommand(agvs=(("agv", "teleport"),)),
        JointCommand(machines=(("missing", MachineCommand()),)),
        JointCommand(quality=(("missing", "START"),)),
        JointCommand(rankings=(("input", ("invented",)),)),
    ],
)
def test_invalid_semantic_envelopes_reject_atomically(command):
    sim = ProductionSimulator(small_scenario())
    before = deepcopy(vars(sim))
    with pytest.raises(ValueError):
        sim.step(command)
    assert vars(sim) == before
    assert act(sim, "INTERACT")["tick"] == 1


def test_infeasible_proposal_does_not_cancel_independent_action():
    sim = ProductionSimulator(small_scenario())
    row = sim.step(
        JointCommand(
            agvs=(("agv", "INTERACT"),),
            machines=(("machine", MachineCommand("demand/attempt/1", "normal")),),
        )
    )
    assert row["rejections"] == {"machine:machine": "invalid"}
    assert sim.agvs["agv"]["job"] == "demand/attempt/1"
    assert sim.machine_state["machine"]["status"] == "IDLE"
    assert sim.tick == 1


def test_public_reads_are_detached_and_do_not_advance_or_emit_events():
    sim = ProductionSimulator(small_scenario())
    before = deepcopy(vars(sim))
    state, view = sim.snapshot(), sim.decision()
    state["agvs"]["agv"]["cell"][0] = 999
    view["jobs"].clear()
    sim.prepare_rankings({})
    assert vars(sim) == before


def test_reordering_unordered_factory_collections_preserves_every_record():
    scenario = small_scenario()
    _, expected = run_commands(scenario)
    for buffers in permutations(scenario.factory.buffers):
        for ports in permutations(scenario.factory.ports):
            changed = replace(
                scenario,
                factory=replace(scenario.factory, buffers=buffers, ports=ports),
            )
            assert run_commands(changed)[1] == expected


def test_movement_is_independent_of_command_insertion_order():
    case = small_scenario(mode="dynamic", tick_limit=5)
    factory = replace(
        case.factory,
        grid=GridDesign(4, 3),
        agvs=(
            AGVDesign("agv", "AGV", Cell(0, 0)),
            AGVDesign("other", "Other", Cell(1, 0)),
        ),
    )
    case = replace(case, factory=factory)
    proposals = (("agv", "RIGHT"), ("other", "RIGHT"))
    rows = [
        ProductionSimulator(case).step(JointCommand(agvs=ordering))
        for ordering in permutations(proposals)
    ]
    assert rows[0]["state"] == rows[1]["state"]
    assert rows[0]["events"] == rows[1]["events"]
    assert rows[0]["rejections"] == rows[1]["rejections"] == {}


def test_same_machine_successor_needs_explicit_unload_and_reload():
    case = replace(
        small_scenario(),
        demands=(
            Demand(
                "demand",
                (
                    ProductionStep("first", "drill", 2),
                    ProductionStep("second", "drill", 3),
                ),
            ),
        ),
    )
    sim = ProductionSimulator(case)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    assert sim.machine_choices("machine") == []
    act(sim, "INTERACT")
    act(sim, "INTERACT")
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    act(sim)
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    assert sim.tick == 13 and sim.completed == {"demand"}


def test_no_progress_is_truncation_and_terminal_calls_are_atomic():
    sim = ProductionSimulator(small_scenario(tick_limit=3))
    for _ in range(3):
        sim.step(JointCommand())
    assert sim.done and sim.status == "truncated" and not sim.completed
    before = deepcopy(vars(sim))
    with pytest.raises(ValueError, match="ended"):
        sim.step(JointCommand())
    assert vars(sim) == before


@pytest.mark.parametrize(
    "corruption,reason",
    [
        ("lost", "lost demand"),
        ("owner", "identity mismatch"),
        ("duplicate", "multiple physical owners"),
    ],
)
def test_conservation_invariants_detect_corrupted_state(corruption, reason):
    sim = ProductionSimulator(small_scenario())
    if corruption == "lost":
        sim.storage["input"]["pool"].clear()
    elif corruption == "owner":
        sim.jobs["demand/attempt/1"]["location"] = "agv"
    else:
        sim.storage["input"]["pool"].append("demand/attempt/1")
    with pytest.raises(AssertionError, match=reason):
        sim._check()


def test_hash_seed_does_not_change_entire_committed_trajectory():
    code = f"""
import json, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_static_engine import run_commands, small_scenario
sim, records = run_commands(small_scenario())
print(json.dumps(records, sort_keys=True))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        )
        for seed in ("1", "17", "321")
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])[-1]["tick"] == 8


def test_base_imports_do_not_attempt_optional_framework_or_config_imports():
    code = """
import importlib.abc, sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gymnasium','ray','pettingzoo','torch','ortools','pyjobshop','wandb','pydantic','yaml'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectOptional())
import smartsom
import smartsom.domain.production
import smartsom.engine.production
import smartsom.algorithms.production
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", ["SPTPolicy", "FirstFeasiblePolicy"])
def test_public_rule_names_drive_the_single_grid_core(name):
    import smartsom.algorithms as algorithms
    from smartsom.engine import Simulator

    scenario = small_scenario()
    sim = Simulator(scenario)
    assert type(sim) is ProductionSimulator
    policy = getattr(algorithms, name)(scenario.factory)
    while not sim.done:
        ranking = sim.prepare_rankings(policy.rank(sim.decision()))
        sim.step(policy.act(sim.decision(ranking)))
    assert sim.tick == 8 and sim.completed == {"demand"}
