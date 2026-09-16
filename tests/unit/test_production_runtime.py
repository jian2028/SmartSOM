"""Hand-calculated physical transitions, independent of playback projections."""

from dataclasses import replace

import pytest

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferTarget,
    Cell,
    FactoryDesign,
    Footprint,
    GridDesign,
    MachineDesign,
    MachineTarget,
    PoolStorage,
    PortBinding,
    PortDesign,
)
from smartsom.domain.production import (
    Demand,
    JointCommand,
    MachineCommand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator


def small_scenario(**kwargs):
    factory = FactoryDesign(
        "small",
        "Hand calculated",
        GridDesign(3, 2),
        operation_types=("drill",),
        machines=(
            MachineDesign(
                "machine", "Machine", Footprint(1, 1, 1, 1), operation_types=("drill",)
            ),
        ),
        buffers=(
            BufferDesign(
                "input",
                "Input",
                Footprint(0, 1, 1, 1),
                "system_input",
                PoolStorage(None),
            ),
            BufferDesign(
                "output",
                "Output",
                Footprint(2, 1, 1, 1),
                "system_output",
                PoolStorage(None),
            ),
        ),
        ports=(
            PortDesign(
                "in_port",
                "Input",
                Cell(0, 0),
                bindings=(PortBinding(BufferTarget("input"), ("pickup",)),),
            ),
            PortDesign(
                "machine_port",
                "Machine",
                Cell(1, 0),
                bindings=(PortBinding(MachineTarget("machine")),),
            ),
            PortDesign(
                "out_port",
                "Output",
                Cell(2, 0),
                bindings=(PortBinding(BufferTarget("output"), ("drop_off",)),),
            ),
        ),
        agvs=(AGVDesign("agv", "AGV", Cell(0, 0)),),
    )
    return ProductionScenario(
        factory, (Demand("demand", (ProductionStep("op", "drill", 2),)),), **kwargs
    )


def act(sim, action="WAIT", machine=None):
    return sim.step(
        JointCommand(
            agvs=(("agv", action),),
            machines=() if machine is None else (("machine", machine),),
        )
    )


def loaded_machine(sim):
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    return sim.machine_state["machine"]["job"]


def test_machine_duration_override_changes_time_without_changing_capability():
    scenario = small_scenario()
    step = ProductionStep("op", "drill", 2, {"machine": 5})
    scenario = replace(scenario, demands=(Demand("demand", (step,)),))
    sim = ProductionSimulator(scenario)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    assert sim.machine_state["machine"]["nominal"] == 5
    assert sim.machine_state["machine"]["remaining"] == 4
    assert step.ticks_on("another_machine") == 2


@pytest.mark.parametrize("machine", ["missing", "machine"])
def test_duration_override_cannot_grant_capability(machine):
    scenario = small_scenario()
    factory = scenario.factory
    if machine == "machine":
        factory = replace(
            factory,
            machines=(replace(factory.machines[0], operation_types=()),),
        )
    scenario = replace(
        scenario,
        factory=factory,
        demands=(Demand("demand", (ProductionStep("op", "drill", 2, {machine: 3}),)),),
    )
    with pytest.raises(ValueError, match="unknown or incapable"):
        ProductionSimulator(scenario)


def test_machine_duration_snapshot_roundtrip_is_immutable_and_portable():
    from smartsom.config.codec import primitive
    from smartsom.config.production import scenario_from_snapshot

    table = {"machine": 5}
    step = ProductionStep("op", "drill", 2, table)
    table["machine"] = 999
    scenario = replace(small_scenario(), demands=(Demand("demand", (step,)),))
    payload = primitive(scenario)
    assert payload["demands"][0]["steps"][0]["machine_nominal_ticks"] == {"machine": 5}
    assert scenario_from_snapshot(payload) == scenario


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_machine_duration_rejects_invalid_ticks(value):
    with pytest.raises(ValueError, match="positive integers"):
        ProductionStep("op", "drill", 2, {"machine": value})


def test_hand_calculated_eight_ticks():
    sim = ProductionSimulator(small_scenario())
    job = loaded_machine(sim)
    assert sim.tick == 3
    assert sim.machine_state["machine"]["status"] == "READY"
    act(sim, machine=MachineCommand(job, "normal"))
    assert sim.tick == 4
    assert sim.machine_state["machine"]["remaining"] == 1
    act(sim)
    assert sim.tick == 5
    assert sim.machine_state["machine"]["status"] == "BLOCKED"
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    result = act(sim, "INTERACT")
    assert sim.tick == 8
    assert sim.done and sim.status == "completed"
    assert sim.completed == {"demand"}
    assert sim.agvs["agv"]["job"] is None
    assert result["state"]["metrics"]["passed"] == 1
    with pytest.raises(ValueError, match="ended"):
        act(sim)


def test_outage_pauses_remaining_work_without_restarting():
    sim = ProductionSimulator(small_scenario(outages=(Outage("machine", 4, 7),)))
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    assert sim.tick == 4 and sim.machine_state["machine"]["down"]
    for _ in range(3):
        act(sim)
        assert sim.machine_state["machine"]["remaining"] == 1
    assert not sim.machine_state["machine"]["down"]
    act(sim)
    assert sim.tick == 8 and sim.machine_state["machine"]["status"] == "BLOCKED"
    assert sim.machine_state["machine"]["elapsed"] == 2


def test_swaps_cancel_both_and_propagate_stationary_occupancy():
    case = small_scenario(mode="dynamic", tick_limit=10)
    factory = replace(
        case.factory,
        agvs=case.factory.agvs + (AGVDesign("other", "Other", Cell(1, 0)),),
    )
    sim = ProductionSimulator(replace(case, factory=factory))
    row = sim.step(JointCommand(agvs=(("agv", "RIGHT"), ("other", "LEFT"))))
    assert row["rejections"] == {"agv:agv": "conflict", "agv:other": "conflict"}
    assert sim.agvs["agv"]["cell"] == [0, 0]
    row = sim.step(JointCommand(agvs=(("agv", "RIGHT"), ("other", "RIGHT"))))
    assert row["rejections"] == {}
    assert sim.agvs["agv"]["cell"] == [1, 0]


def test_rankings_are_complete_stable_identity_permutations():
    case = small_scenario()
    case = replace(
        case, demands=case.demands + (replace(case.demands[0], demand_id="second"),)
    )
    sim = ProductionSimulator(case)
    jobs = sim.selectable("input")
    ranking = {"input": list(reversed(jobs))}
    assert sim.prepare_rankings(ranking)["input"] == list(reversed(jobs))
    assert sim.interaction("agv", ranking)[1] == jobs[-1]
    with pytest.raises(ValueError, match="cover"):
        sim.prepare_rankings({"input": [jobs[0], jobs[0]]})
    assert sim.tick == 0


def test_snapshot_is_detached_and_public_does_not_leak_defects():
    sim = ProductionSimulator(small_scenario())
    state = sim.snapshot(public=True)
    assert all("defective" not in row for row in state["jobs"].values())
    state["agvs"]["agv"]["cell"][0] = 99
    assert sim.agvs["agv"]["cell"] == [0, 0]


def test_rule_run_recording_and_relocated_playback(tmp_path):
    import shutil

    from smartsom.config.production import AlgorithmConfig, scenario_from_snapshot
    from smartsom.experiments.production import execute
    from smartsom.trace.production import Playback

    root = execute(
        small_scenario(), AlgorithmConfig(), output_root=tmp_path, verbose=False
    )
    assert sorted(x.name for x in root.iterdir()) == ["run.json", "trace.jsonl"]
    recording = Playback(root)
    assert recording.last_tick == 8
    assert recording.row(3)["state"]["machines"]["machine"]["status"] == "READY"
    assert recording.row(5)["state"]["machines"]["machine"]["status"] == "BLOCKED"
    assert recording.row(8)["state"]["completed"] == ["demand"]
    assert (
        scenario_from_snapshot(recording.manifest["inputs"]["scenario"])
        == small_scenario()
    )
    moved = tmp_path / "moved"
    expected = recording.row(7)
    shutil.move(root, moved)
    assert Playback(moved).row(7) == expected


def test_record_false_does_not_change_run(tmp_path):
    import json

    from smartsom.config.production import AlgorithmConfig
    from smartsom.experiments.production import execute
    from smartsom.trace.production import Playback

    a = execute(
        small_scenario(), AlgorithmConfig(), output_root=tmp_path, verbose=False
    )
    b = execute(
        small_scenario(),
        AlgorithmConfig(),
        output_root=tmp_path,
        verbose=False,
        record=False,
    )
    assert [x.name for x in b.iterdir()] == ["run.json"]
    assert (
        json.loads((a / "run.json").read_text())["result"]
        == json.loads((b / "run.json").read_text())["result"]
    )
    with pytest.raises(ValueError, match="no recorded"):
        Playback(b)


def test_applied_rank_is_recorded_before_pickup_prunes_state():
    sim = ProductionSimulator(small_scenario())
    jobs = sim.selectable("input")
    row = act(sim, "INTERACT")
    assert row["rankings"]["input"] == jobs
    assert row["state"]["rankings"]["input"] == []


def test_finite_output_capacity_blocks_delivery_and_zero_never_admits():
    for capacity in (0, 1):
        case = small_scenario()
        factory = replace(
            case.factory,
            buffers=(
                case.factory.buffers[0],
                replace(case.factory.buffers[1], storage=PoolStorage(capacity)),
            ),
        )
        case = replace(
            case,
            factory=factory,
            demands=case.demands + (replace(case.demands[0], demand_id="second"),),
            tick_limit=80,
        )
        from smartsom.algorithms.production import GreedyProductionPolicy

        sim = ProductionSimulator(case)
        policy = GreedyProductionPolicy(factory)
        while not sim.done:
            sim.step(policy.act(sim.decision(policy.rank(sim.decision()))))
        assert len(sim.completed) == capacity
        assert len(sim.storage["output"]["pool"]) == capacity
        assert sim.status == "truncated"


def test_ambiguous_commands_are_rejected_before_execution():
    with pytest.raises(ValueError, match="duplicate resource"):
        JointCommand(agvs=(("agv", "WAIT"), ("agv", "RIGHT")))
    with pytest.raises(ValueError, match="both"):
        MachineCommand("job")


def test_announcements_do_not_release_jobs_early():
    case = small_scenario(mode="dynamic", tick_limit=10)
    sim = ProductionSimulator(
        replace(case, demands=(replace(case.demands[0], release_at=4, reveal_at=2),))
    )
    assert not sim.jobs and not sim.decision()["announced"]
    act(sim)
    act(sim)
    assert sim.decision()["announced"][0]["demand_id"] == "demand"
    assert not sim.jobs
    act(sim)
    act(sim)
    assert sim.selectable("input") == ["demand/attempt/1"]
    assert not sim.decision()["announced"]


def test_recording_audit_reexecutes_commands(tmp_path):
    from smartsom.config.production import AlgorithmConfig
    from smartsom.experiments.production import execute
    from smartsom.trace.production import audit

    root = execute(
        small_scenario(), AlgorithmConfig(), output_root=tmp_path, verbose=False
    )
    result = audit(root)
    assert result["status"] == "passed"


def quality_scenario(error_rate="0", **kwargs):
    from decimal import Decimal

    from smartsom.domain.factory_design import (
        InspectionSlotTarget,
        InspectionStationDesign,
        ScrapBinDesign,
        ScrapBinTarget,
        SlotDesign,
    )
    from smartsom.domain.quality import QualityMode

    case = small_scenario(**kwargs)
    factory = replace(
        case.factory,
        factory_id="quality",
        grid=GridDesign(5, 2),
        machines=(
            replace(
                case.factory.machines[0],
                quality_modes=(
                    QualityMode("normal", Decimal(1), Decimal(error_rate)),
                    QualityMode("fast", Decimal("0.5"), Decimal("0.5")),
                ),
            ),
        ),
        buffers=(
            case.factory.buffers[0],
            replace(case.factory.buffers[1], footprint=Footprint(4, 1, 1, 1)),
        ),
        inspection_stations=(
            InspectionStationDesign(
                "inspection",
                "Inspection",
                Footprint(2, 1, 1, 1),
                slots=(SlotDesign("slot", Cell(0, 0)),),
            ),
        ),
        scrap_bins=(ScrapBinDesign("scrap", "Scrap", Footprint(3, 1, 1, 1)),),
        ports=case.factory.ports[:2]
        + (
            PortDesign(
                "inspection_port",
                "Inspection",
                Cell(2, 0),
                bindings=(PortBinding(InspectionSlotTarget("inspection", "slot")),),
            ),
            PortDesign(
                "scrap_port",
                "Scrap",
                Cell(3, 0),
                bindings=(PortBinding(ScrapBinTarget("scrap"), ("drop_off",)),),
            ),
            replace(case.factory.ports[2], cell=Cell(4, 0)),
        ),
    )
    return replace(case, factory=factory)


def inspected_job(sim):
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    assert sim.tick == 8 and sim.jobs[job]["location"] == "inspection"
    return job


@pytest.mark.parametrize("error_rate,expected", [("0", "PASS"), ("1", "FAIL")])
def test_inspection_locks_station_and_reveals_once(error_rate, expected):
    sim = ProductionSimulator(quality_scenario(error_rate))
    job = inspected_job(sim)
    # START and a simultaneous pickup conflict; neither wins implicitly.
    row = sim.step(
        JointCommand(agvs=(("agv", "INTERACT"),), quality=(("inspection", "START"),))
    )
    assert row["rejections"] == {
        "quality:inspection": "conflict",
        "agv:agv": "conflict",
    }
    assert sim.station_state["inspection"]["status"] == "IDLE"
    sim.step(JointCommand(quality=(("inspection", "START"),)))
    assert sim.station_state["inspection"]["remaining"] == 1
    assert sim.interaction("agv") is None
    assert sim.jobs[job]["quality"] == "UNKNOWN"
    row = act(sim)
    assert sim.jobs[job]["quality"] == expected
    assert sim.inspection_jobs("inspection") == []
    assert sum(e["kind"] == "quality_revealed" for e in row["events"]) == 1
    assert sim.interaction("agv")[1] == job


def test_confirmed_scrap_replaces_original_demand_without_duplicate():
    sim = ProductionSimulator(quality_scenario("1"))
    job = inspected_job(sim)
    sim.step(JointCommand(quality=(("inspection", "START"),)))
    act(sim)
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    replacement = sim.selectable("input")
    assert replacement == ["demand/attempt/2"]
    assert sim.jobs[replacement[0]]["demand"] == sim.jobs[job]["demand"]
    assert sim.metrics["pre_output_scrap"] == 1
    assert not sim.completed


def test_quality_rule_can_complete_multiple_demands():
    from smartsom.algorithms.production import GreedyProductionPolicy

    case = quality_scenario()
    case = replace(
        case,
        demands=tuple(
            replace(case.demands[0], demand_id=f"demand_{i}") for i in range(3)
        ),
    )
    sim = ProductionSimulator(case)
    policy = GreedyProductionPolicy(case.factory)
    while not sim.done:
        sim.step(policy.act(sim.decision(policy.rank(sim.decision()))))
    assert sim.status == "completed"
    assert len(sim.completed) == 3


def test_post_buffer_holds_completion_and_blocks_until_capacity_frees():
    case = small_scenario()
    # A capacity-zero POST is a real unavailable facility, not infinite caching.
    post = BufferDesign(
        "post",
        "Post",
        Footprint(3, 1, 1, 1),
        "machine_post",
        PoolStorage(0),
        machine_id="machine",
    )
    factory = replace(
        case.factory, grid=GridDesign(4, 2), buffers=case.factory.buffers + (post,)
    )
    sim = ProductionSimulator(replace(case, factory=factory))
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    assert sim.machine_state["machine"]["status"] == "BLOCKED"
    assert sim.machine_state["machine"]["job"] == job
    assert sim.interaction("agv") is None
    assert not sim.machine_choices("machine")


def test_new_post_jobs_are_available_at_next_decision_boundary():
    case = small_scenario()
    post = BufferDesign(
        "post",
        "Post",
        Footprint(3, 1, 1, 1),
        "machine_post",
        PoolStorage(1),
        machine_id="machine",
    )
    port = PortDesign(
        "post_port",
        "Post",
        Cell(3, 0),
        bindings=(PortBinding(BufferTarget("post"), ("pickup",)),),
    )
    factory = replace(
        case.factory,
        grid=GridDesign(4, 2),
        buffers=case.factory.buffers + (post,),
        ports=case.factory.ports + (port,),
    )
    sim = ProductionSimulator(replace(case, factory=factory))
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    row = act(sim)
    assert sim.tick == 5
    assert sim.machine_state["machine"]["status"] == "IDLE"
    assert sim.storage["post"]["pool"] == [job]
    assert sim.decision()["rankings"]["post"] == [job]
    assert any(
        e["kind"] == "machine_released" and e["tick"] == 5 for e in row["events"]
    )


def test_completion_at_outage_boundary_wins():
    case = small_scenario(outages=(Outage("machine", 5, 9),))
    sim = ProductionSimulator(case)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    assert sim.tick == 5
    assert sim.machine_state["machine"]["down"]
    assert sim.machine_state["machine"]["status"] == "BLOCKED"
    assert sim.jobs[job]["step"] == 1


def test_builtin_designs_resolve_capabilities_with_energy_inactive():
    from smartsom.studio.templates import load_template_1, load_template_2

    for load in (load_template_1, load_template_2):
        factory = load()
        case = ProductionScenario(
            factory,
            (
                Demand(
                    "job",
                    tuple(
                        ProductionStep(f"op_{i}", kind, 2)
                        for i, kind in enumerate(factory.operation_types)
                    ),
                ),
            ),
        )
        sim = ProductionSimulator(case)
        assert len(sim.machines) in (4, 8)
        assert len(sim.agvs) == 4


def test_fixed_processing_sample_preserves_nominal_then_applies_quality_multiplier():
    from decimal import Decimal

    from smartsom.domain.production import ProcessingSample

    case = small_scenario(
        processing_samples=(ProcessingSample("demand", "op", "machine", 5),),
        processing_low=10,
        processing_high=10,
    )
    machine = case.factory.machines[0]
    mode = replace(machine.quality_modes[0], time_scale=Decimal("0.5"))
    case = replace(
        case,
        factory=replace(
            case.factory, machines=(replace(machine, quality_modes=(mode,)),)
        ),
    )
    sim = ProductionSimulator(case)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    state = sim.machine_state["machine"]
    assert state["nominal"] == 1
    assert state["elapsed"] == 1 and state["remaining"] == 2  # half-up: 5 * 0.5 -> 3
    assert "remaining" not in sim.snapshot(public=True)["machines"]["machine"]


def test_fixed_quality_draw_and_hidden_probability_do_not_leak_unrevealed_result():
    from decimal import Decimal

    from smartsom.domain.production import QualitySample

    case = small_scenario(
        quality_samples=(QualitySample("demand", "op", 0),),
        quality_probability_visibility="hidden",
    )
    machine = case.factory.machines[0]
    mode = replace(machine.quality_modes[0], error_rate=Decimal("0.5"))
    case = replace(
        case,
        factory=replace(
            case.factory, machines=(replace(machine, quality_modes=(mode,)),)
        ),
    )
    sim = ProductionSimulator(case)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    assert sim.jobs[job]["defective"] is True
    public = sim.snapshot(public=True)["jobs"][job]
    assert "defective" not in public and public["quality"] == "UNKNOWN"
    assert public["risk"] == -1
    assert sim._draw("quality", job, "op") == 0
    assert sim._draw("quality", "demand/attempt/2", "op") != 0


@pytest.mark.parametrize(
    "sample",
    [
        ("missing", "op", "machine", 2),
        ("demand", "missing", "machine", 2),
        ("demand", "op", "missing", 2),
    ],
)
def test_fixed_processing_samples_reject_invalid_references(sample):
    from smartsom.domain.production import ProcessingSample

    with pytest.raises(ValueError, match="sample references"):
        ProductionSimulator(
            small_scenario(processing_samples=(ProcessingSample(*sample),))
        )


@pytest.mark.parametrize(
    "field", ["demands", "outages", "processing_samples", "quality_samples"]
)
def test_direct_scenario_rejects_untyped_rows(field):
    with pytest.raises(ValueError, match="must contain"):
        replace(small_scenario(), **{field: ({"unknown": "row"},)})


def test_direct_demand_and_outage_reject_invalid_types():
    with pytest.raises(ValueError, match="ProductionStep"):
        Demand("demand", ({"operation_id": "op"},))
    with pytest.raises(ValueError, match="input_id"):
        replace(small_scenario().demands[0], input_id=True)
    with pytest.raises(ValueError, match="machine_id"):
        Outage(True, 1, 2)


@pytest.mark.parametrize("replication", range(5))
def test_rule_routes_around_parked_agvs_and_does_not_cycle_through_storage(replication):
    from pathlib import Path

    from smartsom.algorithms.production import GreedyProductionPolicy
    from smartsom.config import resolve_training_run
    from smartsom.config.study import study_roots

    recipe = resolve_training_run(
        Path(__file__).resolve().parents[2] / "configs/runs/learning_sb3.yaml"
    ).resolved
    scenario = recipe.episode(study_roots(202, "S00-micro", replication, "")[0])
    sim = ProductionSimulator(scenario)
    policy = GreedyProductionPolicy(scenario.factory, rule="spt")
    qualified_at = None
    while not sim.done:
        rankings = sim.prepare_rankings(policy.rank(sim.decision()))
        sim.step(policy.act(sim.decision(rankings)))
        if len(sim.completed) == len(scenario.demands) and qualified_at is None:
            qualified_at = sim.tick
    assert set(sim.completed) == {d.demand_id for d in scenario.demands}
    assert qualified_at < scenario.tick_limit
