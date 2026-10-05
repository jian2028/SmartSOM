"""Public V3 routes permit intermediate inspection and unchecked Output."""

from dataclasses import replace
from decimal import Decimal

import pytest
from test_travel_time_matrix import scenario

from smartsom.algorithms.production_composition import (
    BoundaryCoordinator,
    replay_boundary,
)
from smartsom.algorithms.production_rules import PolicyChoice, RulePolicy
from smartsom.domain.production import ProductionStep, QualitySample
from smartsom.engine.production import ProductionSimulator


def loaded_sim(*, step, quality, defective=False):
    original = scenario("auto")
    demand = replace(
        original.demands[0],
        steps=(
            ProductionStep("first", "operation_1", 1),
            ProductionStep("second", "operation_1", 1),
        ),
    )
    sim = ProductionSimulator(replace(original, demands=(demand,)), contract="v3")
    job = next(iter(sim.jobs))
    vehicle = next(iter(sim.agvs))
    sim._remove(job)
    sim.jobs[job].update(
        location=vehicle, slot=None, step=step, quality=quality, defective=defective
    )
    sim.agvs[vehicle]["job"] = job
    return sim, job, vehicle


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("quality", ["UNKNOWN", "PASS", "FAIL"])
def test_routes_use_observed_quality_and_processing_progress(step, quality):
    sim, job, vehicle = loaded_sim(step=step, quality=quality)
    destinations = set(sim.protocol.destination_owners(job))
    outputs = {o for o in sim.storage if sim.roles[o] == "system_output"}
    if quality == "FAIL":
        assert destinations == set(sim.scrap)
    else:
        assert destinations & set(sim.scrap) == set()
        assert destinations & set(sim.stations) == (
            set(sim.stations) if quality == "UNKNOWN" else set()
        )
        assert destinations & outputs == (outputs if step == 2 else set())
        machines = {
            sim.pre.get(m, m)
            for m, machine in sim.machines.items()
            if "operation_1" in machine.operation_types
        }
        assert destinations & machines == (machines if step < 2 else set())
    sim.protocol.begin()
    assert {c.action.owner for c in sim.protocol.dispatch_candidates(vehicle)} == (
        {o for o in destinations if sim.protocol.ports_for(o, "drop_off")}
    )
    sim.protocol.abort()


@pytest.mark.parametrize("step", [1, 2])
def test_unknown_routes_and_public_observations_do_not_reveal_latent_defect(step):
    observations = []
    candidates = []
    for defective in (False, True):
        sim, _, vehicle = loaded_sim(step=step, quality="UNKNOWN", defective=defective)
        requests = sim.protocol.begin()
        request = next(r for r in requests if r.owner == vehicle)
        observations.append(request.observation)
        candidates.append(request.candidates)
        sim.protocol.abort()
    assert observations[0] == observations[1]
    assert candidates[0] == candidates[1]


@pytest.mark.parametrize("defective", [False, True])
def test_unknown_output_reveals_quality_and_replaces_only_failed_attempt(defective):
    sim, job, vehicle = loaded_sim(step=2, quality="UNKNOWN", defective=defective)
    output = next(o for o in sim.storage if sim.roles[o] == "system_output")
    assert sim._admit(job, output)
    sim._transfer(vehicle, ("drop", job, output, sim._free(output)))
    assert sim.metrics["submitted"] == 1
    assert sim.metrics["passed"] == int(not defective)
    assert sim.metrics["output_rejected"] == int(defective)
    assert len(sim.completed) == int(not defective)
    assert sim.attempts[sim.jobs[job]["demand"]] == 1 + int(defective)
    assert sim.metrics["pre_output_scrap"] == 0


def test_new_processing_resets_pass_and_preserves_accumulated_defect():
    sim, job, _ = loaded_sim(step=1, quality="PASS", defective=True)
    machine = next(iter(sim.machines))
    sim.jobs[job]["location"] = machine
    sim.machine_state[machine].update(
        job=job,
        status="PROCESSING",
        remaining=1,
        elapsed=0,
        mode=sim.machines[machine].quality_modes[0].quality_mode_id,
    )
    sim._advance()
    assert sim.jobs[job]["step"] == 2
    assert sim.jobs[job]["quality"] == "UNKNOWN"
    assert sim.jobs[job]["defective"] is True
    assert set(sim.stations) <= set(sim.protocol.destination_owners(job))


class IntermediateInspection(RulePolicy):
    """Exercise intermediate inspection using only public jobs and candidates."""

    def choose(self, request):
        vehicle = request.observation["agvs"][request.owner]
        if vehicle["job"] is None:
            return super().choose(request)
        job = request.observation["jobs"][vehicle["job"]]
        targets = [c for c in request.candidates if c.action is not None]
        if job["step"] == 1 and job["quality"] == "UNKNOWN":
            targets = [c for c in targets if c.action.owner.startswith("inspection")]
        elif job["step"] < 2 and job["quality"] != "FAIL":
            targets = [c for c in targets if c.action.owner.startswith("buffer")]
        else:
            targets = [
                c for c in targets if not c.action.owner.startswith("inspection")
            ]
        return PolicyChoice(
            min(targets, key=lambda c: (c.features[1], c.identity)).action
        )


@pytest.mark.parametrize("first_defective", [False, True])
def test_intermediate_inspection_disposal_and_processing_reset_replay(first_defective):
    original = scenario("auto")
    demand = replace(
        original.demands[0],
        steps=(
            ProductionStep("first", "operation_1", 1),
            ProductionStep("second", "operation_1", 1),
        ),
    )
    factory = replace(
        original.factory,
        machines=tuple(
            replace(
                machine,
                quality_modes=tuple(
                    replace(mode, error_rate=Decimal("0.5"))
                    for mode in machine.quality_modes
                ),
            )
            for machine in original.factory.machines
        ),
    )
    case = replace(
        original,
        factory=factory,
        demands=(demand,),
        tick_limit=2000,
        quality_samples=tuple(
            QualitySample(
                demand.demand_id,
                operation,
                0
                if first_defective and attempt == 1 and operation == "first"
                else 2**53 - 1,
                attempt=attempt,
            )
            for attempt in (1, 2)
            for operation in ("first", "second")
        ),
    )
    sim = ProductionSimulator(case, contract="v3")
    replay = ProductionSimulator(case, contract="v3")
    policies = {
        "machine": RulePolicy("machine", "normal_first"),
        "buffer": RulePolicy("buffer", "edd"),
        "dispatcher": IntermediateInspection("dispatcher", "nearest"),
        "mover": RulePolicy("mover", "automatic_travel"),
    }
    controller = BoundaryCoordinator(
        sim, policies, {role: {"default": role} for role in policies}
    )
    saw_intermediate_pass = False
    saw_reset = False
    while not sim.done:
        row = controller.tick()
        replay_boundary(replay, row)
        assert replay.snapshot() == sim.snapshot()
        for job in sim.jobs.values():
            saw_intermediate_pass |= job["step"] == 1 and job["quality"] == "PASS"
            saw_reset |= job["step"] == 2 and job["quality"] == "UNKNOWN"
    assert sim.status == "completed"
    assert saw_intermediate_pass and saw_reset
    assert sim.metrics["pre_output_scrap"] == int(first_defective)
    assert sim.metrics["submitted"] == sim.metrics["passed"] == 1
    assert sim.attempts[demand.demand_id] == 1 + int(first_defective)
