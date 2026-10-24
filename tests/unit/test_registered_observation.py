"""Pure native tensor and public-field contracts; no child-process lifecycle."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from test_production_runtime import (
    act,
    automatic_inspection_case,
    carry_to_inspection,
    small_scenario,
)

from smartsom.config.codec import primitive
from smartsom.config.extensions import ExtensionRef
from smartsom.config.observation_contracts import (
    NETWORK,
    OBSERVATION,
    pin_contract,
    validate_factory,
)
from smartsom.domain.production import MachineCommand
from smartsom.domain.production_decisions import Candidate, DecisionRequest
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.production_protocol import ProductionProtocol
from smartsom.learning.observation_factory import encoder, network_class
from smartsom.learning.production_models import default_network, tensor_inputs
from smartsom.learning.registered_observation import (
    PublicFieldsEncoder,
    nominal_ticks,
    restricted_view,
)


def component(name, normalization="raw", visibility="both", aggregation="per_entity"):
    return dict(
        name=name,
        normalization=normalization,
        visibility=visibility,
        aggregation=aggregation,
    )


def configuration():
    return dict(
        rounding="ceil",
        include_inspection=True,
        context=[
            component("decision", "projection", "actor"),
            component("self_target", visibility="actor"),
            component("local_map", visibility="actor"),
            component("vehicles", "projection"),
            component("machines", "projection"),
            component("inventory", "capacity"),
            component("topology"),
            component("work_summary", "projection", "critic"),
        ],
        candidates=[
            component(name, normalization, "actor")
            for name, normalization in (
                ("job", "projection"),
                ("semantics", "projection"),
                ("machine_mode", "projection"),
                ("dispatch_target", "capacity"),
                ("cargo", "projection"),
                ("movement", "projection"),
            )
        ],
    )


def pair(settings=None):
    return pin_contract(
        ExtensionRef(name=OBSERVATION, parameters=settings or configuration()),
        ExtensionRef(name=NETWORK),
    )


def instance(factory, settings=None, *, algorithm="ppo"):
    observation, network = pair(settings)
    metadata = dict(
        observation=primitive(observation),
        network_implementation=primitive(network),
        role="dispatcher",
        algorithm=algorithm,
        provider="rllib.resource_" + algorithm,
        projection=dict(time_scale=100, count_scale=100),
        physical_contract=dict(processing_rounding="ceil"),
    )
    return encoder(factory, metadata), metadata


@pytest.fixture
def simulation():
    return ProductionSimulator(
        replace(small_scenario(), processing_rounding="ceil"), contract="v3"
    )


def dispatch(sim):
    return next(
        request for request in sim.protocol.begin() if request.role == "dispatcher"
    )


def context_part(encoded, enc, name, *, critic=False):
    layout = next(
        row
        for row in enc.layout
        if row["component"]["name"] == name
        and row["branch"] == ("critic" if critic else "actor")
    )
    start = layout["offset"]
    return encoded["context"][start : start + layout["width"]]


def candidate_part(encoded, enc, name, *, prefix=False):
    start = 0
    for selection in enc.settings.candidates:
        width = enc.widths[selection.name]
        if selection.name == name:
            return encoded["prefix" if prefix else "candidates"][
                :, start : start + width
            ]
        start += width
    raise AssertionError(name)


def test_versioned_pair_rejects_unregistered_or_executable_yaml():
    observation, network = pair()
    assert pin_contract(observation, network) == (observation, network)
    for name in ("announced", "reserved", "intention_counts", "python.module", "eval"):
        settings = configuration()
        settings["context"][0]["name"] = name
        with pytest.raises(ValueError, match="unregistered"):
            pair(settings)
    settings = configuration()
    settings["context"][0]["expression"] = "__import__('os')"
    with pytest.raises(ValueError):
        pair(settings)
    with pytest.raises(ValueError, match="matching"):
        pin_contract(observation, None)
    with pytest.raises(ValueError, match="digest"):
        pin_contract(observation.model_copy(update={"code_sha256": "0" * 64}), network)


def test_legacy_digest_and_default_remain_readable():
    from smartsom.config.policies import PolicyExtensions
    from smartsom.config.policy_contracts import implementation_digest

    assert primitive(PolicyExtensions()) == {"observation": None, "network": None}
    assert (
        implementation_digest()
        == "eb382961bf683559a80235d27854a6573248f233a43a0d1cfa5f98e0cba89beb"
    )
    observation = ExtensionRef(
        name="builtin.physical_job",
        version="1",
        parameters={"include_inspection": True},
        code_sha256=implementation_digest(),
    )
    network = ExtensionRef(
        name="builtin.physical_job_candidate",
        version="1",
        code_sha256=implementation_digest(),
    )
    assert pin_contract(observation, network) == (observation, network)


def test_restricted_components_cannot_access_wide_snapshot(simulation, monkeypatch):
    import smartsom.learning.registered_observation as module

    request = dispatch(simulation)
    poisoned = copy.deepcopy(request.observation)
    poisoned.update(
        announced=[{"due_at": -999}], metrics={"private": 9}, return_value=42
    )
    poisoned["return"] = 1000
    poisoned["jobs"][next(iter(poisoned["jobs"]))]["defective"] = True
    poisoned["machines"]["machine"]["remaining"] = 999
    clean = restricted_view(poisoned, "agv")
    assert not {"announced", "metrics", "return", "return_value"} & clean.keys()
    assert "remaining" not in clean["machines"]["machine"]
    assert "defective" not in next(iter(clean["jobs"].values()))
    assert all("reserved" not in row for row in clean["sources"].values())
    original = module.CONTEXT_ENCODERS["decision"]

    def inspect(enc, current, view, selection):
        assert current.observation is view
        assert "metrics" not in current.observation
        assert all(candidate.features[6] == 0 for candidate in current.candidates)
        return original(enc, current, view, selection)

    monkeypatch.setitem(module.CONTEXT_ENCODERS, "decision", inspect)
    enc, _ = instance(simulation.factory)
    baseline = enc.encode(request)
    altered = enc.encode(replace(request, observation=poisoned))
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], altered[key])
    assert "defective" not in next(iter(request.observation["jobs"].values()))


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("training", [False, True])
def test_location_projection_refreshes_within_same_tick(
    simulation, algorithm, training
):
    request = dispatch(simulation)
    enc, _ = instance(simulation.factory, algorithm=algorithm)
    enc.set_training(training)
    original = enc.encode(request)
    view = copy.deepcopy(request.observation)
    job = next(iter(view["jobs"]))
    # Move resident inventory without changing tick/stage. No cross-request
    # index or filtered-view cache may retain the previous location.
    view["jobs"][job]["location"] = "machine"
    changed = enc.encode(replace(request, observation=view))
    assert not np.array_equal(original["context"], changed["context"])
    restored = enc.encode(request)
    for name in original:
        np.testing.assert_array_equal(original[name], restored[name])


def test_route_projection_is_detached_and_filters_before_copy(simulation):
    request = dispatch(simulation)
    view = copy.deepcopy(request.observation)
    job = next(iter(view["jobs"]))

    class PrivateField:
        def __deepcopy__(self, memo):
            raise AssertionError("discarded data must not be copied")

    for name in ("original_steps", "remaining_steps"):
        view["jobs"][job][name][0]["private_extension"] = PrivateField()
    shipped = copy.deepcopy(request.observation["jobs"][job])
    shipped.update(location="output", original_steps=PrivateField())
    view["jobs"]["shipped"] = shipped
    enc, _ = instance(simulation.factory)
    clean = restricted_view(view, request.owner, physical=enc.physical)
    assert "shipped" not in clean["jobs"]
    assert "private_extension" not in clean["jobs"][job]["original_steps"][0]
    clean["jobs"][job]["original_steps"][0]["nominal_ticks"] = -1
    assert view["jobs"][job]["original_steps"][0]["nominal_ticks"] > 0
    # Early physical filtering must retain the unreleased-job guard.
    shipped["demand"] = "unreleased"
    with pytest.raises(ValueError, match="unreleased"):
        restricted_view(view, request.owner, physical=enc.physical)


def test_both_intention_count_paths_are_removed(simulation):
    request = dispatch(simulation)
    enc, _ = instance(simulation.factory)
    original = enc.encode(request)
    view = copy.deepcopy(request.observation)
    for row in view["sources"].values():
        row["reserved"] = 10**6
    candidates = tuple(
        replace(
            candidate, features=(*candidate.features[:6], 1e6, *candidate.features[7:])
        )
        for candidate in request.candidates
    )
    changed = enc.encode(replace(request, observation=view, candidates=candidates))
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])


def test_own_target_is_preserved(simulation):
    request = dispatch(simulation)
    enc, _ = instance(simulation.factory)
    original = enc.encode(request)
    view = copy.deepcopy(request.observation)
    target = request.candidates[0].action
    view["agvs"]["agv"]["target"] = dict(owner=target.owner, port=target.port)
    changed = enc.encode(replace(request, observation=view))
    assert not np.array_equal(
        context_part(original, enc, "self_target"),
        context_part(changed, enc, "self_target"),
    )


def test_other_agv_intentions_and_attempt_are_absent_in_every_plane_and_prefix():
    from smartsom.domain.factory_design import AGVDesign, Cell

    case = small_scenario()
    factory = replace(
        case.factory,
        agvs=case.factory.agvs + (AGVDesign("other", "Other", Cell(1, 0)),),
    )
    sim = ProductionSimulator(
        replace(case, factory=factory, processing_rounding="ceil"), contract="v3"
    )
    request = dispatch(sim)
    enc, metadata = instance(factory)
    view = copy.deepcopy(request.observation)
    job = next(iter(view["jobs"]))
    buffer = replace(
        request,
        role="buffer",
        owner="input",
        observation=view,
        candidates=(Candidate(job, job, (0.0,) * 12),),
        prefix=(job,),
    )
    baseline = [enc.encode(request), enc.encode(buffer)]
    view["jobs"][job]["attempt"] = 10**6
    view["agvs"]["other"]["target"] = dict(owner="input", port="in_port")
    view["agvs"]["other"]["reservation"] = "input"
    view["agvs"]["other"]["reservation_tick"] = -1000
    view["sources"]["input"]["reserved"] = 10**6
    clean = restricted_view(view, "agv")
    assert "target" not in clean["agvs"]["other"]
    assert "has_target" not in clean["agvs"]["other"]
    assert "attempt" not in clean["jobs"][job]
    changed = [
        enc.encode(replace(request, observation=view)),
        enc.encode(replace(buffer, observation=view)),
    ]
    net = network_class(metadata["network_implementation"])(
        enc.context_size,
        default_network("ppo"),
        "rllib.resource_ppo",
        candidate_width=enc.candidate_width,
        context_layout=enc.context_layout,
    )
    for original, altered in zip(baseline, changed, strict=True):
        for key in original:
            np.testing.assert_array_equal(original[key], altered[key])
        with torch.no_grad():
            before = net(tensor_inputs([original]))
            after = net(tensor_inputs([altered]))
        for left, right in zip(before, after, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Physical resource information is still available to both branches.
    physical = copy.deepcopy(view)
    physical["agvs"]["other"].update(cell=[2, 0], job=job, service={"kind": "pickup"})
    encoded = enc.encode(replace(request, observation=physical))
    assert not np.array_equal(
        context_part(encoded, enc, "vehicles"),
        context_part(changed[0], enc, "vehicles"),
    )


@pytest.mark.parametrize("quality,expected", [("UNKNOWN", [1, 0]), ("PASS", [1, 1])])
def test_quality_presence_and_pass_for_job_prefix_and_cargo(
    simulation, quality, expected
):
    request = dispatch(simulation)
    job = next(iter(request.observation["jobs"]))
    view = copy.deepcopy(request.observation)
    view["jobs"][job]["quality"] = quality
    candidate = Candidate(job, job, (0.0,) * 12)
    buffer = replace(
        request,
        role="buffer",
        owner="input",
        candidates=(candidate,),
        prefix=(job,),
        observation=view,
    )
    enc, _ = instance(simulation.factory)
    encoded = enc.encode(buffer)
    assert candidate_part(encoded, enc, "job")[0, 5:7].tolist() == expected
    assert candidate_part(encoded, enc, "job", prefix=True)[0, 5:7].tolist() == expected
    view["jobs"][job]["location"] = "agv"
    view["agvs"]["agv"]["job"] = job
    cargo = enc.encode(replace(request, observation=view))
    assert candidate_part(cargo, enc, "cargo")[0, 5:7].tolist() == expected
    empty = enc.encode(request)
    assert candidate_part(empty, enc, "cargo")[0, 5:7].tolist() == [0, 0]


def test_disposing_keeps_occupancy_but_removes_work_and_actions():
    sim = ProductionSimulator(automatic_inspection_case(count=1))
    job = next(iter(sim.jobs))
    demand = sim.jobs[job]["demand"]
    due_at = sim.demands[demand].due_at
    release_at = sim.demands[demand].release_at
    carry_to_inspection(sim, job, defective=True)
    act(sim)
    act(sim)
    assert sim.station_state["inspection"]["jobs"][job]["status"] == "DISPOSING"
    protocol = ProductionProtocol(sim)
    view = protocol.public_view()
    enc, _ = instance(sim.factory)
    request = DecisionRequest(sim.tick, "value", "dispatcher", "agv", (), view)
    encoded = enc.encode(request)
    block = context_part(encoded, enc, "inventory").reshape(len(enc.owners), -1)[
        enc.owners.index("inspection")
    ]
    assert block[0] == pytest.approx(1 / 4)
    assert block[7] == pytest.approx(1 / 4)  # physical FAIL remains explicit
    assert block[8] == pytest.approx(1 / 4)
    np.testing.assert_array_equal(
        context_part(encoded, enc, "work_summary", critic=True), np.zeros(8)
    )
    assert sim.selectable("inspection") == []
    with pytest.raises(ValueError, match="FAIL/DISPOSING"):
        enc.encode(
            replace(
                request,
                role="buffer",
                owner="inspection",
                candidates=(Candidate(job, job, (0.0,) * 12),),
            )
        )
    act(sim)
    assert sim.jobs[job]["location"] == "scrap"
    assert sim.metrics["pre_output_scrap"] == 1
    replacement = next(
        key for key, row in sim.jobs.items() if row["demand"] == demand and key != job
    )
    assert sim.attempts[demand] == 2
    replacement_view = protocol.public_view()["jobs"][replacement]
    assert replacement_view["due_at"] == due_at
    assert sim.demands[demand].release_at == release_at


def test_disposal_one_tick_capacity_masks_and_same_tick_arrival_order():
    from smartsom.domain.factory_design import Cell, SlotDesign

    case = automatic_inspection_case(count=2)
    station = replace(
        case.factory.inspection_stations[0], slots=(SlotDesign("slot", Cell(0, 0), 1),)
    )
    case = replace(case, factory=replace(case.factory, inspection_stations=(station,)))
    sim = ProductionSimulator(case)
    first, second = sorted(sim.jobs)
    carry_to_inspection(sim, first, defective=True)
    act(sim)
    assert sim.tick == 2 and sim.jobs[first]["quality"] == "UNKNOWN"
    assert sim._free("inspection") is None
    act(sim)
    assert sim.tick == 3
    assert sim.station_state["inspection"]["jobs"][first] == {
        "status": "DISPOSING",
        "remaining": 1,
        "total": 1,
    }
    assert sim._free("inspection") is None
    assert first not in sim.selectable("inspection")
    protocol = ProductionProtocol(sim)
    assert first not in protocol.ready("inspection")
    assert protocol.source_supply("inspection") == 0
    enc, _ = instance(sim.factory)
    view = protocol.public_view()
    for role, owner in (
        ("machine", "machine"),
        ("buffer", "inspection"),
        ("dispatcher", "agv"),
        ("mover", "agv"),
    ):
        value = enc.encode(DecisionRequest(3, "value", role, owner, (), view))
        inventory = context_part(value, enc, "inventory").reshape(len(enc.owners), -1)[
            enc.owners.index("inspection")
        ]
        assert inventory[[0, 7, 8]].tolist() == [1, 1, 1]
        assert inventory[[1, 2]].tolist() == [0, 0]
        assert value["mask"].shape == (0,)
    for role, action, prefix in (
        ("machine", (first, "normal"), ()),
        ("buffer", first, ()),
        ("buffer", first, (first,)),
    ):
        with pytest.raises(ValueError, match="FAIL/DISPOSING"):
            enc.encode(
                DecisionRequest(
                    3,
                    "proposals",
                    role,
                    "machine" if role == "machine" else "inspection",
                    (Candidate(first, action, (0.0,) * 12),),
                    view,
                    prefix,
                )
            )
    illegal_cargo = copy.deepcopy(view)
    illegal_cargo["agvs"]["agv"]["job"] = first
    with pytest.raises(ValueError, match="FAIL/DISPOSING"):
        enc.encode(
            replace(
                protocol.mover_request("agv"),
                role="dispatcher",
                candidates=tuple(protocol.dispatch_candidates("agv")),
                observation=illegal_cargo,
            )
        )
    # Commands see the full capacity at boundary 3. Disposal completes in the
    # following transition; it cannot make this boundary's drop retroactively legal.
    row = carry_to_inspection(sim, second)
    assert sim.tick == 4 and row["rejections"] == {"agv:agv": "invalid"}
    assert sim.agvs["agv"]["job"] == second
    assert sim.jobs[first]["location"] == "scrap"
    assert sim._free("inspection") == "slot"
    assert sim.metrics["pre_output_scrap"] == 1
    row = act(sim, "INTERACT")
    assert sim.tick == 5 and row["rejections"] == {}
    assert sim.jobs[second]["location"] == "inspection"
    assert sim._free("inspection") is None
    assert sim.station_state["inspection"]["jobs"][second]["remaining"] == 2


def test_unlinked_fail_and_rounding_mismatch_fail_closed(simulation):
    case = automatic_inspection_case(associated=False)
    observation, _ = pair()
    with pytest.raises(ValueError, match="automatic disposal"):
        validate_factory(case.factory, observation)
    enc, metadata = instance(simulation.factory)
    metadata["physical_contract"]["processing_rounding"] = "half_up"
    with pytest.raises(ValueError, match="rounding"):
        encoder(simulation.factory, metadata)
    state = enc.state_dict()
    state["candidate_width"] += 1
    with pytest.raises(ValueError, match="identity/shape"):
        enc.load_state_dict(state)


@pytest.mark.parametrize("capacity,ratio,unlimited", [(2, 0.5, 0), (None, 0.01, 1)])
def test_finite_and_unlimited_capacity_normalization(capacity, ratio, unlimited):
    case = small_scenario()
    factory = replace(
        case.factory,
        buffers=tuple(
            replace(buffer, storage=replace(buffer.storage, capacity=capacity))
            if buffer.role == "system_input"
            else buffer
            for buffer in case.factory.buffers
        ),
    )
    sim = ProductionSimulator(
        replace(case, factory=factory, processing_rounding="ceil"), contract="v3"
    )
    request = dispatch(sim)
    enc, _ = instance(factory)
    encoded = enc.encode(request)
    target = candidate_part(encoded, enc, "dispatch_target")[0]
    assert target[5:7].tolist() == pytest.approx([ratio, ratio])
    assert target[8] == unlimited


def test_output_history_and_empty_machine_timing_do_not_change_inputs(simulation):
    request = dispatch(simulation)
    enc, _ = instance(simulation.factory)
    original = enc.encode(request)
    view = copy.deepcopy(request.observation)
    job = next(iter(view["jobs"]))
    view["jobs"]["shipped-copy"] = {**view["jobs"][job], "location": "output"}
    view["storage"]["output"] = {"pool": ["shipped-copy"] * 100}
    view["machines"]["machine"].update(nominal=999, elapsed=998)
    view["completed"] = ["old"] * 100
    changed = enc.encode(replace(request, observation=view))
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])


def test_loaded_dispatch_output_candidate_has_no_cumulative_occupancy(simulation):
    job = next(iter(simulation.jobs))
    simulation._remove(job)
    simulation.jobs[job].update(location="agv", step=1)
    simulation.agvs["agv"]["job"] = job
    request = dispatch(simulation)
    assert all(candidate.action.owner == "output" for candidate in request.candidates)
    enc, _ = instance(simulation.factory)
    baseline = enc.encode(request)
    view = copy.deepcopy(request.observation)
    view["jobs"]["delivered"] = {**view["jobs"][job], "location": "output"}
    view["storage"]["output"] = {"pool": ["delivered"] * 1000}
    candidates = tuple(
        replace(
            candidate, features=(*candidate.features[:7], 10.0, *candidate.features[8:])
        )
        for candidate in request.candidates
    )
    altered = enc.encode(replace(request, observation=view, candidates=candidates))
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], altered[key])
    output = candidate_part(altered, enc, "dispatch_target")[0]
    assert output[5:7].tolist() == [0, 0]
    assert output[11] == 0


@pytest.mark.parametrize("rounding,expected", [("ceil", 2), ("half_up", 1)])
def test_exact_nominal_rate_mode_reference_and_override(rounding, expected):
    machine = SimpleNamespace(machine_id="m", processing_rate_multiplier="2")
    mode = SimpleNamespace(time_scale="3")
    step = dict(nominal_ticks=9, reference_ticks="4/5", machine_nominal_ticks={})
    assert nominal_ticks(step, machine, mode, rounding) == expected
    step["machine_nominal_ticks"] = {"m": 3}
    assert nominal_ticks(step, machine, mode, "ceil") == 5


def test_candidate_nominal_matches_core_and_hidden_samples_do_not_enter(simulation):
    job = next(iter(simulation.jobs))
    simulation._remove(job)
    simulation.jobs[job]["location"] = "machine"
    simulation.machine_state["machine"].update(job=job, status="READY")
    request = next(
        request for request in simulation.protocol.begin() if request.role == "machine"
    )
    enc, _ = instance(simulation.factory)
    encoded = enc.encode(request)
    duration = candidate_part(encoded, enc, "machine_mode")[0, 0] * 100
    assert duration == pytest.approx(2)
    simulation._start("machine", MachineCommand(job, "normal"))
    assert simulation.machine_state["machine"]["nominal"] == duration
    view = copy.deepcopy(request.observation)
    view["jobs"][job]["defective"] = True
    view["machines"]["machine"]["remaining"] = 999
    changed = enc.encode(replace(request, observation=view))
    for key in encoded:
        np.testing.assert_array_equal(encoded[key], changed[key])


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_scoped_network_actor_q_critic_and_prefix_privacy(simulation, algorithm):
    request = dispatch(simulation)
    enc, metadata = instance(simulation.factory, algorithm=algorithm)
    net = network_class(metadata["network_implementation"])(
        enc.context_size,
        default_network(algorithm),
        "rllib.resource_" + algorithm,
        algorithm,
        candidate_width=enc.candidate_width,
        context_layout=enc.context_layout,
    )
    inputs = tensor_inputs([enc.encode(request)])
    modified = {key: value.clone() for key, value in inputs.items()}
    modified["context"][:, enc.context_layout["actor"] :] += 3
    with torch.no_grad():
        scores, values = net(inputs)
        altered_scores, altered_values = net(modified)
    torch.testing.assert_close(scores, altered_scores, rtol=0, atol=0)
    if algorithm == "ppo":
        assert not torch.equal(values, altered_values)
        modified = {key: value.clone() for key, value in inputs.items()}
        modified["candidates"] += 3
        modified["context"][:, : enc.context_layout["actor"]] += 3
        modified["prefix"] = torch.ones((1, 2, enc.candidate_width))
        modified["prefix_length"] = torch.tensor([2])
        with torch.no_grad():
            _, altered_values = net(modified)
        torch.testing.assert_close(values, altered_values, rtol=0, atol=0)
    else:
        torch.testing.assert_close(values, altered_values, rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["mean", "max", "sum"])
def test_registered_aggregation_and_identity(simulation, aggregation):
    settings = configuration()
    settings["context"] = [component("vehicles", "projection", aggregation=aggregation)]
    enc, _ = instance(simulation.factory, settings)
    assert enc.context_size == 8
    request = dispatch(simulation)
    assert enc.encode(request)["context"].tolist() == [0, 0, 0, 0] * 2
    other = configuration()
    other["context"] = [component("vehicles", "raw", aggregation=aggregation)]
    changed, _ = instance(simulation.factory, other)
    with pytest.raises(ValueError, match="identity/shape"):
        changed.load_state_dict(enc.state_dict())


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_ordinary_yaml_compile_inference_and_package_roundtrip(tmp_path, algorithm):
    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.composable import package
    from smartsom.learning.observation_factory import validate_effective_contract
    from smartsom.learning.production_inference import build_groups, read_package

    root = Path(__file__).resolve().parents[2]
    factory = yaml.safe_load(
        (root / "configs/test/factories/factory_hand.yaml").read_text()
    )
    (tmp_path / "factory.yaml").write_text(yaml.safe_dump(factory))
    workload = dict(
        schema="smartsom.workload/v3",
        demands=[
            dict(
                demand_id="one",
                steps=[
                    dict(
                        operation_id="op", operation_type="operation_1", nominal_ticks=2
                    )
                ],
                due_at=30,
            )
        ],
    )
    (tmp_path / "workload.yaml").write_text(yaml.safe_dump(workload))
    observation, implementation = pair()
    policy = dict(
        kind="new_model",
        extensions=dict(
            observation=primitive(observation),
            network_implementation=primitive(implementation),
            network={
                branch: {"hidden_sizes": [8]}
                for branch in (("actor", "critic") if algorithm == "ppo" else ("q",))
            },
        ),
    )
    method = dict(
        schema="smartsom.algorithm/v2",
        mode="resource",
        learner=dict(
            backend="rllib",
            algorithm=algorithm,
            parameters={
                "batch_size": 2,
                **({"warmup_ticks": 0} if algorithm == "dqn" else {"n_epochs": 1}),
            },
        ),
        agents={
            role: {"default": policy} for role in ("machine", "buffer", "dispatcher")
        },
    )
    method["agents"]["mover"] = dict(default=dict(kind="rule", name="shortest_path"))
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(method))
    experiment = dict(
        schema="smartsom.experiment-config/v4",
        task="train",
        factory="factory.yaml",
        workload="workload.yaml",
        algorithm="algorithm.yaml",
        training=dict(total_ticks=8, ticks_per_update=4, max_ticks=8),
        runtime=dict(
            environment=dict(mode="finite", tick_limit=8, processing_rounding="ceil")
        ),
        execution=dict(tuning="off", calibration_level="off"),
        validation=dict(enabled=False),
        output=dict(root=str(tmp_path / "runs"), name="fields"),
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    prepared = compile_experiment(path).entries[0].prepared
    assert not (tmp_path / "runs").exists()
    policies, learners = build_groups(prepared, training=False)
    assert learners == {}
    policy = policies["dispatcher"]
    assert type(policy.encoder) is PublicFieldsEncoder
    validate_effective_contract(policy.metadata)
    sim = ProductionSimulator(prepared.scenario, contract="v3")
    request = next(
        request for request in sim.protocol.begin() if request.role == "dispatcher"
    )
    before = policy.choose(request)
    package(tmp_path / "model", policy, prepared, "dispatcher", 0, partners={})
    metadata, weights, state = read_package(tmp_path / "model")
    assert metadata["physical_job_encoder"] == policy.encoder.observation_identity
    assert state == policy.encoder.state_dict()
    # Public batch preflight validates already initialized packages through the
    # compiler's dependency-enabled reconstruction, not only build_groups.
    initialized = copy.deepcopy(method)
    initialized["agents"]["dispatcher"]["default"] = dict(
        kind="model", model=dict(source=str(tmp_path / "model"))
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(initialized))
    recompiled = compile_experiment(path, require_dependencies=True).entries[0].prepared
    reloaded, _ = build_groups(recompiled, training=False)
    assert (
        reloaded["dispatcher"].network.actor_context_size
        == policy.encoder.context_layout["actor"]
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(method))
    policy.network.load_state_dict(weights, strict=True)
    assert policy.choose(request).action == before.action
    forged = copy.deepcopy(state)
    forged["candidate_width"] += 1
    with pytest.raises(ValueError, match="identity/shape"):
        policy.encoder.load_state_dict(forged)
    from smartsom.experiments.composable import TrainingSession, allocate

    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    try:
        session.step_update()
        assert session.ticks == 4
        saved = session.state_dict()
    finally:
        session.close()
    root, record, frozen = allocate(prepared, "training")
    restored = TrainingSession(frozen, root, record)
    try:
        restored.restore(saved)
        restored.step_update()
        assert restored.ticks == 8
        assert sum(restored.optimizations.values()) > 0
        for group in restored.settings.groups:
            model = restored.policies[group]
            assert (
                model.network.actor_context_size
                == model.encoder.context_layout["actor"]
            )
    finally:
        restored.close()
    experiment["runtime"]["environment"]["processing_rounding"] = "half_up"
    path.write_text(yaml.safe_dump(experiment))
    with pytest.raises(ValueError, match="rounding"):
        compile_experiment(path)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_compact_context_has_no_cross_branch_padding(
    simulation, algorithm, monkeypatch
):
    from smartsom.learning.registered_observation import CONTEXT_ENCODERS

    enc, metadata = instance(simulation.factory, algorithm=algorithm)
    if algorithm == "dqn":

        def forbidden(*args):
            raise AssertionError("DQN computed a critic-only field")

        monkeypatch.setitem(CONTEXT_ENCODERS, "work_summary", forbidden)
    encoded = enc.encode(dispatch(simulation))
    expected_actor = sum(
        enc._width(field)
        for field in enc.settings.context
        if field.visibility in ("actor", "both")
    )
    expected_critic = (
        sum(
            enc._width(field)
            for field in enc.settings.context
            if field.visibility in ("critic", "both")
        )
        if algorithm == "ppo"
        else 0
    )
    assert enc.context_layout == {"actor": expected_actor, "critic": expected_critic}
    assert encoded["context"].shape == (expected_actor + expected_critic,)
    net = network_class(metadata["network_implementation"])(
        enc.context_size,
        default_network(algorithm),
        metadata["provider"],
        algorithm,
        context_layout=enc.context_layout,
        candidate_width=enc.candidate_width,
    )
    assert net.actor.hidden[0].in_features == expected_actor + 32 + enc.candidate_width
    if algorithm == "ppo":
        assert net.critic.hidden[0].in_features == expected_critic
        assert not hasattr(net, "critic_prefix")
    else:
        assert net.critic is None
    old = {**enc.state_dict(), "public_fields": "smartsom.public-fields/v1/old"}
    with pytest.raises(ValueError, match="identity/shape"):
        enc.load_state_dict(old)


def test_public_fields_is_single_current_interface():
    observation, network = pair()
    assert observation.version == network.version == "compact"
    for version in ("1", "2"):
        with pytest.raises(ValueError, match="remove legacy version"):
            pin_contract(
                observation.model_copy(update={"version": version}),
                network.model_copy(update={"version": version}),
            )


@pytest.mark.parametrize(
    "role,owner,presence",
    [
        ("machine", "machine", 0),
        ("buffer", "input", 0),
        ("dispatcher", "agv", 1),
        ("mover", "agv", 1),
    ],
)
def test_local_map_role_applicability_presence_and_stable_shapes(
    simulation, role, owner, presence
):
    view = simulation.protocol.public_view()
    view["agvs"]["agv"]["cell"] = [2, 0]
    request = DecisionRequest(0, "value", role, owner, (), view)
    enc, _ = instance(simulation.factory)
    encoded = enc.encode(request)
    local = context_part(encoded, enc, "local_map")
    assert local.shape == (51,)
    assert local[0] == presence
    if not presence:
        np.testing.assert_array_equal(local, np.zeros(51))
    else:
        # The top-right outside cell follows the owner at x=2, not origin x=0.
        assert local[1 + 2 * 14] == 1
    assert encoded["context"].shape == (enc.context_size,)
    assert encoded["candidates"].shape == (0, enc.candidate_width)
    assert encoded["prefix"].shape == (0, enc.candidate_width)


def test_same_shape_different_applicability_rejects_identity_and_state_alias(
    simulation,
):
    settings = configuration()
    settings["context"][2]["applies_to"] = ["dispatcher", "mover"]
    enc, _ = instance(simulation.factory, settings)
    changed = copy.deepcopy(settings)
    changed["context"][2]["applies_to"] = ["mover"]
    other, _ = instance(simulation.factory, changed)
    assert (enc.context_size, enc.candidate_width) == (
        other.context_size,
        other.candidate_width,
    )
    assert enc.observation_identity != other.observation_identity
    with pytest.raises(ValueError, match="identity/shape"):
        other.load_state_dict(enc.state_dict())
    copied = enc.state_dict()
    copied["layout"][0]["width"] += 1
    with pytest.raises(ValueError, match="identity/shape"):
        enc.load_state_dict(copied)
    changed["context"][2]["applies_to"] = ["machine"]
    with pytest.raises(ValueError, match="only to Dispatcher/Mover"):
        pair(changed)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("hook_kind", ["forward", "pre"])
def test_value_input_purity_rejects_global_hooks(
    simulation, monkeypatch, algorithm, hook_kind
):
    from smartsom.learning.production_collection import PhysicalCollector
    from smartsom.learning.production_inference import ModelPolicy

    request = dispatch(simulation)
    enc, metadata = instance(simulation.factory, algorithm=algorithm)
    net = network_class(metadata["network_implementation"])(
        enc.context_size,
        default_network(algorithm),
        "rllib.resource_" + algorithm,
        algorithm,
        candidate_width=enc.candidate_width,
        context_layout=enc.context_layout,
    )
    policy = ModelPolicy(enc, net, metadata, 101, training=True)
    assert policy.stateless_value_input

    inputs = tensor_inputs([enc.encode(request)])
    calls = []

    def global_hook(module, *arguments):
        calls.append(module)
        torch.rand(1)
        if hook_kind == "forward" and module is net:
            scores, values = arguments[-1]
            return scores + 1, values + 1
        return None

    register = (
        torch.nn.modules.module.register_module_forward_hook
        if hook_kind == "forward"
        else torch.nn.modules.module.register_module_forward_pre_hook
    )
    with torch.random.fork_rng(), torch.no_grad():
        expected = net(inputs)
        handle = register(global_hook)
        try:
            assert not policy.stateless_value_input
            rng = torch.get_rng_state().clone()
            actual = net(inputs)
            assert calls and not torch.equal(rng, torch.get_rng_state())
            if hook_kind == "forward":
                for before, after in zip(expected, actual, strict=True):
                    torch.testing.assert_close(after, before + 1, rtol=0, atol=0)
            value_calls = []
            value_input = policy.value_input

            def observed_value(*args):
                value_calls.append(args)
                return value_input(*args)

            monkeypatch.setattr(policy, "value_input", observed_value)
            collector = PhysicalCollector(["dispatcher"], 0.99)
            collector.active[(0, 0, request.owner, "dispatcher")] = request.role
            collector.collect(
                0,
                0,
                SimpleNamespace(records=[], sim=simulation),
                {},
                {"dispatcher": policy},
                {"dispatcher": 0.0},
                algorithm=algorithm,
                before=simulation.protocol.public_view(),
            )
            assert len(value_calls) == 2  # Both original eager calls survive.
        finally:
            handle.remove()
    assert policy.stateless_value_input
    local = net.register_forward_hook(lambda *_: None)
    try:
        assert not policy.stateless_value_input
    finally:
        local.remove()
    assert policy.stateless_value_input
    original_encoder = net.actor.encoder
    try:
        net.actor.encoder = torch.nn.Identity()  # Unapproved nested custom graph.
        assert not policy.stateless_value_input
    finally:
        net.actor.encoder = original_encoder
    assert policy.stateless_value_input


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("boundary", ["rollout", "terminal", "truncated", "censored"])
def test_lazy_value_collection_matches_eager_rows_and_midrollout_restore(
    algorithm, boundary
):
    from smartsom.learning.production_collection import PhysicalCollector

    class Policy:
        def __init__(self, pure):
            self.stateless_value_input = pure
            self.calls = []

        def value_input(self, role, owner, view):
            self.calls.append(view["tick"])
            return {"context": [float(view["tick"])]}, float(view["tick"]) / 7

    def exercise(pure, resume=False):
        policy = Policy(pure)
        collector = PhysicalCollector(["dispatcher"], 0.99)
        view = {"tick": 0}
        coordinator = SimpleNamespace(
            records=[],
            sim=SimpleNamespace(
                protocol=SimpleNamespace(public_view=lambda: copy.deepcopy(view))
            ),
        )
        for tick in range(1, 6):
            view["tick"] = tick
            coordinator.records = (
                [
                    dict(
                        group="dispatcher",
                        owner="vehicle",
                        role="dispatcher",
                        actor_mask=True,
                        model_input={"context": [tick]},
                        candidates=[{"identity": "wait"}],
                        candidate="wait",
                        log_probability=-0.1,
                    )
                ]
                if tick in (1, 3)
                else []
            )
            collector.collect(
                0,
                0,
                coordinator,
                {},
                {"dispatcher": policy},
                {"dispatcher": tick / 10},
                algorithm=algorithm,
                before={"tick": tick - 1},
                terminated=boundary == "terminal" and tick == 5,
                truncated=boundary in ("truncated", "censored") and tick == 5,
                bootstrap_inputs=(
                    {("vehicle", "dispatcher"): {"context": [6]}}
                    if boundary == "truncated"
                    else {}
                ),
            )
            if resume and tick == 4:
                restored = PhysicalCollector(["dispatcher"], 0.99)
                restored.load_state_dict(collector.state_dict())
                collector = restored
        rows = collector.drain(algorithm, policies={"dispatcher": policy})
        if algorithm == "ppo":
            # Only the final bootstrap is consumed by physical GAE.
            for row in rows["dispatcher"][:-1]:
                row.pop("bootstrap")
        return rows, collector.counts, policy.calls

    eager = exercise(False)
    lazy = exercise(True)
    assert eager[:2] == lazy[:2]
    assert lazy == exercise(True, resume=True)
    assert len(eager[2]) == 10
    expected = (
        (1 if boundary == "terminal" else 0)
        if algorithm == "dqn"
        else (5 if boundary == "terminal" else 6)
    )
    assert len(lazy[2]) == expected
