"""Opt-in policy-side dispatcher masks, target machine features and per-role modes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.domain.production_decisions import DispatchTarget
from smartsom.engine.production import ProductionSimulator

torch = pytest.importorskip("torch")

from smartsom.learning.production_models import (  # noqa: E402
    MACHINE_FEATURES,
    CandidateNetwork,
    PublicEncoder,
    default_network,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def started():
    prepared = api.prepare(
        api.load_config(ROOT / "configs/test/runs/template1_warmup.yaml"),
        training=False,
    )
    sim = ProductionSimulator(prepared.resolved.scenario, contract="v3")
    requests = sim.protocol.begin()
    request = next(r for r in requests if r.role == "dispatcher")
    yield sim, request
    sim.protocol.abort()


def encoder(sim, **options):
    return PublicEncoder(
        sim.factory,
        dict(time_scale=100, count_scale=100),
        role="central",
        network={**default_network("ppo"), **options},
    )


def with_view(request, edit):
    view = deepcopy(request.observation)
    edit(view)
    return replace(request, observation=view)


def supplied(request):
    sources = request.observation["sources"]
    return [
        max(
            len(sources.get(c.action.owner, {}).get("ready", [])),
            sources.get(c.action.owner, {}).get("supply", 0),
        )
        > 0
        for c in request.candidates
    ]


def test_default_encoder_keeps_the_legal_mask_and_width(started):
    sim, request = started
    plain = encoder(sim)
    encoded = plain.encode(request)
    assert list(encoded["mask"]) == [c.legal for c in request.candidates]
    assert encoder(sim, target_machine_features=True).candidate_width == (
        plain.candidate_width + MACHINE_FEATURES
    )


def test_supply_mask_keeps_only_sources_with_supply(started):
    sim, request = started
    expected = supplied(request)
    assert any(expected) and not all(expected)
    mask = encoder(sim, dispatch_supply_mask=True).encode(request)["mask"]
    assert list(mask) == expected


def test_supply_mask_excludes_supply_already_claimed_by_other_vehicles(started):
    sim, request = started
    sources = request.observation["sources"]
    owners = sorted(
        {c.action.owner for c in request.candidates if c.action.owner in sources}
    )
    claimed_owner, open_owner = owners[:2]
    other = next(k for k in sorted(request.observation["agvs"]) if k != request.owner)

    def claim(view):
        for owner, source in view["sources"].items():
            source.update(ready=[], supply=1 if owner in owners[:2] else 0)
        view["agvs"][other].update(
            job=None, target={"owner": claimed_owner, "port": "p"}
        )

    mask = encoder(sim, dispatch_supply_mask=True).encode(with_view(request, claim))[
        "mask"
    ]
    kept = {c.action.owner for m, c in zip(mask, request.candidates) if m}
    assert kept == {open_owner}


def test_supply_mask_falls_back_to_all_legal_targets(started):
    sim, request = started

    def drain(view):
        for source in view["sources"].values():
            source.update(ready=[], supply=0)

    mask = encoder(sim, dispatch_supply_mask=True).encode(with_view(request, drain))[
        "mask"
    ]
    assert list(mask) == [c.legal for c in request.candidates]


def test_commit_mask_keeps_a_loaded_vehicle_on_its_destination(started):
    sim, request = started
    target = request.candidates[1].action

    def load(view):
        view["agvs"][request.owner].update(
            job="job", target={"owner": target.owner, "port": target.port}
        )

    mask = encoder(sim, dispatch_commit=True).encode(with_view(request, load))["mask"]
    assert [i for i, m in enumerate(mask) if m] == [1]


def test_commit_mask_releases_a_full_destination(started):
    sim, request = started
    target = request.candidates[1].action
    features = list(request.candidates[1].features)
    features[10] = 1.0
    full = replace(
        request,
        candidates=(
            request.candidates[0],
            replace(request.candidates[1], features=tuple(features)),
            *request.candidates[2:],
        ),
    )

    def load(view):
        view["agvs"][request.owner].update(
            job="job", target={"owner": target.owner, "port": target.port}
        )

    mask = encoder(sim, dispatch_commit=True).encode(with_view(full, load))["mask"]
    assert list(mask) == [c.legal for c in request.candidates]


def test_target_machine_features_describe_the_machine_behind_a_buffer(started):
    sim, request = started
    rich = encoder(sim, target_machine_features=True)
    rows = rich.encode(request)["candidates"]
    for candidate, row in zip(request.candidates, rows):
        tail = list(row[-MACHINE_FEATURES:])
        machine = rich.feeds.get(candidate.action.owner)
        if machine is None:
            assert tail == [0.0] * MACHINE_FEATURES
        else:
            idle = request.observation["machines"][machine]["status"] == "IDLE"
            assert tail[0] == 1.0 and tail[1] == float(idle)


def test_role_action_modes_apply_only_to_rows_of_that_role():
    width, size = 5, 20
    torch.manual_seed(0)
    mixed = CandidateNetwork(
        size,
        {
            **default_network("ppo"),
            "role_action_modes": {"dispatcher": "sequential_binary_uniform"},
        },
        "rllib.ppo",
        candidate_width=width,
    )
    binary = CandidateNetwork(
        size,
        {**default_network("ppo"), "action_mode": "sequential_binary_uniform"},
        "rllib.ppo",
        candidate_width=width,
    )
    softmax = CandidateNetwork(
        size, default_network("ppo"), "rllib.ppo", candidate_width=width
    )
    binary.load_state_dict(mixed.state_dict())
    softmax.load_state_dict(mixed.state_dict())
    context = torch.zeros(2, size)
    context[0, 4 + 2] = 1.0  # dispatcher row
    context[1, 4 + 1] = 1.0  # buffer row
    observations = dict(
        context=context,
        candidates=torch.randn(2, 4, width),
        prefix=torch.zeros(2, 0, width),
        mask=torch.ones(2, 4, dtype=torch.bool),
        prefix_length=torch.zeros(2),
    )
    scores = mixed(observations)[0]
    assert torch.allclose(scores[0], binary(observations)[0][0])
    assert torch.allclose(scores[1], softmax(observations)[0][1])


def test_masks_ignore_value_only_requests_for_non_vehicle_owners(started):
    sim, request = started
    machine = sorted(request.observation["machines"])[0]
    value_only = replace(request, owner=machine, candidates=())
    rich = encoder(sim, dispatch_supply_mask=True, dispatch_commit=True)
    assert list(rich.encode(value_only)["mask"]) == []


@pytest.fixture(scope="module")
def round_requests():
    prepared = api.prepare(
        api.load_config(ROOT / "configs/test/runs/template1_warmup.yaml"),
        training=False,
    )
    sim = ProductionSimulator(prepared.resolved.scenario, contract="v3")
    requests = [r for r in sim.protocol.begin() if r.role == "dispatcher"]
    yield sim, requests
    sim.protocol.abort()


def one_source(request, owner, supply=1):
    def edit(view):
        for name, source in view["sources"].items():
            source.update(ready=[], supply=supply if name == owner else 0)

    return with_view(request, edit)


def test_tick_exclusive_mask_blocks_supply_taken_earlier_in_the_round(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    owner = next(
        c.action.owner
        for c in first.candidates
        if c.action.owner in first.observation["sources"]
    )
    rich = encoder(sim, dispatch_tick_exclusive=True)
    first, second = one_source(first, owner), one_source(second, owner)
    taken = [c.action.owner == owner for c in first.candidates]
    assert list(rich.encode(first)["mask"]) == taken
    rich.note_choice(
        first, next(c.action for c in first.candidates if c.action.owner == owner)
    )
    # Nothing else has supply, so the mask falls back to every legal target.
    assert list(rich.encode(second)["mask"]) == [c.legal for c in second.candidates]
    two = encoder(sim, dispatch_tick_exclusive=True)
    first2, second2 = (
        one_source(requests[0], owner, 2),
        one_source(requests[1], owner, 2),
    )
    two.note_choice(
        first2, next(c.action for c in first2.candidates if c.action.owner == owner)
    )
    assert list(two.encode(second2)["mask"]) == taken


def test_tick_exclusive_round_resets_on_a_new_boundary(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    owner = next(
        c.action.owner
        for c in first.candidates
        if c.action.owner in first.observation["sources"]
    )
    rich = encoder(sim, dispatch_tick_exclusive=True)
    rich.note_choice(
        one_source(first, owner),
        next(c.action for c in first.candidates if c.action.owner == owner),
    )
    later = replace(one_source(second, owner), tick=second.tick + 1)
    taken = [c.action.owner == owner for c in later.candidates]
    assert list(rich.encode(later)["mask"]) == taken


def test_tick_exclusive_mask_books_destination_slots(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    owner = first.candidates[0].action.owner

    def capacity_one(request):
        candidates = []
        for c in request.candidates:
            features = list(c.features)
            features[7], features[8], features[9], features[10] = (
                (0.0, 0.01, 1.0, 0.0)
                if c.action.owner == owner
                else (0.0, 0.0, 0.0, 0.0)
            )
            candidates.append(replace(c, features=tuple(features)))

        def load(view):
            for key in (first.owner, second.owner):
                view["agvs"][key].update(job="job", target=None)

        return with_view(replace(request, candidates=tuple(candidates)), load)

    rich = encoder(sim, dispatch_tick_exclusive=True)
    a, b = capacity_one(first), capacity_one(second)
    assert all(rich.encode(a)["mask"] == [c.legal for c in a.candidates])
    rich.note_choice(a, next(c.action for c in a.candidates if c.action.owner == owner))
    mask = rich.encode(b)["mask"]
    assert not any(m for m, c in zip(mask, b.candidates) if c.action.owner == owner)
    assert any(mask)


def test_tick_exclusive_is_inert_when_disabled(round_requests):
    sim, requests = round_requests
    plain = encoder(sim)
    plain.note_choice(requests[0], requests[0].candidates[0].action)
    assert list(plain.encode(requests[1])["mask"]) == [
        c.legal for c in requests[1].candidates
    ]


def test_model_policy_records_its_choice_for_the_next_vehicle(round_requests):
    from smartsom.learning.production_inference import ModelPolicy

    sim, requests = round_requests
    owner = next(
        c.action.owner
        for c in requests[0].candidates
        if c.action.owner in requests[0].observation["sources"]
    )
    rich = encoder(sim, dispatch_tick_exclusive=True)
    network = CandidateNetwork(
        rich.context_size,
        {**default_network("ppo"), "dispatch_tick_exclusive": True},
        "rllib.ppo",
        central=True,
        central_private_end=rich.central_private_end,
        candidate_width=rich.candidate_width,
    )
    policy = ModelPolicy(
        rich, network, dict(role="central", algorithm="ppo"), 101, training=True
    )
    first = one_source(requests[0], owner, 1)
    second = one_source(requests[1], owner, 2)
    assert policy.choose(first).action.owner == owner
    mask = rich.encode(second)["mask"]
    assert list(mask) == [c.action.owner == owner for c in second.candidates]
    third = one_source(requests[2], owner, 2)
    policy.choose(second)
    assert list(rich.encode(third)["mask"]) == [c.legal for c in third.candidates]


def test_idle_fallback_sends_one_empty_vehicle_per_port(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    owner = next(
        c.action.owner
        for c in first.candidates
        if c.action.owner in first.observation["sources"]
    )
    rich = encoder(sim, dispatch_tick_exclusive=True, dispatch_idle_fallback=True)
    first, second = one_source(first, owner, 2), one_source(second, owner, 2)
    chosen = next(c.action for c in first.candidates if c.action.owner == owner)
    rich.note_choice(first, chosen)
    mask = rich.encode(second)["mask"]
    assert not any(
        m and c.action.port == chosen.port for m, c in zip(mask, second.candidates)
    )


def test_idle_fallback_parks_at_the_nearest_free_port(round_requests):
    sim, requests = round_requests
    request = requests[0]
    owner = next(iter(request.observation["sources"]))
    bare = one_source(request, owner, 0)
    rich = encoder(sim, dispatch_tick_exclusive=True, dispatch_idle_fallback=True)
    mask = list(rich.encode(bare)["mask"])
    assert sum(mask) == 1
    kept = request.candidates[mask.index(True)]
    assert kept.features[1] == min(c.features[1] for c in request.candidates if c.legal)
    plain = encoder(sim, dispatch_tick_exclusive=True)
    assert list(plain.encode(bare)["mask"]) == [c.legal for c in bare.candidates]


def joint_encoder(sim):
    rich = encoder(
        sim,
        dispatch_tick_exclusive=True,
        dispatch_commit=True,
        target_machine_features=True,
        dispatch_joint_destination=True,
    )
    # A twin for operation_1 so each pickup has two possible destinations.
    rich.machine_operations["machine_002"].add("operation_1")
    return rich


def test_joint_destination_pairs_sources_with_next_machines(round_requests):
    sim, requests = round_requests
    request = requests[0]
    rich = joint_encoder(sim)
    encoded = rich.encode(request)
    entries = rich.expansion(request)
    assert len(entries) == len(encoded["mask"]) == len(encoded["candidates"])
    pairs = [(i, m) for i, m in entries if m is not None]
    sources = {request.candidates[i].action.owner for i, _ in pairs}
    assert sources == {"buffer_001"}
    assert {m for _, m in pairs} == {"machine_001", "machine_002"}
    # Pair rows carry the pair flag; other rows keep zero joint features.
    flags = encoded["candidates"][:, -3]
    assert [bool(f) for f in flags] == [m is not None for _, m in entries]
    assert encoder(sim).expansion(request) is None


def test_joint_choice_books_the_machine_and_commits_the_delivery(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    rich = joint_encoder(sim)
    rich.capacities["buffer_003"] = 1
    source = next(c for c in first.candidates if c.action.owner == "buffer_001")
    rich.encode(first)
    rich.note_choice(first, source.action, "machine_001")
    supplied = one_source(second, "buffer_001", 2)

    def queue(view):
        # The second vehicle receives the next ready job, also for operation_1.
        view["jobs"]["next"] = deepcopy(view["jobs"]["order_01/attempt/1"])
        view["sources"]["buffer_001"].update(ready=["order_01/attempt/1", "next"])

    supplied = with_view(supplied, queue)
    encoded = rich.encode(supplied)
    open_pairs = {
        m for (i, m), ok in zip(rich.expansion(supplied), encoded["mask"]) if ok and m
    }
    assert open_pairs == {"machine_002"}

    def loaded(view):
        view["agvs"][first.owner].update(
            job="order_01/attempt/1", target=dict(owner="buffer_001", port="p")
        )

    # A loaded vehicle is offered drop-off targets, e.g. both machine inputs.
    drops = tuple(
        replace(
            c,
            identity=f"TARGET:{owner}:{c.action.port}",
            action=DispatchTarget(owner, c.action.port),
        )
        for owner, c in zip(("buffer_003", "buffer_005"), first.candidates)
    )
    carrying = with_view(replace(first, tick=first.tick + 1, candidates=drops), loaded)
    mask = rich.encode(carrying)["mask"]
    kept = {c.action.owner for m, c in zip(mask, carrying.candidates) if m}
    assert kept == {"buffer_003"}
    assert first.owner not in rich.commitments


def test_model_policy_returns_the_joint_index(round_requests):
    from smartsom.learning.production_inference import ModelPolicy

    sim, requests = round_requests
    rich = joint_encoder(sim)
    options = {
        **default_network("ppo"),
        "dispatch_tick_exclusive": True,
        "dispatch_commit": True,
        "target_machine_features": True,
        "dispatch_joint_destination": True,
    }
    network = CandidateNetwork(
        rich.context_size,
        options,
        "rllib.ppo",
        central=True,
        central_private_end=rich.central_private_end,
        candidate_width=rich.candidate_width,
    )
    policy = ModelPolicy(
        rich, network, dict(role="central", algorithm="ppo"), 101, training=True
    )
    request = requests[0]
    choice = policy.choose(request)
    entries = rich.expansion(request)
    index, machine = entries[choice.index]
    assert choice.action == request.candidates[index].action
    if machine is not None:
        assert rich.commitments[request.owner] == (choice.action.owner, machine)


def test_joint_pairs_follow_the_job_the_vehicle_will_receive(round_requests):
    sim, requests = round_requests
    first, second = requests[:2]
    rich = joint_encoder(sim)
    source = next(c for c in first.candidates if c.action.owner == "buffer_001")
    rich.encode(first)
    rich.note_choice(first, source.action, "machine_001")

    def queue(view):
        # The first ready job is claimed; the next one needs operation_2.
        view["jobs"]["next"] = dict(
            view["jobs"]["order_01/attempt/1"], next_operation="operation_2"
        )
        view["sources"]["buffer_001"].update(ready=["order_01/attempt/1", "next"])

    supplied = with_view(one_source(second, "buffer_001", 2), queue)
    rich.encode(supplied)
    pairs = {m for _, m in rich.expansion(supplied) if m}
    assert pairs and pairs == {
        m for m, ops in rich.machine_operations.items() if "operation_2" in ops
    }

    # Nothing left to receive: the source stays a plain choice without pairs.
    third = with_view(
        one_source(requests[2], "buffer_001", 2),
        lambda v: v["sources"]["buffer_001"].update(ready=["order_01/attempt/1"]),
    )
    rich.encode(third)
    assert not any(m for _, m in rich.expansion(third))


def test_source_releases_the_job_for_the_waiting_vehicles_machine(round_requests):
    sim, requests = round_requests
    first = requests[0]
    rich = joint_encoder(sim)
    source = next(c for c in first.candidates if c.action.owner == "buffer_001")
    rich.encode(first)
    rich.note_choice(first, source.action, "machine_001")
    cell = list(rich.port_cells[source.action.port])

    def arrived(view):
        view["agvs"][first.owner].update(
            cell=cell, target=dict(owner="buffer_001", port=source.action.port)
        )
        job = view["jobs"]["order_01/attempt/1"]
        view["jobs"]["other"] = dict(job, next_operation="operation_2")

    jobs = ("other", "order_01/attempt/1")
    release = with_view(
        replace(
            first,
            role="buffer",
            owner="buffer_001",
            candidates=tuple(
                replace(first.candidates[0], identity=j, action=None) for j in jobs
            ),
        ),
        arrived,
    )
    mask = rich.encode(release)["mask"]
    assert [bool(m) for m in mask] == [False, True]
    # Once that job is in the prefix, nothing else is wanted: no narrowing.
    rest = replace(
        release, prefix=("order_01/attempt/1",), candidates=release.candidates[:1]
    )
    assert [bool(m) for m in rich.encode(rest)["mask"]] == [True]
