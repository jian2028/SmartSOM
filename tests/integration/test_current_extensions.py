"""Explicit extension source, schema and nonstandard input shape acceptance."""

from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.reservation import compile_reservation
from tests.helpers.algorithm import algorithm_document

ROOT = Path(__file__).resolve().parents[2]


def authored(tmp_path, method):
    origin = ROOT / "experiments/examples/small/default/experiment.yaml"
    experiment = yaml.safe_load(origin.read_text())
    experiment["scenario"] = str(origin.parent / experiment["scenario"])
    algorithm = algorithm_document(
        {
            "method": method,
            "epochs": 1,
            "observation": {"path": str(ROOT / "examples/current/observation.py")},
            "network": {"path": str(ROOT / "examples/current/network.py")},
        }
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    experiment["algorithm"] = "algorithm.yaml"
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    return tmp_path / "experiment.yaml"


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_custom_shape_freezes_and_continues(tmp_path, method):
    pytest.importorskip("torch")
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.experiments.reservation_runtime import episode
    from smartsom.learning.records import recorded_request, request_from_dict

    path = authored(tmp_path, method)
    plan = compile_reservation(
        path,
        sets=[
            "scenario.jobs.orders.total=8",
            "scenario.jobs.orders.initial=4",
            f"algorithm.production.policies.{method}.learner.params.width=8",
        ],
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    path.write_text("invalid: later edit")
    _, factory, workload, options, _ = load_inputs(frozen / "experiment.yaml")
    policy = build_policy(options, 9)
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        1000,
        max_ticks=80,
        metrics_level="off",
    )
    request = policy.records[0]["request"]
    assert len(request.inputs["context"]) == 3
    assert request_from_dict(recorded_request(request)).inputs == request.inputs
    policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    checkpoint = tmp_path / "current.pt"
    policy.save(checkpoint)
    restored = build_policy(options, 9)
    restored.load(checkpoint)
    assert restored.updates == policy.updates > 0
    from smartsom.config.components import digest_files

    frozen_source = Path(options.observation.path)
    assert options.observation.sha256 == digest_files(
        {frozen_source.name: frozen_source.read_bytes()}
    )
    frozen_source.write_text("# modified frozen source")
    from smartsom.algorithms.observation import build_observation

    with pytest.raises(ValueError, match="source mismatch"):
        build_observation(options)


def test_schema_mismatch_rejected_before_run(tmp_path):
    path = authored(tmp_path, "ppo")
    algorithm = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    network = tmp_path / "network.py"
    network.write_text(
        (ROOT / "examples/current/network.py")
        .read_text()
        .replace("example.public-counts/v1", "different/v1")
    )
    algorithm["production"]["policies"][algorithm["production"]["default"]]["network"][
        "path"
    ] = str(network)
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    with pytest.raises(ValueError, match="INPUT_SCHEMA mismatch"):
        compile_reservation(path)


@pytest.mark.parametrize(
    "method,target_clock",
    [
        ("ppo", "network_gradient_steps"),
        ("dqn", "network_gradient_steps"),
        ("dqn", "global_gradient_steps"),
    ],
)
def test_mode_sensitive_extension_has_one_forward_contract(
    tmp_path, method, target_clock
):
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.algorithms.observation import build_observation
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
    from smartsom.domain.public_facts import freeze

    path = authored(tmp_path, method)
    network = tmp_path / "network.py"
    text = (ROOT / "examples/current/network.py").read_text()
    text = text.replace(
        "class Network(nn.Module):",
        "class FeatureBatchNorm(nn.BatchNorm1d):\n"
        "    def forward(self, x):\n"
        "        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)\n\n\n"
        "class Network(nn.Module):",
    ).replace("nn.Tanh()", "nn.Sequential(FeatureBatchNorm(width), nn.Dropout(0.5))")
    network.write_text(text)
    algorithm = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    algorithm["production"]["policies"][method]["learner"]["params"].update(
        width=8,
        batch_size=1,
        target_update_interval=2,
        target_update_clock=target_clock,
    )
    algorithm["production"]["policies"][algorithm["production"]["default"]]["network"][
        "path"
    ] = str(network)
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    plan = compile_reservation(path)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    _, _, _, options, _ = load_inputs(frozen / "experiment.yaml")
    raw = BaseDecision(
        0,
        "machine",
        "m",
        "process",
        tuple(BaseCandidate(str(i), (str(i), "normal")) for i in range(3)),
    )
    request = build_observation(options)(
        freeze({"tick": 0, "jobs": (), "vehicles": ()}), raw
    )
    evaluation = build_policy(options, 7, training=False, selection="argmax")
    initial = {k: v.clone() for k, v in evaluation.learner.models.state_dict().items()}
    rng = torch.get_rng_state().clone()
    assert len({evaluation.choose(request, None) for _ in range(100)}) == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(initial[k], v)
        for k, v in evaluation.learner.models.state_dict().items()
    )
    assert not any(m.training for m in evaluation.learner.models.modules())

    policy = build_policy(options, 7)
    buffers = {k: v.clone() for k, v in policy.learner.models.named_buffers()}
    weights = {k: v.clone() for k, v in policy.learner.models.named_parameters()}
    critic = [0.0] * 48
    policy.choose(request, critic)
    policy.choose(replace(request, tick=1), critic)
    if method == "ppo":
        # Before optimization, likelihoods agree for both singleton and mixed
        # batches; hidden layer mode cannot create a spurious PPO policy ratio.
        for rows in ([policy.records[0]], policy.records):
            packed = policy.learner.pack(
                [r["request"] for r in rows], [critic] * len(rows), options.count_scale
            )
            logits, _ = policy.learner.models["machine"](packed)
            log_probs = torch.log_softmax(logits, -1)
            ratios = torch.stack(
                [
                    (log_probs[i, r["index"]] - r["log_prob"]).exp()
                    for i, r in enumerate(rows)
                ]
            )
            assert torch.allclose(ratios, torch.ones_like(ratios), atol=1e-6)
    else:
        targets = {k: v.clone() for k, v in policy.learner.targets.state_dict().items()}
    policy.update([1.0, 2.0], terminal=False, final_critic=critic)
    assert policy.updates > 0
    assert any(
        not torch.equal(weights[k], v)
        for k, v in policy.learner.models.named_parameters()
    )
    assert all(
        torch.equal(buffers[k], v) for k, v in policy.learner.models.named_buffers()
    )
    if method == "dqn":
        assert all(
            torch.equal(targets[k], v)
            for k, v in policy.learner.targets.state_dict().items()
        )
        assert not any(p.requires_grad for p in policy.learner.targets.parameters())
        # Explicit target refresh copies buffers as well as learned parameters.
        for name, buffer in policy.learner.models.named_buffers():
            if name.endswith("running_mean"):
                buffer.fill_(0.5)
        policy.update([], terminal=True)
        assert policy.updates == 2
        # Only machine received experience. Global mode refreshes all targets;
        # network mode must leave the other networks (including BN buffers) alone.
        for key, target in policy.learner.targets.items():
            refreshed = target_clock == "global_gradient_steps" or key == "machine"
            assert policy.learner.key_updates[key] == (2 if key == "machine" else 0)
            assert policy.learner.target_updates[key] == int(refreshed)
            for name, value in target.state_dict().items():
                expected = (
                    policy.learner.models[key].state_dict()[name]
                    if refreshed
                    else targets[f"{key}.{name}"]
                )
                assert torch.equal(expected, value), (target_clock, key, name)
        assert not any(m.training for m in policy.learner.targets.modules())
    checkpoint = tmp_path / "checkpoint.pt"
    policy.save(checkpoint)
    restored = build_policy(options, 99)
    restored.load(checkpoint)
    assert not any(m.training for m in restored.learner.models.modules())
    assert all(
        torch.equal(policy.learner.models.state_dict()[k], v)
        for k, v in restored.learner.models.state_dict().items()
    )
    if method == "dqn":
        assert not any(m.training for m in restored.learner.targets.modules())
        assert restored.learner.target_updates == policy.learner.target_updates
        assert all(
            torch.equal(policy.learner.targets.state_dict()[key], value)
            for key, value in restored.learner.targets.state_dict().items()
        )


def compact_authored(tmp_path, method="dqn", social=False, action_mode="direct"):
    path = authored(tmp_path, method)
    algorithm = yaml.safe_load(
        (ROOT / "examples/current/compact-base.yaml").read_text()
    )
    policy = algorithm["production"]["policies"]["dqn"]
    policy["learner"]["target"] = f"smartsom.learning.{method}.create_learner"
    policy["learner"]["params"].update(batch_size=4, epochs=1, width=8)
    policy["observation"]["params"]["social"] = social
    for name in ("observation", "network"):
        policy[name]["path"] = str(ROOT / "examples/current" / policy[name]["path"])
        policy[name]["source_root"] = str(ROOT / "examples/current/compact")
    algorithm["action_representation"]["default"] = action_mode
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    return path


@pytest.mark.parametrize(
    "method,social,action_mode",
    [
        ("dqn", False, "direct"),
        ("dqn", True, "rules"),
        ("ppo", False, "rules"),
        ("ppo", True, "direct"),
    ],
)
def test_semantic_observation_update_restore_and_frozen_bundle(
    tmp_path, method, social, action_mode
):
    torch = pytest.importorskip("torch")
    from smartsom.algorithms.base_policy import build_policy
    from smartsom.experiments.reservation_runtime import episode
    from smartsom.learning.records import recorded_request, request_from_dict

    path = compact_authored(tmp_path, method, social, action_mode)
    plan = compile_reservation(
        path, sets=["scenario.jobs.orders.total=8", "scenario.jobs.orders.initial=4"]
    )
    frozen = plan.materialize(tmp_path / "frozen")
    _, factory, workload, options, _ = load_inputs(frozen)
    policy = build_policy(options, 9)
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        1000,
        max_ticks=80,
        metrics_level="off",
    )
    assert policy.records
    for record in policy.records:
        request = record["request"]
        assert not hasattr(request, "context") and not hasattr(request, "sets")
        assert all(not hasattr(c, "features") for c in request.candidates)
        assert request_from_dict(recorded_request(request)).inputs == request.inputs
        assert len(request.inputs["base"]) == 2
        assert all(len(row) == (2 if social else 0) for row in request.inputs["social"])
    request = policy.records[0]["request"]
    before = {name: p.clone() for name, p in policy.learner.models.named_parameters()}
    policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    assert policy.updates > 0
    assert any(
        not torch.equal(before[n], p)
        for n, p in policy.learner.models.named_parameters()
    )
    checkpoint = tmp_path / "compact.pt"
    policy.save(checkpoint)
    restored = build_policy(options, 9)
    restored.load(checkpoint)
    assert restored.updates == policy.updates
    for n, p in restored.learner.models.state_dict().items():
        assert torch.equal(p, policy.learner.models.state_dict()[n])
    from smartsom.domain.training import bootstrap_input

    assert restored.choose(
        request, bootstrap_input(critic, request) if method == "ppo" else None
    ) in {c.action for c in request.candidates}
    helper = Path(options.observation.source_root) / "features.py"
    assert helper.is_file()
    helper.write_text(helper.read_text() + "\n# changed layout\n")
    from smartsom.algorithms.observation import build_observation

    with pytest.raises(ValueError, match="source mismatch"):
        build_observation(options)


def test_compact_base_and_social_share_exact_base_inputs():
    from examples.current.compact.observation import create_observation
    from smartsom.engine.public_facts import public_facts
    from tests.unit.test_base_reservations import decide, sim_with

    sim = sim_with()
    requests = decide(sim)
    facts = public_facts(sim)
    for request in requests:
        base = create_observation()(facts, request)
        social = create_observation(social=True)(facts, request)
        assert base["base"] == social["base"]
        assert base["candidates"] == social["candidates"]
        assert all(row == () for row in base["social"])


def test_custom_observation_skips_base_projection_and_preserves_physics(
    tmp_path, monkeypatch
):
    from smartsom.algorithms.observation import build_observation
    from smartsom.config.base_diagnostic import BaseAlgorithm
    from smartsom.traffic.rule_mover import SharedMover
    from tests.unit.test_base_reservations import decide, sim_with

    compact_authored(tmp_path)
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    )
    old, new = sim_with(n=8), sim_with(n=8)
    new.protocol.observation_transform = build_observation(options)
    new.protocol.social_arm = "all"  # Even old SL construction is bypassed.

    from smartsom.observations.base import Projection

    original = Projection.build

    def guarded(projection):
        assert projection.core._protocol is not new.protocol, (
            "custom observation invoked Base projection"
        )
        return original(projection)

    monkeypatch.setattr(Projection, "build", guarded)
    assert not hasattr(new.protocol, "job_vector")
    movers = [SharedMover(sim.protocol.routes) for sim in (old, new)]
    seen = set()

    def choose(request):
        seen.add((request.group, request.stage))
        return min(
            c.action
            for c in request.candidates
            if c.action not in (("STOP",), ("EMPTY_WAIT",))
        )

    for _ in range(250):
        requests = [decide(sim, choose) for sim in (old, new)]
        assert [
            [(c.identity, c.action) for c in r.candidates] for r in requests[0]
        ] == [[(c.identity, c.action) for c in r.candidates] for r in requests[1]]
        assert old.snapshot() == new.snapshot()
        if old.done:
            break
        for sim, mover in zip((old, new), movers):
            sim.protocol.advance(mover.moves(sim.protocol.traffic_view(), sim.tick))
    assert old.done and new.done
    assert ("machine", "process") in seen and ("pre", "service") in seen


@pytest.mark.parametrize("value", ["typo", False])
def test_invalid_observation_request_fields_fails_compilation(tmp_path, value):
    path = authored(tmp_path, "dqn")
    observation = tmp_path / "observation.py"
    observation.write_text(
        (ROOT / "examples/current/observation.py").read_text()
        + f"\nencode.request_fields = {value!r}\n"
    )
    algorithm = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    algorithm["production"]["policies"]["dqn"]["observation"]["path"] = str(observation)
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    with pytest.raises(ValueError, match="request_fields"):
        compile_reservation(path)


def test_semantic_rule_projection_and_mixed_bindings_preserve_candidates(tmp_path):
    from smartsom.algorithms.observation import build_observation
    from smartsom.config.base_diagnostic import BaseAlgorithm
    from smartsom.traffic.rule_mover import SharedMover
    from tests.unit.test_base_reservations import decide, sim_with

    compact_authored(tmp_path, action_mode="rules")
    document = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    document["production"]["bindings"] = {"machine.process": "machine_rule"}
    document["production"]["policies"]["machine_rule"] = {
        "kind": "policy",
        "observation": {"target": "smartsom.observations.rules.create_observation"},
        "implementation": {"target": "smartsom.policies.rule.create_policy"},
    }
    custom = BaseAlgorithm.model_validate(document)
    document["production"]["policies"]["dqn"]["observation"] = {
        "target": "smartsom.representations.base.observation.create_observation"
    }
    native = BaseAlgorithm.model_validate(document)
    sims = [sim_with(n=8), sim_with(n=8)]
    for sim, options in zip(sims, (native, custom)):
        sim.protocol.observation_transform = build_observation(options)
    movers = [SharedMover(sim.protocol.routes) for sim in sims]
    seen = set()

    def choose(request):
        return min(
            c.action
            for c in request.candidates
            if c.action not in (("STOP",), ("EMPTY_WAIT",))
        )

    while not sims[0].done:
        left, right = [decide(sim, choose) for sim in sims]
        assert len(left) == len(right)
        for a, b in zip(left, right):
            seen.add(b.group)
            assert [(c.identity, c.action) for c in a.candidates] == [
                (c.identity, c.action) for c in b.candidates
            ]
            assert a.rule_audit == b.rule_audit
            if b.group == "machine":
                assert a == b and "candidates" in b.inputs
            else:
                assert not hasattr(b, "context") and not hasattr(b, "sets")
                assert all(not hasattr(c, "features") for c in b.candidates)
        assert sims[0].snapshot() == sims[1].snapshot()
        for sim, mover in zip(sims, movers):
            sim.protocol.advance(mover.moves(sim.protocol.traffic_view(), sim.tick))
    assert sims[1].done and {"machine", "dispatcher", "pre", "post"} <= seen


def test_compact_statistics_have_hand_calculated_values():
    from examples.current.compact.observation import create_observation
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
    from smartsom.domain.public_facts import freeze

    facts = freeze(
        {
            "tick": 20,
            "jobs": [
                {"id": "j", "due_at": 10, "remaining_operations": [{"type": "A"}]}
            ],
            "vehicles": [],
            "buffers": [
                {"id": "pre", "inventory": {"pool": ["j"]}, "ready": ["j"]},
                {"id": "post", "inventory": {"pool": []}, "ready": []},
                {"id": "peer_pre", "inventory": {"pool": ["j"]}, "ready": []},
            ],
            "machines": [
                {
                    "id": "m",
                    "pre": "pre",
                    "post": "post",
                    "job": None,
                    "capabilities": ["A"],
                    "status": "READY",
                },
                {
                    "id": "peer",
                    "pre": "peer_pre",
                    "post": "peer_post",
                    "job": None,
                    "capabilities": ["A"],
                    "status": "PROCESSING",
                },
            ],
        }
    )
    decision = BaseDecision(
        20,
        "machine",
        "m",
        "process",
        (BaseCandidate("start:j:normal", ("j", "normal")),),
    )
    actual = create_observation(social=True, count_scale=10, time_scale=100)(
        facts, decision
    )
    assert actual == {
        "base": (0.1, 0.1),
        "candidates": ((0.1, -0.1),),
        "social": ((1.0, 0.1),),
    }


def test_named_output_constructor_detaches_nested_containers_and_rejects_nonstring_keys():
    from smartsom.domain.public_facts import Facts, freeze

    rows = []
    result = freeze(Facts((("rows", rows),)))
    rows.append(9)
    assert result["rows"] == ()
    with pytest.raises(TypeError, match="keys must be strings"):
        freeze({7: "seven"})


def test_public_demand_iteration_is_stable_and_excludes_unreleased_ids():
    from smartsom.domain.base_decisions import BaseDecision
    from smartsom.engine.decision_facts import DecisionFacts
    from tests.unit.test_base_reservations import sim_with

    sim = sim_with(n=8)
    raw = BaseDecision(sim.tick, "machine", "m", "process", ())
    with DecisionFacts(sim.protocol, raw) as facts:
        assert list(facts.demands) == sorted(sim.released)
        assert list(facts.demands.items()) == [
            (key, sim.demands[key]) for key in sorted(sim.released)
        ]
    with pytest.raises(RuntimeError):
        list(facts.demands)


def test_closed_public_queries_do_not_retain_the_request_graph():
    import gc
    import weakref

    from smartsom.domain.base_decisions import BaseDecision
    from smartsom.engine.decision_facts import DecisionFacts
    from tests.unit.test_base_reservations import sim_with

    sim = sim_with(n=8)
    raw = BaseDecision(sim.tick, "machine", "m", "process", ())
    enabled = gc.isenabled()
    gc.disable()
    try:
        with DecisionFacts(sim.protocol, raw) as facts:
            snapshot = facts["jobs"]
            table, routes = facts.jobs, facts.routes
            job = next(iter(table))
            row = table[job]
            reference = weakref.ref(facts)
        del facts
        # Saved rows/queries must not keep the entire global snapshot alive.
        assert reference() is None
        assert table[job] is row
        assert snapshot
        with pytest.raises(RuntimeError):
            len(table)
        with pytest.raises(RuntimeError):
            routes.to_owner((0, 0), "input")
    finally:
        if enabled:
            gc.enable()


def test_global_job_snapshot_refreshes_public_values_within_one_tick():
    from dataclasses import replace

    from smartsom.engine.public_facts import public_facts
    from tests.unit.test_base_reservations import sim_with

    sim = sim_with(n=2)
    before = public_facts(sim)
    job = before["jobs"][0]["id"]
    row = sim.jobs[job]
    demand_id = row["demand"]
    due = sim.demands[demand_id].due_at
    row.update(step=1, risk=0.25)
    sim.demands[demand_id] = replace(sim.demands[demand_id], due_at=due + 7)
    after = public_facts(sim)
    old = next(r for r in before["jobs"] if r["id"] == job)
    new = next(r for r in after["jobs"] if r["id"] == job)
    assert before["tick"] == after["tick"]
    assert old["step"] == 0 and new["step"] == 1
    assert old["due_at"] == due and new["due_at"] == due + 7
    assert old["public_risk"] != 0.25 and new["public_risk"] == 0.25
    assert len(old["remaining_operations"]) == 2
    assert len(new["remaining_operations"]) == 1
    assert new["remaining_operations"][0]["type"] == "B"
