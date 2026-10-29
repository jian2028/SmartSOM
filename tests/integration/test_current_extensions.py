"""Explicit extension source, schema and nonstandard input shape acceptance."""

from pathlib import Path

import pytest
import yaml

from smartsom.config.algorithm_migration import migrate_algorithm
from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.reservation import compile_reservation

ROOT = Path(__file__).resolve().parents[2]


def authored(tmp_path, method):
    origin = ROOT / "configs/base-diagnostic/small.yaml"
    experiment = yaml.safe_load(origin.read_text())
    for name in ("factory", "workload"):
        experiment[name] = str(origin.parent / experiment[name])
    algorithm = migrate_algorithm(
        {
            "schema": "smartsom.base-algorithm/v1",
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
            "workload.orders=8",
            "workload.initial_orders=4",
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


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_mode_sensitive_extension_has_one_forward_contract(tmp_path, method):
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.algorithms.observation import build_observation
    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
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
        width=8, batch_size=1, target_update_interval=2
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
        tuple(
            BaseCandidate(str(i), (str(i), "normal"), vector(i, i + 1))
            for i in range(3)
        ),
        vector(),
        ((), (), (), ()),
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
        assert all(
            torch.equal(policy.learner.models.state_dict()[k], v)
            for k, v in policy.learner.targets.state_dict().items()
        )
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
