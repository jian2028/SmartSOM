"""Declarative implementations are typed, frozen and local to each policy."""

from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config.codec import primitive
from smartsom.config.extensions import ExtensionRef
from smartsom.config.policies import PolicyExtensions
from smartsom.config.policy_contracts import pin_contract

ROOT = Path(__file__).resolve().parents[2]


def physical(inspection):
    return ExtensionRef(
        name="builtin.physical_job",
        version="1",
        parameters={"include_inspection": inspection},
    )


def masked():
    return ExtensionRef(name="builtin.physical_job_candidate", version="1")


def test_new_default_does_not_change_historical_policy_serialization():
    assert primitive(PolicyExtensions()) == {"observation": None, "network": None}


def test_frozen_pair_and_source_integrity():
    observation, network = pin_contract(physical(True), masked())
    assert len(observation.code_sha256) == 64
    assert network.code_sha256 == observation.code_sha256
    assert pin_contract(observation, network) == (observation, network)
    with pytest.raises(ValueError, match="digest changed"):
        pin_contract(observation.model_copy(update={"code_sha256": "0" * 64}), network)


@pytest.mark.parametrize(
    "parameters",
    [{}, {"include_inspection": "true"}, {"include_inspection": True, "unknown": 1}],
)
def test_invalid_physical_parameters(parameters):
    with pytest.raises(ValueError):
        pin_contract(
            ExtensionRef(
                name="builtin.physical_job", version="1", parameters=parameters
            ),
            masked(),
        )


def test_unpaired_or_unknown_implementation_rejected():
    for observation, network in (
        (physical(True), None),
        (None, masked()),
        (None, ExtensionRef(name="arbitrary.module", version="1")),
    ):
        with pytest.raises(ValueError):
            pin_contract(observation, network)


def test_instance_specific_inspection_and_candidate_privacy():
    torch = pytest.importorskip("torch")
    from smartsom import api
    from smartsom.engine.production import ProductionSimulator
    from smartsom.learning import physical_job_observation, production_models
    from smartsom.learning.policy_factory import encoder, network_class
    from smartsom.learning.production_models import default_network, tensor_inputs

    assert not physical_job_observation._INSTALLED
    native_class = production_models.PublicEncoder
    prepared = api.prepare(
        api.load_config(ROOT / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    factory = prepared.scenario.factory
    factory = replace(
        factory,
        buffers=tuple(
            replace(buffer, storage=replace(buffer.storage, capacity=10))
            if buffer.role == "system_input"
            else buffer
            for buffer in factory.buffers
        ),
    )
    for algorithm in ("ppo", "dqn"):
        instances = []
        for inspection in (True, False):
            observation, implementation = pin_contract(physical(inspection), masked())
            metadata = dict(
                observation=primitive(observation),
                network_implementation=primitive(implementation),
                projection=dict(time_scale=100, count_scale=100),
                provider="rllib.resource_" + algorithm,
                role="dispatcher",
            )
            enc = encoder(factory, metadata)
            instances.append(enc)
            sim = ProductionSimulator(
                replace(prepared.scenario, factory=factory), contract="v3"
            )
            requests = sim.protocol.begin()
            request = next(
                request for request in requests if request.role == "dispatcher"
            )
            encoded = enc.encode(request)
            net = network_class(metadata["network_implementation"])(
                enc.context_size,
                default_network(algorithm),
                metadata["provider"],
                algorithm,
                candidate_width=enc.candidate_width,
            )
            original = tensor_inputs([encoded], "cpu")
            altered = {key: value.clone() for key, value in original.items()}
            altered["context"][:, -8:] += 100
            with torch.no_grad():
                scores, values = net(original)
                changed, _ = net(altered)
            torch.testing.assert_close(scores, changed, rtol=0, atol=0)
            assert torch.isfinite(values).all()
        assert instances[0].observation_identity != instances[1].observation_identity
        with pytest.raises(ValueError, match="identity changed"):
            instances[0].load_state_dict(instances[1].state_dict())
    assert production_models.PublicEncoder is native_class
    assert not physical_job_observation._INSTALLED
