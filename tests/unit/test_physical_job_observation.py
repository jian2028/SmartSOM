"""Opt-in observation identity and actor/critic separation contracts."""

import copy

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
import numpy as np
import torch

from smartsom.learning import physical_job_observation as module
from smartsom.learning.production_models import default_network


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("dimensions", [(998, 127, 220), (1904, 220, 371)])
@pytest.mark.parametrize("central", [False, True])
def test_private_summary_mask_and_independent_clone(algorithm, dimensions, central):
    context_size, width, private_end = dimensions
    network = module.PhysicalJobNetwork(
        context_size,
        default_network(algorithm),
        "rllib.resource_" + algorithm,
        algorithm=algorithm,
        candidate_width=width,
        central=central,
        central_private_end=private_end,
    )
    clone = copy.deepcopy(network)
    state = copy.deepcopy(network.state_dict())
    with torch.no_grad():
        next(clone.parameters()).add_(1)
    clone.load_state_dict(state, strict=True)
    for a, b in zip(network.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.data_ptr() != b.data_ptr()
    inputs = dict(
        context=torch.zeros(1, context_size),
        candidates=torch.zeros(1, 2, width),
        prefix=torch.zeros(1, 0, width),
        mask=torch.tensor([[True, False]]),
        prefix_length=torch.tensor([0]),
    )
    altered = {key: value.clone() for key, value in inputs.items()}
    altered["context"][:, -8:] = 2
    with torch.no_grad():
        scores, values = network(inputs)
        changed_scores, changed_values = network(altered)
    torch.testing.assert_close(scores, changed_scores, rtol=0, atol=0)
    assert torch.isfinite(scores).all() and torch.isfinite(values).all()
    if algorithm == "dqn":
        assert network.critic is None
        torch.testing.assert_close(values, changed_values, rtol=0, atol=0)
    else:
        assert not torch.equal(values, changed_values)


def test_empty_summary_and_stable_schema():
    np.testing.assert_array_equal(
        module.critic_features({"jobs": {}}, None, 100, 100), np.zeros(8)
    )
    assert module.SCHEMA == "smartsom.physical-job-observation/v1"
    assert len(module.QUALIFIED_SOURCE_SHA256) == 64


def test_invalid_inspection_choice_does_not_install():
    with pytest.raises(ValueError, match="explicit inspection"):
        module.install(include_inspection=None)


def test_package_schema_rejects_old_identity(monkeypatch):
    monkeypatch.setattr(module, "_INSTALLED", True)
    monkeypatch.setattr(module, "IDENTITY", module.SCHEMA + "/inspection=True")
    module.validate_package_identity({"physical_job_encoder": module.IDENTITY})
    for metadata in (
        {},
        {"batch04_encoder": "old-local-hash"},
        {"physical_job_encoder": module.SCHEMA + "/inspection=False"},
    ):
        with pytest.raises(ValueError, match="encoder package rejected"):
            module.validate_package_identity(metadata)


def test_public_reference_features_ignore_latent_truth():
    from types import SimpleNamespace

    factory = SimpleNamespace(
        machines=[
            SimpleNamespace(
                machine_id="m",
                operation_types=("op",),
                processing_rate_multiplier=1,
                quality_modes=[SimpleNamespace(quality_mode_id="normal", time_scale=1)],
            )
        ]
    )
    row = dict(
        demand="released",
        physical_job_original_reference_work=20,
        remaining_steps=[dict(operation_type="op", nominal_ticks=20)],
        location="m",
        due_at=100,
        defective=False,
    )
    view = dict(
        jobs={"j": row},
        released=["released"],
        tick=10,
        machines={"m": dict(job="j", status="PROCESSING", nominal=20, elapsed=10)},
    )
    expected = (0.8, 0.5, 0.1)
    np.testing.assert_allclose(module.processing_features("j", view, factory), expected)
    row["defective"] = True
    view["future_orders"] = [dict(due_at=-100000)]
    np.testing.assert_allclose(module.processing_features("j", view, factory), expected)
    np.testing.assert_allclose(
        module.critic_features(view, factory, 100, 100),
        (0.8, 0.8, 0, 0.1, 0.1, 0.001, 0.5, 1),
    )
    view["released"] = []
    with pytest.raises(ValueError, match="unreleased"):
        module.processing_features("j", view, factory)


@pytest.mark.parametrize("archive", [False, True])
def test_encoder_schema_rejected_before_missing_weights(tmp_path, monkeypatch, archive):
    import json
    import zipfile

    from smartsom import api
    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )
    from smartsom.domain.travel_time import physical_contract
    from smartsom.learning.production_inference import read_package

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    prepared = api.prepare(
        api.load_config(root / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    metadata = dict(
        role="machine",
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        physical_contract=physical_contract(prepared.scenario),
    )
    monkeypatch.setattr(module, "_INSTALLED", True)
    monkeypatch.setattr(module, "IDENTITY", module.SCHEMA + "/inspection=True")
    if archive:
        source = tmp_path / "metadata.zip"
        with zipfile.ZipFile(source, "w") as stream:
            stream.writestr("model.json", json.dumps(metadata))
    else:
        source = tmp_path / "model"
        source.mkdir()
        (source / "model.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="encoder package rejected"):
        read_package(source, metadata_validator=module.validate_package_identity)


@pytest.mark.parametrize("override,expected", [(None, 10.5), (12, 12.0)])
def test_exact_public_reference_and_machine_override(override, expected):
    from types import SimpleNamespace

    machine = SimpleNamespace(
        machine_id="m",
        operation_types=("op",),
        processing_rate_multiplier=1,
        quality_modes=[SimpleNamespace(quality_mode_id="normal", time_scale=1)],
    )
    step = dict(
        operation_type="op",
        nominal_ticks=10,
        reference_ticks="21/2",
        machine_nominal_ticks={} if override is None else {"m": override},
    )
    assert (
        module._normal_reference(step, SimpleNamespace(machines=[machine])) == expected
    )


def test_policy_restore_rejects_identity_before_rng_or_encoder_mutation():
    from smartsom.learning.production_inference import ModelPolicy

    encoder = module.PhysicalJobEncoder.__new__(module.PhysicalJobEncoder)
    encoder.observation_identity = module.SCHEMA + "/inspection=False"
    policy = ModelPolicy.__new__(ModelPolicy)
    policy.encoder = encoder

    class NoMutation:
        def set_state(self, *args):
            raise AssertionError("RNG mutated before identity validation")

    policy.generator = NoMutation()
    state = dict(
        generator=None,
        encoder={"physical_job_encoder": module.SCHEMA + "/inspection=True"},
    )
    with pytest.raises(ValueError, match="observation identity changed"):
        policy.load_state_dict(state)
    encoder.validate_state_dict({"physical_job_encoder": encoder.observation_identity})


@pytest.mark.parametrize("sampler_mismatch", [False, True])
def test_session_restore_preflights_all_policies_before_mutation(sampler_mismatch):
    from types import SimpleNamespace

    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )
    from smartsom.experiments.composable import TrainingSession
    from smartsom.learning.production_inference import ModelPolicy

    encoder = module.PhysicalJobEncoder.__new__(module.PhysicalJobEncoder)
    encoder.observation_identity = module.SCHEMA + "/inspection=False"
    policy = ModelPolicy.__new__(ModelPolicy)
    policy.encoder = encoder
    session = TrainingSession.__new__(TrainingSession)
    session.prepared = SimpleNamespace(scientific_sha256="fixture")
    session.config = SimpleNamespace(
        runtime=SimpleNamespace(num_envs=1, sampling_processes=0)
    )
    session.policies = {"machine": policy}
    session.ticks = 123
    state = dict(
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        scientific_sha256="fixture",
        ticks=999,
        policies={
            "machine": {
                "encoder": {"physical_job_encoder": module.SCHEMA + "/inspection=True"}
            }
        },
    )
    if sampler_mismatch:
        state["sampler_states"] = [copy.deepcopy(state["policies"])]
        state["policies"]["machine"]["encoder"]["physical_job_encoder"] = (
            encoder.observation_identity
        )
    with pytest.raises(ValueError, match="observation identity changed"):
        session.restore(state)
    assert session.ticks == 123


def test_real_spawn_sampling_installs_observation_contract(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "spawn_probe.py"
    script.write_text(
        (root / "tests/helpers/physical_job_contract_probe.py")
        .read_text()
        .replace("__ROOT__", repr(str(root)))
    )
    env = dict(
        os.environ,
        SMARTSOM_TEST_ROOT=str(root),
        PYTHONUTF8="1",
        PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_SPAWN" in result.stdout


def test_native_matching_full_restore_and_identity_rejection(tmp_path):
    pytest.importorskip("ray")
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "restore_probe.py"
    script.write_text(
        (root / "tests/helpers/physical_job_contract_probe.py")
        .read_text()
        .replace("__ROOT__", repr(str(root)))
    )
    env = dict(
        os.environ,
        SMARTSOM_TEST_ROOT=str(root),
        PYTHONUTF8="1",
        PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
    )
    result = subprocess.run(
        [sys.executable, str(script), "--restore"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MATCHING_NATIVE" in result.stdout


def test_same_dimensions_different_inspection_state_rejects_before_mutation(
    monkeypatch,
):
    from dataclasses import replace
    from pathlib import Path

    from smartsom import api

    root = Path(__file__).resolve().parents[2]
    prepared = api.prepare(
        api.load_config(root / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    factory = prepared.scenario.factory
    factory = replace(
        factory,
        buffers=tuple(
            replace(b, storage=replace(b.storage, capacity=10))
            if b.role == "system_input"
            else b
            for b in factory.buffers
        ),
    )
    monkeypatch.setattr(module, "INCLUDE_INSPECTION", True)
    monkeypatch.setattr(module, "IDENTITY", module.SCHEMA + "/inspection=True")
    included = module.PhysicalJobEncoder(
        factory, {"time_scale": 100, "count_scale": 100}
    )
    monkeypatch.setattr(module, "INCLUDE_INSPECTION", False)
    monkeypatch.setattr(module, "IDENTITY", module.SCHEMA + "/inspection=False")
    excluded = module.PhysicalJobEncoder(
        factory, {"time_scale": 100, "count_scale": 100}
    )
    assert included.context_size == excluded.context_size
    assert included.candidate_width == excluded.candidate_width
    assert included.global_capacity > excluded.global_capacity
    before = copy.deepcopy(excluded.state_dict())
    for invalid in (
        included.state_dict(),
        {k: v for k, v in before.items() if k != "physical_job_encoder"},
    ):
        with pytest.raises(ValueError, match="observation identity changed"):
            excluded.load_state_dict(invalid)
        assert excluded.state_dict() == before
    excluded.load_state_dict(before)
    assert excluded.state_dict() == before
