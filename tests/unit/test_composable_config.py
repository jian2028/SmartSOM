"""Classified file ownership, strict training selection and model identity."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from smartsom import api
from smartsom.config.codec import ConfigurationError
from smartsom.config.experiment import apply_overrides
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.domain.production_decisions import ACTION_CONTRACT, OBSERVATION_CONTRACT
from smartsom.learning.production_contract import factory_identity

ROOT = Path(__file__).resolve().parents[2]


def project(tmp_path, *, training=True):
    config = api.load_config(ROOT / "configs/test/runs/train_machine_ppo.yaml")
    comp = yaml.safe_load(Path(config.composition).read_text())
    for group in comp["groups"].values():
        group["policy"] = str(
            (Path(config.composition).parent / group["policy"]).resolve()
        )
    comp["pickup_matching"] = str(
        (Path(config.composition).parent / comp["pickup_matching"]).resolve()
    )
    target = tmp_path / "composition.yaml"
    target.write_text(yaml.safe_dump(comp))
    config.composition = str(target)
    config.output.root = str(tmp_path / "runs")
    if not training:
        config.training = None
    return config, comp, target


def test_show_config_resolves_classified_graph_without_allocating(tmp_path):
    config, _, _ = project(tmp_path)
    preview = api.show_config(config)
    assert preview["groups"]["machine"]["training"]
    assert not preview["groups"]["mover"]["training"]
    assert preview["bindings"]["machine"]["default"] == "machine"
    assert preview["seeds"] == {"training": 101, "validation": 303, "evaluation": 202}
    assert not Path(config.output.root).exists()


def test_fresh_partner_cannot_run_frozen(tmp_path):
    config, _, _ = project(tmp_path, training=False)
    with pytest.raises(ConfigurationError, match="no executable weights"):
        prepare_v3(config)


def test_rule_cannot_be_selected_for_training(tmp_path):
    config, _, _ = project(tmp_path)
    config.training.groups = ("buffer",)
    with pytest.raises(ConfigurationError, match="rules cannot"):
        prepare_v3(config)


def test_independent_resource_override_and_shared_file(tmp_path):
    config, comp, target = project(tmp_path)
    comp["groups"]["machine_special"] = copy.deepcopy(comp["groups"]["machine"])
    comp["bindings"]["machine"]["overrides"] = {"machine_001": "machine_special"}
    target.write_text(yaml.safe_dump(comp))
    config.training.groups = ("machine", "machine_special")
    prepared = prepare_v3(config)
    policies = json.loads(prepared.policies_json)
    assert policies["machine"] == policies["machine_special"]
    bindings = json.loads(prepared.composition_json)["bindings"]
    assert bindings["machine"]["overrides"]["machine_001"] == "machine_special"
    comp["bindings"]["machine"]["overrides"]["missing"] = "machine"
    target.write_text(yaml.safe_dump(comp))
    with pytest.raises(ConfigurationError, match="unknown machine"):
        prepare_v3(config)


def test_cli_main_overrides_are_strict_and_nonmutating(tmp_path):
    config, _, _ = project(tmp_path)
    updated = apply_overrides(config, [("training.total_ticks", 128)])
    assert updated.training.total_ticks == 128
    assert config.training.total_ticks == 4096
    with pytest.raises(ConfigurationError):
        apply_overrides(config, [("training.total_steps", 128)])


def test_wrong_algorithm_network_and_central_partial_binding_fail(tmp_path):
    config, comp, target = project(tmp_path)
    config.training.algorithm = "dqn"
    config.training.parameters = str(ROOT / "configs/test/algorithms/dqn.yaml")
    with pytest.raises(ConfigurationError, match="network branches"):
        prepare_v3(config)
    comp["controller"] = {
        "policy": str(ROOT / "configs/test/policies/central_new_rllib.yaml")
    }
    target.write_text(yaml.safe_dump(comp))
    with pytest.raises(ConfigurationError, match="independent role"):
        prepare_v3(config)


def test_checkpoint_alias_freezes_and_old_contract_rejected(tmp_path):
    config, comp, target = project(tmp_path)
    scenario = prepare_v3(config).scenario
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text("{}")
    update = source / "checkpoints/update-000001/groups/original_machine"
    update.mkdir(parents=True)
    weights = b"metadata-only fixture"
    (update / "weights.pt").write_bytes(weights)
    encoder = b"{}"
    (update / "encoder.json").write_bytes(encoder)
    metadata = {
        "encoder_sha256": hashlib.sha256(encoder).hexdigest(),
        "role": "machine",
        "algorithm": "ppo",
        "backend": "rllib",
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "factory_identity": factory_identity(scenario.factory),
        "weights_file": "weights.pt",
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
    }
    (update / "model.json").write_text(json.dumps(metadata))
    alias = source / "checkpoints/best.json"
    alias.write_text(json.dumps({"checkpoint": "update-000001"}))
    policy = tmp_path / "machine.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.policy/v1",
                "role": "machine",
                "implementation": {
                    "kind": "model",
                    "model": {
                        "source": "source",
                        "checkpoint": "best",
                        "group": "original_machine",
                    },
                },
            }
        )
    )
    comp["groups"]["machine"]["policy"] = str(policy)
    target.write_text(yaml.safe_dump(comp))
    frozen = prepare_v3(config)
    alias.write_text(json.dumps({"checkpoint": "update-000099"}))
    assert json.loads(frozen.policies_json)["machine"]["resolved_model"][
        "source"
    ] == str(update)
    alias.write_text(json.dumps({"checkpoint": "update-000001"}))
    metadata["action_contract"] = "old"
    (update / "model.json").write_text(json.dumps(metadata))
    with pytest.raises(ConfigurationError, match="retraining"):
        prepare_v3(config)
