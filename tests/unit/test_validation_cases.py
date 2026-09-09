"""External validation scenarios freeze identities without simulator execution."""

import json
import shutil

import pytest

from smartsom.api import load_config, load_preset, show_config
from smartsom.config.codec import ConfigurationError, canonical_json
from smartsom.config.experiment import PRESET_ROOT, prepare, prepare_frozen
from smartsom.config.validation import read_validation_inputs


def test_explicit_cases_materialize_once_and_survive_source_removal(tmp_path):
    inputs = tmp_path / "inputs"
    shutil.copytree(PRESET_ROOT, inputs)
    config = load_config(inputs / "configs/runs/learning_sb3.yaml")
    config.validation.enabled = True
    config.validation.replications = 2
    config.validation.scenarios = (config.scenario,)
    prepared = prepare(config)
    rows = read_validation_inputs(prepared.validation_json)
    assert len(rows) == 2 and rows[0]["input_id"] != rows[1]["input_id"]
    assert rows[0]["episode"].factory == prepared.resolved.base.factory
    shutil.rmtree(inputs)
    config.algorithm.learning_rate = 0.0005
    candidate = prepare_frozen(config, prepared)
    assert candidate.validation_json == prepared.validation_json
    assert candidate.scientific_sha256 != prepared.scientific_sha256
    config.validation.seed = 999
    with pytest.raises(ConfigurationError, match="retain"):
        prepare_frozen(config, prepared)


def test_validation_snapshot_tampering_is_rejected():
    config = load_preset("marl_micro")
    config.validation.scenarios = (config.scenario,)
    rows = json.loads(prepare(config).validation_json)
    rows[0]["world_seed"] += 1
    with pytest.raises(ConfigurationError, match="identity"):
        read_validation_inputs(canonical_json(rows))


def test_duplicate_or_incompatible_validation_cases_fail_before_training():
    config = load_preset("sb3_micro")
    config.validation.scenarios = (config.scenario, config.scenario)
    with pytest.raises(ConfigurationError, match="duplicate"):
        prepare(config)
    config.validation.scenarios = (load_preset("competition").scenario,)
    with pytest.raises(ConfigurationError, match="incompatible"):
        prepare(config)


def test_preview_counts_validation_and_evaluation_cases():
    config = load_preset("marl_micro")
    config.validation.scenarios = (config.scenario,)
    config.evaluation.scenarios = (config.scenario, config.scenario)
    config.evaluation.baselines = ("spt",)
    result = show_config(config)
    assert result["validation"]["frozen_external_inputs"]
    assert result["validation"]["inputs_per_validation"] == 5
    assert result["evaluation"]["runs"] == 20


def test_validation_source_locations_do_not_change_scientific_identity(tmp_path):
    prepared = []
    for name in ("one", "two"):
        root = tmp_path / name
        shutil.copytree(PRESET_ROOT, root)
        config = load_config(root / "configs/runs/learning_sb3.yaml")
        config.validation.enabled = True
        config.validation.scenarios = (config.scenario,)
        prepared.append(prepare(config))
    assert prepared[0].validation_json != prepared[1].validation_json
    assert prepared[0].scientific_sha256 == prepared[1].scientific_sha256
