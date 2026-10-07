"""Historical packages cannot bypass V3.1 guards through source evaluation."""

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v3 import model_location
from smartsom.config.policies import ModelSelector
from smartsom.domain.production_decisions import ACTION_CONTRACT, OBSERVATION_CONTRACT
from smartsom.domain.travel_time import physical_contract, validate_model_contract
from smartsom.experiments import composable

ROOT = Path(__file__).resolve().parents[2]


def prepared():
    return api.prepare(
        api.load_config(ROOT / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )


def metadata(role, mismatch, scenario):
    value = {
        "role": role,
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "physical_contract": physical_contract(scenario),
    }
    if mismatch == "action":
        value["action_contract"] = "smartsom.production-actions/v3"
    elif mismatch == "observation":
        value["observation_contract"] = "smartsom.production-observations/v3"
    else:
        del value["physical_contract"]["dispatch_semantics"]
    return value


@pytest.mark.parametrize("role", ["machine", "buffer", "dispatcher", "central"])
@pytest.mark.parametrize("mismatch", ["action", "observation", "physical"])
@pytest.mark.parametrize("archive", [False, True])
def test_metadata_and_direct_read_reject_before_weight_access(
    tmp_path, role, mismatch, archive
):
    payload = metadata(role, mismatch, prepared().scenario)
    if archive:
        source = tmp_path / "metadata-only.zip"
        with zipfile.ZipFile(source, "w") as stream:
            stream.writestr("model.json", json.dumps(payload))
    else:
        source = tmp_path / "metadata-only"
        source.mkdir()
        (source / "model.json").write_text(json.dumps(payload))
    # Deliberately no weights: compatibility rejection must happen first.
    with pytest.raises(ValueError, match="incompatible; retraining"):
        model_location(ModelSelector(source=str(source)))
    pytest.importorskip("torch")
    from smartsom.learning.production_inference import read_package

    with pytest.raises(ValueError, match="incompatible; retraining"):
        read_package(source)


@pytest.mark.parametrize("role", ["machine", "central"])
@pytest.mark.parametrize("mismatch", ["action", "observation", "physical"])
def test_public_source_evaluation_rejects_before_allocating(
    tmp_path, monkeypatch, role, mismatch
):
    frozen = prepared()
    declaration = {"role": role, "implementation": {"kind": "new"}}
    frozen = replace(
        frozen,
        policies_json=canonical_json({role: declaration}),
        composition_json=canonical_json(
            {"controller": {"policy": "central"}} if role == "central" else {}
        ),
    )
    old = metadata(role, mismatch, frozen.scenario)
    monkeypatch.setattr(composable, "prepared_from_run", lambda source: frozen)
    monkeypatch.setattr(
        composable,
        "checkpoint_path",
        lambda source, selection: tmp_path / "metadata-only",
    )
    monkeypatch.setattr(
        "smartsom.config.experiment_v3.model_location",
        lambda selector: {"source": "unused", "metadata": old},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible source must not allocate output")

    monkeypatch.setattr(composable, "allocate", forbidden)
    source = tmp_path / "historical-run"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"schema": "smartsom.experiment/v3"}))
    with pytest.raises(ValueError, match="incompatible; retraining"):
        api.evaluate(source=source, output_root=tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("mismatch", ["action", "observation", "physical"])
def test_model_group_loader_rejects_frozen_metadata_before_reader(
    monkeypatch, mismatch
):
    pytest.importorskip("torch")
    from smartsom.learning import production_inference

    frozen = prepared()
    declarations = {
        "machine": {
            "role": "machine",
            "implementation": {"kind": "model"},
            "resolved_model": {
                "source": "unused",
                "metadata": metadata("machine", mismatch, frozen.scenario),
            },
        }
    }
    frozen = replace(frozen, policies_json=canonical_json(declarations))
    monkeypatch.setattr(
        production_inference,
        "read_package",
        lambda source: (_ for _ in ()).throw(AssertionError("reader must not run")),
    )
    with pytest.raises(ValueError, match="incompatible; retraining"):
        production_inference.build_groups(frozen, training=False)


def test_current_contract_accepts_same_provider_and_rejects_changed_physics():
    scenario = prepared().scenario
    current = metadata("machine", "action", scenario)
    current["action_contract"] = ACTION_CONTRACT
    validate_model_contract(current, scenario)
    current["physical_contract"]["processing_rounding"] = (
        "half_up" if scenario.processing_rounding == "ceil" else "ceil"
    )
    with pytest.raises(ValueError, match="transport/processing"):
        validate_model_contract(current, scenario)
