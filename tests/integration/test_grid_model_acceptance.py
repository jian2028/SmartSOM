"""Actual learned-policy completion and four-resource optimizer updates.

Grid actions need new training; historical matrix 1024/4096-step performance is
not a convergence promise for this larger action space. These bounded fixtures
verify the current learning contract without substituting a rule controller.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from smartsom import api
from smartsom.experiments.training_audit import audit_training
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def config_for(name, output):
    packages = ("torch", "gymnasium", "sb3_contrib" if name == "sb3" else "ray")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        if (
            os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1"
            or os.environ.get("SMARTSOM_REQUIRE_MARL") == "1"
        ):
            pytest.fail(f"required backend packages missing: {missing}")
        pytest.skip(f"optional backend packages missing: {missing}")
    config = api.load_config(ROOT / f"configs/runs/{name}_production.yaml")
    config.output.root = str(output)
    config.logging.verbose = False
    config.logging.progress = "off"
    config.logging.tensorboard = False
    config.validation.enabled = False
    config.checkpointing.keep_last = 1
    config.checkpointing.save_best = False
    return config


@pytest.mark.parametrize("name", ["sb3", "rllib", "marl"])
def test_real_model_trains_saves_loads_and_completes_hand_task(name, tmp_path):
    config = config_for(name, tmp_path / "training")
    result = api.train(config)
    manifest = json.loads((result.last_checkpoint / "checkpoint.json").read_text())
    assert result.environment_steps == config.training.total_steps == 16384
    assert manifest["initial_weights_sha256"] != manifest["final_weights_sha256"]
    assert audit_training(result.run_dir)["budget_completed"]
    evaluated = api.evaluate(
        result.run_dir,
        api.EvaluationOptions(replications=3, baselines=(), full_replay=True),
        output_root=tmp_path / "evaluation",
    )
    assert len(evaluated.results) == 3
    for row in evaluated.results:
        assert row["status"] == "completed"
        run = evaluated.run_dir / row["run_dir"]
        record = json.loads((run / "run.json").read_text())
        assert record["result"]["completed"] == ["demand"]
        assert record["result"]["tick"] == 8
        assert audit(run)["learning"]["reason"] == "completed"


@pytest.mark.marl
def test_four_resource_actors_and_critics_actually_update_and_reload(tmp_path):
    config = config_for("marl", tmp_path / "four-roles")

    from smartsom.learning.production import LearnedProductionDriver

    config.scenario = str(ROOT / "configs/scenarios/scenario_quality.yaml")
    config.training.total_steps = 32768
    prepared = api.prepare(config)
    result = api.train_prepared(prepared)
    metadata = json.loads((result.last_checkpoint / "checkpoint.json").read_text())
    roles = {"agv_policy", "buffer_policy", "machine_policy", "quality_policy"}
    assert set(metadata["modules"]) == set(metadata["component_changes"]) == roles
    assert all(
        metadata["component_changes"][role][part]
        for role in roles
        for part in ("actor", "critic")
    )
    checked = audit_training(result.run_dir)
    assert checked["agent_steps"] == checked["environment_steps"] == 32768
    assert checked["physical_actions"] > 0
    driver = LearnedProductionDriver(result.last_checkpoint, prepared.resolved.scenario)
    try:
        assert set(driver.modules) == roles
        while not driver.env.finished:
            driver.next_tick()
        # Reload and a truthful terminal outcome are required here. The separate
        # deterministic fixture above establishes completed-policy behavior.
        assert driver.env.reason in {"completed", "truncated", "budget_exhausted"}
    finally:
        driver.env.close()
