"""A real one-environment lazy pool retains complete process ownership.

Four physical ticks exercise native updates and final evaluation. This is a
bounded engineering check, not a training/performance experiment.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import prepare
from smartsom.experiments.composable import allocate
from smartsom.experiments.tuning_session import AdaptiveSession, verify_commit

pytest.importorskip("torch")
psutil = pytest.importorskip("psutil")

pytestmark = pytest.mark.learning
ROOT = Path(__file__).resolve().parents[2]


def test_real_lazy_sampler_ownership_is_complete_before_all_slots_spawn(tmp_path):
    config = api.load_config(ROOT / "configs/test/runs/train_all_ppo.yaml")
    config.training.total_ticks = 4
    config.training.ticks_per_update = 1
    config.training.record_initial = False
    config.validation.enabled = False
    config.evaluation.replications = 1
    config.evaluation.checkpoint = "last"
    config.output.root = str(tmp_path / "runs")
    config.scenario_overrides["tick_limit"] = 4
    config.runtime.numerical_threads = 1
    config.runtime.num_envs = 1
    config.runtime.sampling_processes = 2
    frozen = prepare(config)
    parameters = json.loads(frozen.parameters_json)
    parameters.update(batch_size=2, n_epochs=1)
    frozen = replace(frozen, parameters_json=canonical_json(parameters))
    root, record, frozen = allocate(frozen, "training")
    record["tuning"] = {"experiment_id": "lazy-pool-ownership"}
    session = AdaptiveSession(frozen, root, record)
    worker_ids = set()
    try:
        initial = json.loads((root / "tuning-runtime.json").read_text())
        assert not initial["children_complete"]
        assert session.session.executor._processes == {}
        first = session.step()
        assert not first["done"] and first["physical_ticks"] == 1
        pool = session.session.executor._processes
        # Exactly one submit occurred: the configured second worker need not
        # exist for this safe committed update to have a complete registry.
        assert len(pool) == 1 < config.runtime.sampling_processes
        worker_ids.update(pool)
        runtime = json.loads((root / "tuning-runtime.json").read_text())
        assert runtime["phase"] == "committed" and runtime["children_complete"]
        registered = {p["pid"]: p["create_time"] for p in runtime["child_processes"]}
        for pid in pool:
            assert registered[pid] == psutil.Process(pid).create_time()
        while True:
            result = session.step()
            worker_ids.update(session.session.executor._processes)
            if result["done"]:
                break
        assert result["updates"] == result["physical_ticks"] == 4
        assert verify_commit(session.last_commit)["phase"] == "experiment_complete"
        assert (root / "evaluation/tuning-final.json").is_file()
        final = json.loads((root / "tuning-runtime.json").read_text())
        assert final["children_complete"]
        assert worker_ids <= {p["pid"] for p in final["child_processes"]}
    finally:
        session.close()
    closed = json.loads((root / "tuning-runtime.json").read_text())
    assert closed["phase"] == "closed" and closed["children_complete"]
    assert worker_ids <= {p["pid"] for p in closed["child_processes"]}
    assert all(not psutil.pid_exists(pid) for pid in worker_ids)
