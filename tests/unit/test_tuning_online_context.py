"""Online continuation keeps layouts frozen and refreshes machine evidence."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from smartsom import api
from smartsom.config.codec import digest
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments import tuning_batch as batch
from smartsom.experiments.evidence import write_json
from smartsom.experiments.performance_profiles import hardware_shape
from smartsom.experiments.tuning_resources import GIB, ResourceSnapshot


def test_online_resume_refreshes_hardware_without_recalibrating(tmp_path, monkeypatch):
    prepared = prepare_v3(
        api.load_config(
            Path(__file__).resolve().parents[2]
            / "configs/test/runs/train_machine_ppo.yaml"
        ),
        training=True,
    )
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared), "control_spec": {}},),
        active_limit=0,
        calibration_level="online",
        output_root=str(tmp_path),
        provenance={"kind": "author-plan"},
    )
    root, plan, state = batch.allocate_batch(inputs)
    before = ResourceSnapshot(16, 64 * GIB, 64 * GIB, external_cpu_load=0)
    after = ResourceSnapshot(4, 32 * GIB, 32 * GIB, external_cpu_load=0)
    saved = batch.calibrate(
        root, plan, monitor=SimpleNamespace(snapshot=lambda **_kwargs: before)
    )
    groups = batch._groups(plan["entries"], online=True)[1]
    saved["frozen_plan_sha256"] = digest(plan)
    saved["eligible_ids"] = sorted(groups)
    for context in saved["online_context"].values():
        context["hint"] = {"concurrency": 4, "throughput": 100}
    write_json(root / "calibration.json", saved)
    monkeypatch.setattr(
        "smartsom.experiments.tuning_resources.ResourceMonitor",
        lambda: SimpleNamespace(snapshot=lambda **_kwargs: after),
    )
    monkeypatch.setattr(
        batch,
        "calibrate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen online layout was recalibrated")
        ),
    )
    result = batch._execute_batch(root, plan, state, recommend_only=True)
    current = json.loads((root / "calibration.json").read_text())
    assert result["recommendations"] == saved["recommendations"]
    assert current["groups"] == saved["groups"]
    assert hardware_shape(before, mode="balanced") != hardware_shape(
        after, mode="balanced"
    )
    for context in current["online_context"].values():
        assert context["hardware"] == hardware_shape(after, mode="balanced")
        assert context["hint"] is None
