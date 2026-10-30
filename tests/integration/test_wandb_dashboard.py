"""Validate the actual workspace SDK without authenticating or writing to W&B."""

import importlib.util
import json
import os
from pathlib import Path
from shutil import copyfile

import pytest

if importlib.util.find_spec("wandb_workspaces") is None:
    if os.environ.get("SMARTSOM_REQUIRE_TRACKING") == "1":
        raise RuntimeError("dashboard CI requires wandb_workspaces")
    pytest.skip("optional dashboard SDK is not installed", allow_module_level=True)

from smartsom.config.base_diagnostic import TrackingOptions
from smartsom.telemetry.experiment_tracking import ExperimentTracking
from smartsom.telemetry.wandb_dashboard import build_views, create_dashboard


def seed_run(root, seed, cost, modes=("sample", "argmax")):
    run = root / f"seed-{seed}"
    (run / "inputs").mkdir(parents=True)
    configs = Path(__file__).resolve().parents[2] / "configs"
    for kind, name in (
        ("factory", "factories/small"),
        ("algorithm", "algorithms/base-ppo"),
    ):
        copyfile(configs / (name + ".yaml"), run / "inputs" / (kind + ".yaml"))
    (run / "inputs/experiment.yaml").write_text("train_seeds: [101, 102, 103]\n")
    manifest = {
        "run_id": f"local-{seed}",
        "seed": seed,
        "source": {"commit": "test", "implementation_sha256": "fixed"},
        "inputs": {"factory": "factory", "workload": "workload"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    tracker = ExperimentTracking(
        run, TrackingOptions(mode="disabled"), job_type="train"
    )
    tracker.start(manifest, {"train_seeds": [101, 102, 103]})
    for mode in modes:
        tracker.log(
            {
                "phase": "validation",
                "selection": mode,
                "kind": "summary",
                "training_progress": {"episodes": 10},
                "methods": {
                    "ppo": {
                        "all": {
                            "metrics": {
                                "delivery_fraction": 1,
                                "drained": 1,
                                "joint_cost": cost,
                                "qualified_coverage": 1,
                                "mean_tardiness": cost,
                                "final_passing_rate": 1,
                            },
                            "actual_cases": 4,
                            "planned_cases": 4,
                            "complete_inventory": True,
                        }
                    }
                },
            }
        )
    tracker.close()
    path = run / "tracking/status.json"
    state = json.loads(path.read_text())
    state.update(
        online_run_id=f"cloud-{seed}", entity="engineering", project="SmartSOM"
    )
    path.write_text(json.dumps(state))
    return run


def test_real_sdk_separates_modes_and_preserves_exact_partial_statistics(tmp_path):
    seed_run(tmp_path, 101, 1)
    seed_run(tmp_path, 102, 3)
    views, definitions = build_views(tmp_path)
    assert len(views) == 2 and definitions
    for view in views:
        model = view._to_model()
        assert model is not None
        assert len(view.sections[0].panels) == 6
        stats = view.sections[2].panels[0].markdown
        assert "1.41421" in stats and "2/3" in stats and "provisional" in stats
        assert "cloud-101" in str(view.runset_settings.filters)
        for panel in view.sections[0].panels:
            assert panel.query == {
                "summaryTable": {"tableKey": "research/summary_history"}
            }
    assert any(
        item["op"] == "stdev"
        for definition in definitions.values()
        for transform in definition["transform"]
        for item in transform.get("aggregate", [])
    )


def test_explicit_creation_always_builds_new_private_presets_and_views(
    tmp_path, monkeypatch
):
    import wandb
    import wandb_workspaces.workspaces as ws

    import smartsom.telemetry.wandb_dashboard as dashboard

    seed_run(tmp_path, 101, 1, modes=("sample",))
    monkeypatch.setattr(dashboard, "resolve_run", lambda source, roots: tmp_path)
    presets, saves = [], []

    class API:
        def create_custom_chart(self, **kwargs):
            presets.append(kwargs)
            return "engineering/" + kwargs["name"]

    monkeypatch.setattr(wandb, "Api", API)

    def save(view):
        saves.append(view)

    monkeypatch.setattr(ws.Workspace, "save", save)
    monkeypatch.setattr(
        ws.Workspace, "url", property(lambda self: "https://wandb.ai/example/new-view")
    )
    reads = []

    def load(cls, url):
        reads.append(url)
        return saves[-1]

    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(load))
    assert create_dashboard("unused")["views"][0]["verified"]
    create_dashboard("unused")
    assert len(reads) == 2
    assert len(saves) == 2 and saves[0] is not saves[1]
    assert presets[0]["name"] != presets[1]["name"]
    assert all(preset["access"] == "private" for preset in presets)

    def missing_sections(cls, url):
        from types import SimpleNamespace

        return SimpleNamespace(sections=[])

    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(missing_sections))
    with pytest.raises(RuntimeError, match="did not preserve"):
        create_dashboard("unused")


def test_real_offline_sdk_serializes_only_new_table_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="offline"), job_type="train"
    )
    tracker.start(
        {
            "seed": 101,
            "source": {"commit": "engineering", "implementation_sha256": "fixed"},
            "inputs": {},
        },
        {"train_seeds": [101, 102]},
    )
    for episode in (1, 2):
        tracker.log(
            {
                "phase": "validation",
                "kind": "summary",
                "selection": "sample",
                "training_progress": {"episodes": episode},
                "methods": {
                    "ppo": {
                        "all": {
                            "metrics": {"joint_cost": float(episode)},
                            "actual_cases": 4,
                            "planned_cases": 4,
                            "complete_inventory": True,
                        }
                    }
                },
            }
        )
    tracker.close()
    tables = [
        json.loads(path.read_text())
        for path in (tmp_path / "tracking").rglob("*.table.json")
    ]
    rows = [
        table["data"] for table in tables if "joint_cost" in table.get("columns", [])
    ]
    assert len(rows) == 2
    assert all(len(batch) == 1 for batch in rows)
