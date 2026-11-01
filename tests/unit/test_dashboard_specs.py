"""Cloud-independent chart contracts and metadata boundaries."""

from smartsom.telemetry.dashboard_specs import comparison_spec, entity_spec
from smartsom.telemetry.experiment_tracking import entity_values


def test_summary_chart_uses_sample_variance_and_keeps_missing_seeds_missing():
    spec = comparison_spec(["world2", "world1"])
    transforms = spec["transform"]
    aggregates = [item for item in transforms if "aggregate" in item]
    assert aggregates[0]["groupby"] == ["series", "seed", "episode"]
    ops = {item["as"]: item["op"] for item in aggregates[1]["aggregate"]}
    assert ops["sample_sd"] == "stdev"
    assert ops["actual_seeds"] == "valid"
    assert any("actual_seeds > 1" in item.get("calculate", "") for item in transforms)
    assert spec["facet"]["column"]["field"] == "coverage"
    assert spec["params"][1]["bind"]["options"] == ["all", "world1", "world2"]


def test_machine_denominator_is_elapsed_time_not_sum_of_overlapping_vehicle_states():
    assert entity_values(
        "machines", {"processing": 6, "down": 2, "idle": 1, "blocked": 1}
    ) == {
        "processing": 6,
        "down": 2,
        "idle": 1,
        "blocked": 1,
        "observed_ticks": 10,
        "available_ticks": 8,
    }
    vehicle = entity_values(
        "vehicles",
        {"loaded": 8, "empty": 2, "moving": 4, "stationary": 6, "service_at_start": 3},
    )
    assert vehicle["observed_ticks"] == 10
    spec = entity_spec(
        ["m0", "m1"], ["processing", "observed_ticks", "available_ticks"], "machines"
    )
    assert spec["params"][0]["value"] == "processing_fraction"
    assert spec["layer"][1]["encoding"]["opacity"]["value"] == 0
    assert "world" in spec["layer"][0]["transform"][0]["groupby"]


def test_research_evidence_keeps_recorded_values_and_marks_absence(tmp_path):
    import json

    from smartsom.telemetry.wandb_dashboard import research_evidence

    (tmp_path / "research-summary.json").write_text(
        json.dumps(
            {"selection": "sample", "robustness": [{"ratio": 0.8, "status": "valid"}]}
        )
    )
    rows = [{"root": str(tmp_path), "status": {"online_run_id": "abc"}, "events": []}]
    result = research_evidence(rows, "sample")
    assert result["robustness"][0]["result"][0]["ratio"] == 0.8
    assert result["CAR"]["status"] == "not_recorded"
    assert research_evidence(rows, "argmax")["robustness"]["status"] == "not_recorded"


def test_car_requires_paired_identity_and_preserves_censoring(tmp_path):
    import json

    from smartsom.telemetry.wandb_dashboard import car_statistics

    records = []
    for obs, budget in (("base", 100), ("sl", 50)):
        root = tmp_path / obs
        root.mkdir()
        outcome = {
            "status": "reached",
            "first_qualifying_budget": budget,
            "protocol_sha256": "same",
            "budget_unit": "episodes",
            "panel_identity": {
                "train_seed": 1,
                "decoder": "sample",
                "panel_sha256": "x",
            },
        }
        (root / "learning-target.json").write_text(
            json.dumps({"modes": {"sample": outcome}})
        )
        records.append(
            {
                "root": str(root),
                "status": {"comparison": {"method": "ppo", "observation": obs}},
                "events": [
                    {
                        "selection": "sample",
                        "evaluation_identity": {
                            "sl_comparison": {"non_sl_sha256": "same"}
                        },
                    }
                ],
            }
        )
    assert car_statistics(records, "sample")["comparisons"][0]["ratio"] == 2
    records[0]["events"] = []
    assert car_statistics(records, "sample")["comparisons"][0]["ratio"] is None


def test_statistics_separates_method_identity_for_same_or_distinct_seeds(tmp_path):
    from smartsom.telemetry.wandb_dashboard import _statistics, _table

    for second_seed in (1, 2):
        records = []
        for identity, seed, cost in [("one", 1, 10), ("two", second_seed, 90)]:
            records.append(
                {
                    "root": str(tmp_path),
                    "manifest": {"seed": seed},
                    "status": {
                        "comparison": {"variant": "base", "method_identity": identity}
                    },
                    "events": [
                        {
                            "kind": "summary",
                            "phase": "validation",
                            "selection": "sample",
                            "training_progress": {"episodes": 3},
                            "methods": {
                                "configured": {
                                    "all": {
                                        "metrics": {"joint_cost": cost},
                                        "complete_inventory": True,
                                    }
                                }
                            },
                        }
                    ],
                }
            )
        rows = [
            r for r in _statistics(records, "sample") if r["metric"] == "joint_cost"
        ]
        assert {(r["method_identity"], r["mean"], r["actual_seeds"]) for r in rows} == {
            ("one", 10, 1),
            ("two", 90, 1),
        }
        assert "[one]" in _table(rows) and "[two]" in _table(rows)


def test_rule_identity_ignores_learning_but_preserves_actions_and_mover(tmp_path):
    import yaml

    from smartsom.telemetry.tracking_metadata import metadata, method_identity
    from tests.helpers.algorithm import algorithm_options

    (tmp_path / "inputs").mkdir()
    identities = []
    for arm in ("base", "all"):
        config = algorithm_options(
            schema="smartsom.base-algorithm/v1", social_arm=arm
        ).model_dump(mode="json", by_alias=True)
        (tmp_path / "inputs/algorithm.yaml").write_text(yaml.safe_dump(config))
        identities.append(metadata(tmp_path, {})["comparison"])
    assert identities[0]["method_identity"] != identities[1]["method_identity"]
    assert method_identity(identities[0], "rule") == method_identity(
        identities[1], "rule"
    )
    config["decision_protocol"]["pre"]["accept"]["allow_stop"] = False
    (tmp_path / "inputs/algorithm.yaml").write_text(yaml.safe_dump(config))
    changed = metadata(tmp_path, {})["comparison"]
    assert method_identity(changed, "rule") != method_identity(identities[0], "rule")
