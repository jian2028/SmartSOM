"""Presentation of committed native progress without mixing physical units."""


def presentation(progress, *, task=None):
    row = dict(progress)
    tracking = row.get("tracking") or {}
    row["wandb_url"] = tracking.get("url")
    row["tracking_status"] = tracking.get("sync_status")
    if "seeds" in row:
        row["seed_details"] = [
            {
                key: seed.get(key)
                for key in (
                    "seed",
                    "status",
                    "episodes",
                    "episode_budget",
                    "physical_ticks",
                    "tick_budget",
                    "updates",
                    "tracking",
                )
            }
            for seed in row["seeds"]
        ]
    phase = row.get("phase") or task or "train"
    row.setdefault("stage", phase)
    if phase in {"train", "training"}:
        if "completed" in row and ("episode_tick" in row or "tick" in row):
            row["episode_deliveries"] = row["completed"]
        if "losses" in row:
            row["learner_diagnostics"] = {
                group: values["update_diagnostics"]
                for group, values in row["losses"].items()
                if isinstance(values, dict) and "update_diagnostics" in values
            }
        if row.get("episode_budget") is not None:
            completed, total, unit = (
                row.get("episodes", 0),
                row["episode_budget"],
                "episodes",
            )
        else:
            completed, total, unit = (
                row.get("physical_ticks", 0),
                row.get("tick_budget"),
                "physical ticks",
            )
        row["committed_physical_ticks"] = row.get("physical_ticks", 0)
        row["inflight_episode_tick"] = row.get("episode_tick", 0)
    elif phase in {"evaluate", "evaluation", "validation", "calibrate", "calibration"}:
        if "episode_tick" in row:
            row["case_tick"] = row.pop("episode_tick")
        elif "tick" in row:
            row["case_tick"] = row.pop("tick")
        completed, total, unit = (
            row.get("completed_cases", 0),
            row.get("total_cases"),
            "cases",
        )
    else:
        completed, total, unit = (
            row.get("completed"),
            row.get("total"),
            row.get("unit", "tasks"),
        )
    row["completed"] = completed
    row["progress_completed"] = completed
    return row, total, unit
