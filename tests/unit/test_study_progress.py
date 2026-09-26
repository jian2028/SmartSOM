"""Overall study progress includes expensive evaluation and paired controls."""

import json

import pytest

from smartsom.telemetry.study_progress import StudyWork, case_work


def plan():
    return {
        "recipe": {
            "total_ticks": 100,
            "ticks_per_update": 10,
            "max_ticks": 10,
            "validation_every_updates": 2,
            "validation_cases": 2,
            "evaluation_cases": 2,
            "controls": ["initial", "rule", "random"],
        },
        "entries": [
            {
                "id": a,
                "algorithm": a,
                "H_case": "h0",
                "V_case": "low",
                "transport": "auto",
            }
            for a in ("ppo", "dqn")
        ],
    }


def row(ticks=0, stage="sampling", **values):
    return {"stage": stage, "values": {"physical_ticks": ticks, **values}}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def tracker(tmp_path):
    work = StudyWork(tmp_path, plan())
    work.started = 0
    return work


def test_budget_includes_validation_test_initial_and_unique_shared_controls(tmp_path):
    work = tracker(tmp_path)
    # Each entry: 100 training + 100 validation + 20 test + 20 initial.
    # The shared rule/random cases cost 40 in total, not 80.
    assert work.entry_total == 240 and work.shared_total == 40 and work.total == 520
    tasks = {
        a: row(
            20,
            "validation",
            updates=2,
            validation_finished=0,
            validation_requested=2,
            validation_tick=5,
            validation_tick_limit=10,
            validation_case_active=True,
        )
        for a in ("ppo", "dqn")
    }
    overview = work.overview(tasks, {"entries": {}}, now=0)
    assert overview["work_completed"] == 50
    assert overview["training_completed"] == 40
    assert overview["eta_seconds"] is None


def test_ended_case_does_not_also_count_as_next_active_case():
    values = {
        "validation_finished": 1,
        "validation_requested": 2,
        "validation_tick": 2,
        "validation_tick_limit": 10,
        "validation_case_active": False,
    }
    assert case_work(values, "validation", 10) == 10
    values.update(validation_case_active=True, validation_tick=0)
    assert case_work(values, "validation", 10) == 10
    values.update(validation_tick=5)
    assert case_work(values, "validation", 10) == 15
    values.update(validation_finished=2)
    assert case_work(values, "validation", 10) == 20


def test_saved_validation_is_not_added_twice(tmp_path):
    work = tracker(tmp_path)
    source = tmp_path / "experiment"
    write(source / "logs/validation-000002.json", [{}, {}])
    tasks = {
        "ppo": row(
            20,
            "validation",
            updates=2,
            training_run_directory=str(source),
            validation_finished=2,
            validation_requested=2,
            validation_tick=10,
            validation_tick_limit=10,
        )
    }
    assert work.overview(tasks, {"entries": {}}, now=0)["work_completed"] == 40
    tasks["ppo"] = row(
        40,
        "validation",
        updates=4,
        training_run_directory=str(source),
        validation_finished=1,
        validation_requested=2,
        validation_tick=5,
        validation_tick_limit=10,
        validation_case_active=True,
    )
    assert work.overview(tasks, {"entries": {}}, now=5)["work_completed"] == 75


def test_shared_partial_control_is_counted_once_across_partners(tmp_path):
    work = tracker(tmp_path)
    tasks = {
        a: row(
            100,
            "evaluation",
            evaluation_kind="rule control",
            evaluation_finished=1,
            evaluation_requested=2,
            evaluation_tick=5,
            evaluation_tick_limit=10,
            evaluation_case_active=True,
        )
        for a in ("ppo", "dqn")
    }
    assert work.overview(tasks, {"entries": {}}, now=0)["work_completed"] == 215
    write(tmp_path / "controls/h0_low_auto_rule.json", [{}, {}])
    assert work.overview(tasks, {"entries": {}}, now=5)["work_completed"] == 220


def test_model_test_credit_survives_switch_to_initial_control(tmp_path):
    work = tracker(tmp_path)
    tested = tmp_path / "tested"
    write(tested / "run.json", {"results": [{}, {}]})
    state = {"entries": {"ppo": {"evaluation_dir": str(tested)}}}
    tasks = {
        "ppo": row(
            100,
            "evaluation",
            evaluation_kind="initial control",
            evaluation_finished=1,
            evaluation_requested=2,
            evaluation_tick=0,
            evaluation_tick_limit=10,
            evaluation_case_active=True,
        )
    }
    assert work.overview(tasks, state, now=0)["work_completed"] == 130


def test_eta_uses_new_work_and_wall_time_not_cached_resume_progress(tmp_path):
    work = tracker(tmp_path)
    state = {"entries": {}}
    work.overview({"ppo": row(30)}, state, now=0)
    assert work.overview({"ppo": row(30)}, state, now=30)["eta_seconds"] is None
    # Restart estimator for a clean, monotonically increasing clock.
    work = tracker(tmp_path)
    work.overview({"ppo": row(30)}, state, now=0)
    result = work.overview({"ppo": row(60)}, state, now=30)
    assert result["elapsed_seconds"] == 30
    assert result["eta_seconds"] == pytest.approx(460)


def test_completed_and_failed_results_are_distinct(tmp_path):
    work = tracker(tmp_path)
    state = {"entries": {"ppo": {"status": "failed"}, "dqn": {"status": "completed"}}}
    partial = work.overview({"ppo": row(10)}, state, now=0)
    assert (
        partial["work_completed"] == 250
        and partial["work_completed"] < partial["work_total"]
    )
    for kind in ("rule", "random"):
        write(tmp_path / f"controls/h0_low_auto_{kind}.json", [{}, {}])
    state["entries"]["ppo"]["status"] = "completed"
    finished = work.overview({}, state, now=30)
    assert finished["work_completed"] == finished["work_total"] == 520
    assert finished["eta_seconds"] == 0
