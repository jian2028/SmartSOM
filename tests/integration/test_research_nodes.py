"""Real complete-update boundaries at exact physical ticks, including restart."""

import json
from types import SimpleNamespace

import pytest

from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation_session import ReservationSession
from tests.unit.test_algorithm_components import author, recipe


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_exact_physical_nodes_resume_without_repeating_updates(tmp_path, method):
    path = author(tmp_path, recipe(method))
    plan = compile_reservation(
        path,
        sets=[
            "experiment.drain_frozen=true",
            "experiment.train_ticks=19",
            "experiment.validation={unit: physical_ticks, milestones: [0, 7, 19], modes: [argmax]}",
            "experiment.tracking.mode=disabled",
            "workload.orders=4",
            "workload.initial_orders=2",
            f"algorithm.production.policies.{method}.learner.params.epochs=1",
        ],
    )
    frozen = tmp_path / "inputs"
    frozen.mkdir()
    plan.materialize(frozen)
    args = SimpleNamespace(
        experiment=frozen / "experiment.yaml",
        output=tmp_path / "run",
        seed=101,
        seconds=300,
        ticks=19,
        calibration=None,
        numerical_threads=1,
    )
    # Exercise actual training boundaries, replacing only the expensive panel observer.
    observed = []
    original = ReservationSession.validate

    def observe(self):
        if self.validation_node in self.ex.validation.milestones:
            marker = json.loads((self.checkpoints / "current.json").read_text())
            state = json.loads(
                (self.checkpoints / marker["generation"] / "state.json").read_text()
            )
            assert state["ticks"] == self.ticks
            observed.append((self.ticks, self.policy.updates))

    from unittest.mock import patch

    with patch.object(ReservationSession, "validate", observe):
        session = ReservationSession(args)
        session.step()
        assert session.ticks == 7
        assert session.rows[-1]["status"] != "completed"
        updates = session.policy.updates
        session.close()
        resumed = ReservationSession(args)
        assert resumed.ticks == 7 and resumed.policy.updates == updates
        assert resumed.rollout_limit() == 12
        result = resumed.step()
        assert result["done"] and resumed.ticks == 19
        assert [r["tick"] for r in resumed.rows] == [7, 12]
        resumed.close()
    assert [t for t, _ in observed] == [0, 7, 7, 19]
    assert callable(original)


def test_public_study_models_panels_report_and_existing_result_guard(tmp_path):
    import yaml

    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.experiments.launch import compile_launch, run_launch
    from tests.unit.test_research_packages import case_file

    folder = tmp_path / "experiments/network/01-check/shared"
    folder.mkdir(parents=True)
    path = author(folder, recipe("ppo"))
    document = yaml.safe_load(path.read_text())
    source = __import__("pathlib").Path(document["workload"])
    workload = yaml.safe_load(source.read_text())
    workload.update(
        orders=4,
        initial_orders=2,
        mean_interarrival=2,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    (folder / "workload.yaml").write_text(yaml.safe_dump(workload))
    document.update(
        workload="workload.yaml",
        train_seeds=2,
        train_ticks=19,
        drain_frozen=True,
        final_evaluation=True,
        validation={
            "unit": "physical_ticks",
            "milestones": [0, 7, 19],
            "modes": ["argmax"],
        },
        run={"task": "train", "seconds": 180},
        display={"progress": "off"},
    )
    path.write_text(yaml.safe_dump(document))
    _, factory, workload, _, _ = load_inputs(path)
    for split, seed in (("validation", 31), ("test", 32)):
        case_file(folder, split, factory, workload, seed)
    document["case_sets"] = {"validation": "validation.json", "test": "test.json"}
    path.write_text(yaml.safe_dump(document))
    plan = compile_launch(folder.parent)
    result = run_launch(plan)
    assert result["status"] == "completed", result
    root = plan.output_root
    assert (root / "report/report.md").is_file()
    for seed in (101, 102):
        trial = root / "trials/shared" / f"seed-{seed}"
        assert (trial / "last.pt").is_file()
        assert (trial / "source/uv.lock").is_file()
        assert (trial / "evaluation/argmax/cases.jsonl").is_file()
        for node in (0, 7, 19):
            evidence = json.loads(
                (trial / f"validation/tick-{node:08d}/model.pt.json").read_text()
            )
            assert evidence["physical_ticks"] == node
        progress = json.loads((trial / "progress.json").read_text())
        assert progress["physical_ticks"] == 19
    with pytest.raises(ValueError, match="already exist"):
        run_launch(plan)


def test_import_only_validation_still_evaluates_generated_test(tmp_path):
    import yaml

    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.experiments.launch import compile_launch, run_launch
    from tests.unit.test_research_packages import case_file

    path = author(tmp_path, recipe("rule"))
    document = yaml.safe_load(path.read_text())
    document.update(
        drain_frozen=True,
        test_cases=4,
        run={
            "task": "evaluate",
            "methods": ["rule"],
            "split": "test",
            "selection": "argmax",
            "seconds": 60,
            "output": str(tmp_path / "result"),
        },
    )
    path.write_text(yaml.safe_dump(document))
    _, factory, workload, _, _ = load_inputs(path)
    case_file(tmp_path, "validation", factory, workload, 31)
    document["case_sets"] = {"validation": "validation.json"}
    path.write_text(yaml.safe_dump(document))
    result = run_launch(compile_launch(path))
    assert result["status"] == "completed"
    cases = (tmp_path / "result/cases.jsonl").read_text().splitlines()
    assert len(cases) == 4
