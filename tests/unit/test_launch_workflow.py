"""Public operation choices preserve frozen scientific conditions."""

import json
from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.launch import resolve_experiment
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.cli import _parser, main
from smartsom.experiments.launch import compare, compile_launch, update_execution
from smartsom.experiments.reservation import allocate, load

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "configs/base-diagnostic/small.yaml"


def test_tracking_group_is_launch_unique_and_inherited(tmp_path):
    from smartsom.experiments.reservation import arguments

    plan = compile_reservation(SOURCE)
    one = allocate(plan, root=tmp_path / "one")
    two = allocate(plan, root=tmp_path / "two")
    group = arguments(one)[0].tracking_group
    assert group != arguments(two)[0].tracking_group
    child = allocate(plan, root=tmp_path / "child", tracking_group=group)
    assert arguments(child)[0].tracking_group == group
    assert load(child)["tracking_group"] == group


def test_profile_cli_precedence_and_scientific_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    profile = tmp_path / "server.yaml"
    profile.write_text(
        "launch:\n  mode: background\nconcurrency:\n  max_concurrent: 4\ntracking:\n  mode: offline\n"
    )
    before = compile_reservation(SOURCE)
    after = compile_reservation(
        SOURCE,
        profile=profile,
        operations={
            "launch": {"mode": "foreground"},
            "concurrency": {"max_concurrent": 3},
        },
    )
    assert after.experiment.launch.mode == "foreground"
    assert after.operations["concurrency"]["max_concurrent"] == 3
    assert after.origins["launch.mode"] == "CLI"
    assert after.origins["tracking.mode"] == str(profile)
    assert after.scientific_sha256 == before.scientific_sha256
    changed = compile_reservation(
        SOURCE,
        sets=["algorithm.production.policies.ppo.learner.params.learning_rate=0.002"],
    )
    assert changed.scientific_sha256 != before.scientific_sha256


@pytest.mark.parametrize(
    "value",
    [
        {"algorithm": {"learning_rate": 1}},
        {"runtime": {"device": "cuda"}},
        {"concurrency": {"max_concurrent": 0}},
        {"launch": {"background": True}},
    ],
)
def test_profile_rejects_scientific_or_invalid_fields(tmp_path, value):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        compile_reservation(SOURCE, profile=path)


def test_names_are_explicit_and_ambiguous_names_fail(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    folder = tmp_path / "configs/runs"
    (folder / "one").mkdir(parents=True)
    (folder / "one/train.yaml").touch()
    assert resolve_experiment("train", tmp_path) == folder / "one/train.yaml"
    (folder / "two").mkdir()
    (folder / "two/train.yml").touch()
    with pytest.raises(ValueError, match="candidates"):
        resolve_experiment("train", tmp_path)


def test_foreground_background_mutually_exclusive():
    parser = _parser()
    assert parser.parse_args(["run", "a.yaml"]).background is None
    assert parser.parse_args(["run", "a.yaml", "--foreground"]).background is False
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "a.yaml", "--foreground", "--background"])


def test_resume_monitor_change_preserves_frozen_bytes_and_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    plan = compile_reservation(SOURCE)
    root = allocate(plan, root=tmp_path / "run")
    original = {p: p.read_bytes() for p in (root / "inputs").rglob("*") if p.is_file()}
    update_execution(
        root,
        operations={
            "tracking": {"mode": "offline", "tensorboard": True},
            "launch": {"mode": "background"},
        },
    )
    assert all(p.read_bytes() == data for p, data in original.items())
    assert load(root)["scientific_sha256"] == plan.scientific_sha256
    ex, *_ = load_inputs(root / "inputs/experiment.yaml")
    assert ex.tracking.mode == "offline" and ex.tracking.tensorboard
    assert ex.launch.mode == "background"
    assert len(json.loads((root / "execution.json").read_text())["history"]) == 1
    assert all("scientific." not in row["field"] for row in compare(SOURCE, root))


def test_check_directory_and_file_share_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    data = yaml.safe_load(SOURCE.read_text())
    for name in ("factory", "workload", "algorithm"):
        data[name] = str(SOURCE.parent / data[name])
    data["run"] = {"output": str(tmp_path / "runs/a")}
    folder = tmp_path / "inputs"
    folder.mkdir()
    path = folder / "a.yaml"
    path.write_text(yaml.safe_dump(data))
    overrides = {
        "operations": {"concurrency": {"max_concurrent": 4, "mode": "adaptive"}}
    }
    single = compile_launch(path, **overrides)
    batch = compile_launch(folder, **overrides)
    assert batch.operations == single.operations
    assert batch.files[0]["compiled"].scientific_sha256 == single.scientific_sha256
    assert (
        main(["check", str(folder), "--max-concurrent", "4", "--wandb", "disabled"])
        == 0
    )
    assert not (tmp_path / "runs").exists()


def test_catalog_ids_results_and_historical_read_only(tmp_path):
    from smartsom.experiments.catalog import (
        describe,
        require_current,
        resolve_run,
        results,
    )

    root = allocate(compile_reservation(SOURCE), root=tmp_path / "new")
    row = describe(root)
    assert resolve_run(row["short_id"][:6], [tmp_path]) == root
    assert results(tmp_path, algorithm="ppo", seed=101)[0]["directory"] == str(root)
    old = tmp_path / "old"
    old.mkdir()
    (old / "run.json").write_text('{"schema":"old/v1","status":"completed"}')
    assert describe(old)["status"] == "completed"
    with pytest.raises(ValueError, match="read-only"):
        require_current(old)


def test_legacy_batch_schema_does_not_authorize_current_control(tmp_path):
    from smartsom.experiments.catalog import require_current

    (tmp_path / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-batch-plan/v1", "contract": "v4"})
    )
    with pytest.raises(ValueError, match="read-only"):
        require_current(tmp_path)


def test_explicit_output_name_survives_output_root(tmp_path, capsys):
    assert (
        main(["check", str(SOURCE), "--output-root", str(tmp_path), "--name", "chosen"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["output"] == str(tmp_path / "chosen")


def test_batch_resource_override_preserves_child_tracking(tmp_path, monkeypatch):
    from smartsom.experiments.author_batch import allocate as allocate_batch

    monkeypatch.setenv("WANDB_MODE", "disabled")
    folder = tmp_path / "author"
    folder.mkdir()
    for i, mode in enumerate(("disabled", "offline")):
        data = yaml.safe_load(SOURCE.read_text())
        for key in ("factory", "workload", "algorithm"):
            data[key] = str(SOURCE.parent / data[key])
        data["run"] = {"output": str(tmp_path / "runs" / str(i)), "seed": 101 + i}
        data["tracking"] = {"mode": mode, "project": f"project{i}"}
        (folder / f"trial{i}.yaml").write_text(yaml.safe_dump(data))
    root = allocate_batch(compile_launch(folder))
    update_execution(root, operations={"resources": {"cpus": 2}})
    for i, mode in enumerate(("disabled", "offline")):
        ex, *_ = load_inputs(
            root / "experiments" / f"trial{i}" / "inputs/experiment.yaml"
        )
        assert ex.tracking.mode == mode and ex.tracking.project == f"project{i}"
        assert ex.resources.cpus == 2
    assert main(["results", str(tmp_path / "runs")]) == 0


def test_matrix_and_directory_share_one_flat_admission_plan(tmp_path):
    import yaml

    source = yaml.safe_load(SOURCE.read_text())
    for key in ("factory", "workload", "algorithm"):
        source[key] = str((SOURCE.parent / source[key]).resolve())
    source["matrix"] = {
        "experiment.run.seed": [101, 102],
        "algorithm.decision_protocol.pre.accept.allow_stop": [False, True],
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(source))
    single = compile_launch(path, operations={"concurrency": {"max_concurrent": 2}})
    directory = compile_launch(
        tmp_path, operations={"concurrency": {"max_concurrent": 2}}
    )
    assert len(single.files) == len(directory.files) == 4
    assert len({r["scientific_sha256"] for r in single.files}) == 4
    assert single.operations["concurrency"]["max_concurrent"] == 2
    assert [r["id"] for r in single.files] == [r["id"] for r in directory.files]
    source["matrix"] = {"experiment.tracking.mode": ["disabled", "offline"]}
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="matrix axis"):
        compile_launch(path)


def test_single_run_budget_is_admission_not_scientific_override(tmp_path):
    from smartsom.experiments.launch import check_resource_budget

    plan = compile_reservation(SOURCE, operations={"resources": {"cpus": 0.5}})
    root = allocate(plan, root=tmp_path / "limited")
    with pytest.raises(ValueError, match="cannot admit"):
        check_resource_budget(root)
    assert not json.loads((root / "resource-admission.json").read_text())["accepted"]
    assert not (root / "manifest.json").exists()


def test_expanded_matrix_id_collision_fails_before_output_allocation(tmp_path, capsys):
    folder = tmp_path / "author"
    folder.mkdir()
    source = yaml.safe_load(SOURCE.read_text())
    for key in ("factory", "workload", "algorithm"):
        source[key] = str(SOURCE.parent / source[key])
    source["run"] = {"output": str(tmp_path / "runs/a")}
    source["matrix"] = {"experiment.run.seed": [101]}
    (folder / "trial.yaml").write_text(yaml.safe_dump(source))
    source.pop("matrix")
    source["run"]["seed"] = 102
    (folder / "trial--0001.yaml").write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="expanded batch ID") as error:
        compile_launch(folder)
    assert str(folder / "trial.yaml") in str(error.value)
    assert str(folder / "trial--0001.yaml") in str(error.value)
    for command in ("check", "run"):
        assert main([command, str(folder), "--wandb", "disabled"]) == 2
        assert "expanded batch ID" in capsys.readouterr().err
        assert not (tmp_path / "runs").exists()


def test_batch_resume_initializes_and_reapplies_explicit_limits(tmp_path):
    from smartsom.experiments import runtime_control
    from smartsom.experiments.author_batch import _runtime_initialize
    from smartsom.experiments.author_batch import allocate as allocate_batch

    source = yaml.safe_load(SOURCE.read_text())
    for key in ("factory", "workload", "algorithm"):
        source[key] = str((SOURCE.parent / source[key]).resolve())
    source.setdefault("run", {})["output"] = str(tmp_path / "runs" / "run")
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "one.yaml").write_text(yaml.safe_dump(source))
    root = allocate_batch(compile_launch(folder))
    update_execution(
        root,
        operations={"concurrency": {"max_concurrent": 7}, "resources": {"cpus": 4}},
    )
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    _runtime_initialize(root, plan, state)
    assert runtime_control.snapshot(root)["limit"] == 7
    assert runtime_control.snapshot(root)["budget"]["cpus"] == 4
    runtime_control.submit(root, "limit", value=3)
    runtime_control.apply(root)
    update_execution(root, operations={"concurrency": {"max_concurrent": 7}})
    assert runtime_control.apply(root)["limit"] == 7


def test_historical_batch_control_rejected_without_writes(tmp_path):
    from smartsom.experiments import runtime_control

    (tmp_path / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-batch-plan/v1", "contract": "v4"})
    )
    runtime_control.initialize(tmp_path, 2)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["batch-limit", str(tmp_path), "9"]) == 2
    assert all(p.read_bytes() == data for p, data in before.items())
