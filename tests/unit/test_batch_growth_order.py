"""Bounded batch lifecycle acceptance; no scientific simulator or model runs."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from smartsom.experiments import author_batch, batch_append, runtime_control
from smartsom.experiments.evidence import write_json
from smartsom.experiments.launch import compile_launch, run_launch
from tests.unit.test_batch_append_ownership import open_queue
from tests.unit.test_incremental_batch import (
    batch,
    compile_batch,
    hashes,
)
from tests.unit.test_incremental_batch import (
    lightweight as _lightweight,
)


@pytest.fixture
def lightweight(monkeypatch):
    return _lightweight.__wrapped__(monkeypatch)


def add_large(folder):
    for index, suffix in enumerate(("base", "sl")):
        data = yaml.safe_load((folder / f"medium-{suffix}.yaml").read_text())
        data["train_ticks"] = 30 + index
        (folder / f"large-{suffix}.yaml").write_text(yaml.safe_dump(data))


def chosen(folder, ids, **kwargs):
    return compile_launch(
        folder,
        ids=ids,
        operations={"output_root": str(folder.parent / "results")},
        **kwargs,
    )


def test_new_members_supplement_and_repeat_preserve_original_archives(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    first = run_launch(compile_batch(folder, first_train_seeds=1))
    root = folder.parent / "results"
    original = {p: (root / p).read_bytes() for p in ("plan.json", "manifest.json")}
    old = hashes(root / "trials/medium-base/seed-4101")
    add_large(folder)
    selected = chosen(folder, ["large-sl"], first_train_seeds=1, resume_missing=True)
    result = run_launch(selected)
    assert result["completed_tasks"] == 3 and result["submitted_tasks"] == 1
    assert lightweight[-1] == "large-sl--seed-4101"
    assert all("large-base" not in task for task in lightweight)
    _, plan, state = author_batch.load(root)
    assert len(plan["inventory"]) == 6 and len(plan["files"]) == 18
    assert state["files"]["large-base--seed-4101"]["status"] == "deferred"
    assert run_launch(selected)["submitted_tasks"] == 0
    result = run_launch(chosen(folder, ["large-sl"], resume_missing=True))
    assert result["submitted_tasks"] == 2 and result["completed_tasks"] == 5
    assert len(lightweight) == len(set(lightweight)) == 5
    assert len(runtime_control.snapshot(root)["inventory_extensions"]) == 1
    assert hashes(root / "trials/medium-base/seed-4101") == old
    assert {p: (root / p).read_bytes() for p in original} == original
    assert result["directory"] == first["directory"]


def test_active_append_two_submitters_claim_once_and_keep_stale_driver_safe(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    root = open_queue(folder)
    _, _, driver_state = author_batch.load(root)
    original = (root / "plan.json").read_bytes()
    add_large(folder)

    def submit():
        return runtime_control.append(
            root, folder / "large-base.yaml", first_train_seeds=1
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(len(r["tasks"]) for r in receipts) == [0, 1]
    live = runtime_control.snapshot(root)
    assert len(live["inventory_extensions"]) == 1
    assert set(live["additions"]) == {"large-base--seed-4101"}
    with runtime_control.transaction(root) as state:
        state["additions"]["large-base--seed-4101"]["status"] = "failed"
    # A driver that began before inventory growth must not erase appended state.
    author_batch._save(root, driver_state)
    _, plan, saved = author_batch.load(root)
    assert len(plan["files"]) == 18
    assert saved["files"]["large-base--seed-4101"]["status"] == "failed"
    assert (
        runtime_control.append(root, folder / "large-base.yaml", first_train_seeds=1)[
            "tasks"
        ]
        == []
    )
    assert (root / "plan.json").read_bytes() == original


def test_active_append_edited_seed_selects_once(tmp_path, lightweight):
    folder = batch(tmp_path)
    root = open_queue(folder)
    path = folder / "medium-base.yaml"
    data = yaml.safe_load(path.read_text())
    data["train_seeds"].extend([101, 102])
    path.write_text(yaml.safe_dump(data))

    def submit():
        return batch_append.append(
            root,
            folder,
            ids=["medium-base"],
            selectors=[{"train_seeds": [101]}],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(len(r["tasks"]) for r in receipts) == [0, 1]
    _, plan, _ = author_batch.load(root)
    assert sum(row["id"] == "medium-base--seed-101" for row in plan["files"]) == 1
    assert "medium-base--seed-102" not in runtime_control.snapshot(root)["additions"]


def test_seed_append_interruption_retries_original_request(
    tmp_path, lightweight, monkeypatch
):
    folder = batch(tmp_path)
    root = open_queue(folder)
    path = folder / "medium-base.yaml"
    data = yaml.safe_load(path.read_text())
    data["train_seeds"].extend([101, 102])
    path.write_text(yaml.safe_dump(data))
    collect = batch_append._collect

    def interrupted(*args, **kwargs):
        raise RuntimeError("fixture before queue commit")

    monkeypatch.setattr(batch_append, "_collect", interrupted)
    with pytest.raises(RuntimeError, match="before queue commit"):
        batch_append.append(
            root, folder, ids=["medium-base"], selectors=[{"train_seeds": [101]}]
        )
    data["train_seeds"].append(103)
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(batch_append, "_collect", collect)
    with pytest.raises(ValueError, match="restore the prior Experiment YAML"):
        batch_append.append(
            root, folder, ids=["medium-base"], selectors=[{"train_seeds": [101]}]
        )
    data["train_seeds"] = [4101, 4102, 4103, 102]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="restore the prior Experiment YAML"):
        batch_append.append(
            root, folder, ids=["medium-base"], selectors=[{"train_seeds": [102]}]
        )
    pending = next((root / "inventory").glob("pending-*.json"))
    snapshot = json.loads(pending.read_text())["authored_sources"][str(path)]
    path.write_text(snapshot)
    receipt = batch_append.append(
        root, folder, ids=["medium-base"], selectors=[{"train_seeds": [101]}]
    )
    assert receipt["tasks"] == ["medium-base--seed-101"]


def test_growth_still_rejects_old_science_and_member_removal(tmp_path, lightweight):
    folder = batch(tmp_path)
    run_launch(compile_batch(folder, first_train_seeds=1))
    root = folder.parent / "results"
    before = hashes(root)
    add_large(folder)
    path = folder / "medium-sl.yaml"
    old = path.read_text()
    path.write_text(old.replace("train_ticks: 13", "train_ticks: 99"))
    with pytest.raises(ValueError, match="scientific inventory changed"):
        run_launch(
            chosen(folder, ["large-base"], first_train_seeds=1, resume_missing=True)
        )
    assert hashes(root) == before
    path.write_text(old)
    (folder / "small-sl.yaml").unlink()
    with pytest.raises(ValueError, match="scientific inventory changed"):
        run_launch(chosen(folder, ["large-base"], resume_missing=True))
    assert hashes(root) == before


def stages(folder):
    for name in ("small-base", "small-sl"):
        path = folder / (name + ".yaml")
        data = yaml.safe_load(path.read_text())
        data["batch"] = {"stage": 1}
        path.write_text(yaml.safe_dump(data))


def test_deferred_medium_seeds_do_not_block_small_but_real_prior_work_does(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    stages(folder)
    run_launch(compile_batch(folder, first_train_seeds=1))
    small = chosen(
        folder, ["small-base", "small-sl"], first_train_seeds=1, resume_missing=True
    )
    result = run_launch(small)
    assert result["completed_tasks"] == 4 and result["submitted_tasks"] == 2
    assert not any("4102" in key or "4103" in key for key in lightweight)


@pytest.mark.parametrize("status", ["queued", "running", "failed", "paused", "stopped"])
def test_previously_submitted_predecessor_cannot_be_bypassed(
    tmp_path, lightweight, status
):
    folder = batch(tmp_path)
    stages(folder)
    run_launch(compile_batch(folder, first_train_seeds=1))
    root = folder.parent / "results"
    state = json.loads((root / "batch.json").read_text())
    state["files"]["medium-base--seed-4102"]["status"] = status
    write_json(root / "batch.json", state)
    result = run_launch(
        chosen(folder, ["small-base"], first_train_seeds=1, resume_missing=True)
    )
    assert result["status"] == "failed" and "medium-base--seed-4102" in result["error"]
    assert all("small" not in key for key in lightweight)


def test_explicit_order_is_selection_only_and_stage_precedes_priority(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    stages(folder)
    ids = ["small-sl", "medium-sl", "medium-base", "small-base"]
    plan = chosen(folder, ids, first_train_seeds=1)
    default = chosen(folder, list(reversed(ids)), first_train_seeds=1)
    assert [r["id"] for r in plan.files] == [r["id"] for r in default.files]
    assert plan.inventory == default.inventory
    run_launch(plan)
    assert lightweight == [
        "medium-sl--seed-4101",
        "medium-base--seed-4101",
        "small-sl--seed-4101",
        "small-base--seed-4101",
    ]
    assert default.selection["tasks"] == [
        name + "--seed-4101" for name in reversed(ids)
    ]
    with pytest.raises(ValueError, match="duplicate selected task"):
        chosen(folder, ["medium-base", "medium-base"], first_train_seeds=1)


def test_append_order_is_stable_tail_and_never_preempts(tmp_path, lightweight):
    folder = batch(tmp_path)
    root = open_queue(folder)
    add_large(folder)
    order = ["large-sl", "large-base"]
    receipt = batch_append.append(root, folder, ids=order, first_train_seeds=1)
    assert receipt["tasks"] == ["large-sl--seed-4101", "large-base--seed-4101"]
    state = runtime_control.snapshot(root)
    assert runtime_control.priority_order(
        state, list(reversed(state["launch_order"]))
    ) == [
        "medium-base--seed-4101",
        "medium-sl--seed-4101",
        "large-sl--seed-4101",
        "large-base--seed-4101",
    ]
    # User runtime priority still wins, without changing a running task or lease.
    state["priority_tasks"] = ["large-base--seed-4101"]
    assert (
        runtime_control.priority_order(state, state["launch_order"])[0]
        == "large-base--seed-4101"
    )


def test_interrupted_inventory_commit_reuses_only_verified_unstarted_snapshots(
    tmp_path, lightweight, monkeypatch
):
    folder = batch(tmp_path)
    root = open_queue(folder)
    add_large(folder)
    original = batch_append._collect

    def interrupted(*args, **kwargs):
        raise RuntimeError("fixture before queue commit")

    monkeypatch.setattr(batch_append, "_collect", interrupted)
    with pytest.raises(RuntimeError, match="before queue commit"):
        batch_append.append(root, folder / "large-base.yaml", first_train_seeds=1)
    assert not runtime_control.snapshot(root).get("inventory_extensions")
    assert len(author_batch.load(root)[1]["files"]) == 12
    pending = hashes(root / "trials/large-base")
    monkeypatch.setattr(batch_append, "_collect", original)
    assert batch_append.append(root, folder / "large-base.yaml", first_train_seeds=1)[
        "tasks"
    ] == ["large-base--seed-4101"]
    assert hashes(root / "trials/large-base") == pending
    _, plan, _ = author_batch.load(root)
    assert len(plan["files"]) == 18


def test_extension_tamper_rejected_and_parser_accepts_explicit_order(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    root = open_queue(folder)
    add_large(folder)
    batch_append.append(root, folder / "large-base.yaml", first_train_seeds=1)
    ref = runtime_control.snapshot(root)["inventory_extensions"][0]
    path = root / "inventory" / (ref + ".json")
    doc = json.loads(path.read_text())
    doc["files"][0]["stage"] = 99
    write_json(path, doc)
    with pytest.raises(ValueError, match="inventory extension changed"):
        author_batch.load(root)
    from smartsom.experiments.cli import _parser

    for command in (["run", str(folder)], ["add", str(root), str(folder)]):
        args = _parser().parse_args(
            command + ["--id", "medium-sl", "--select", "train_seeds: [4101]"]
        )
        assert args.batch_selectors == [{"train_seeds": [4101]}]


def test_reciprocal_append_releases_nested_locks_and_can_retry(
    tmp_path, lightweight, monkeypatch
):
    """Two destination→owner submissions must fail fast, never wait in a cycle."""
    import threading

    from smartsom.experiments import background

    folders, roots = [], []
    for name in ("left", "right"):
        parent = tmp_path / name
        parent.mkdir()
        folder = batch(parent)
        folders.append(folder)
        roots.append(open_queue(folder))
    real_initialize = author_batch._runtime_initialize
    barrier = threading.Barrier(2)

    def simultaneous_owner(root, *args, **kwargs):
        barrier.wait(timeout=5)
        return real_initialize(root, *args, **kwargs)

    monkeypatch.setattr(author_batch, "_runtime_initialize", simultaneous_owner)
    # Fixture queues have no active processes; ownership still uses real locks.
    monkeypatch.setattr(background, "_check_owner", lambda root: None)
    outcomes = []

    def submit(index):
        try:
            outcomes.append(
                batch_append.append(
                    roots[index],
                    folders[1 - index] / "small-base.yaml",
                    first_train_seeds=1,
                    output_root=roots[1 - index],
                )
            )
        except RuntimeError as error:
            outcomes.append(str(error))

    workers = [threading.Thread(target=submit, args=(i,), daemon=True) for i in (0, 1)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=8)
    assert not any(worker.is_alive() for worker in workers)
    assert len(outcomes) == 2
    assert any(
        isinstance(result, str) and "runtime control busy" in result
        for result in outcomes
    )
    assert all(
        isinstance(result, dict) or "runtime control busy" in result
        for result in outcomes
    )
    monkeypatch.setattr(author_batch, "_runtime_initialize", real_initialize)
    # Every lock is released. A completed submission remains deduplicated; a
    # failed submission can be retried without a phantom command or claim.
    for i in (0, 1):
        result = batch_append.append(
            roots[i],
            folders[1 - i] / "small-base.yaml",
            first_train_seeds=1,
            output_root=roots[1 - i],
        )
        assert len(result["tasks"]) <= 1


def test_completed_external_owner_new_deferred_members_are_partial_even_before_save(
    tmp_path, lightweight
):
    from smartsom.experiments.batch_inventory import prepare, read_view
    from smartsom.experiments.catalog import describe

    folder = batch(tmp_path)
    names = ["medium-base", "medium-sl", "small-base", "small-sl"]
    result = run_launch(chosen(folder, names, first_train_seeds=3))
    assert result["status"] == "completed"
    root = folder.parent / "results"
    state = json.loads((root / "batch.json").read_text())
    state["stage"] = "external_queue"
    write_json(root / "batch.json", state)
    add_large(folder)
    compiled = chosen(folder, ["medium-base"], first_train_seeds=1, resume_missing=True)
    _, plan, state = author_batch.load(root)
    # Simulate durable membership commit before the driver state write (also the
    # external owner's no-jobs append branch). No fake completion is published.
    with runtime_control.transaction(root) as live:
        prepare(root, compiled, plan, state, live)
    assert json.loads((root / "batch.json").read_text())["status"] == "completed"
    assert read_view(root)[1]["status"] == "partial"
    assert author_batch.load(root)[2]["status"] == "partial"
    assert describe(root)["status"] == "partial"
    result = run_launch(compiled)
    assert result["status"] == "partial" and result["submitted_tasks"] == 0
    assert result["completed_tasks"] == 12 and result["deferred_tasks"] == 6
    assert len(lightweight) == 12
