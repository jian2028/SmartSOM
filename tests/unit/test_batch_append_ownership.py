"""Owned append fixtures preserve directory identities and frozen inventories."""

import json
from pathlib import Path

import pytest

from smartsom.experiments import (
    author_batch,
    batch_append,
    queue_admission,
    runtime_control,
)
from smartsom.experiments.launch import compile_launch, run_launch
from tests.unit.test_incremental_batch import (  # noqa: F401
    batch,
    compile_batch,
    lightweight,  # noqa: F401
)


def open_queue(folder):
    compiled = compile_batch(folder, first_train_seeds=1)
    root = author_batch.allocate(compiled)
    _, plan, state = author_batch.load(root)
    author_batch._runtime_initialize(root, plan, state)
    return root


@pytest.mark.usefixtures("lightweight")
def test_same_batch_file_directory_append_has_same_task_paths_and_no_duplicates(
    tmp_path,
):
    folder = batch(tmp_path)
    root = open_queue(folder)
    receipt = runtime_control.append(
        root, folder / "small-base.yaml", first_train_seeds=1
    )
    assert receipt["tasks"] == ["small-base--seed-4101"]
    job = runtime_control.snapshot(root)["additions"][receipt["tasks"][0]]
    assert Path(job["root"]) == root / "trials/small-base/seed-4101"
    repeated = runtime_control.append(
        root, folder, ids=["small-base"], first_train_seeds=1
    )
    assert repeated["tasks"] == []
    more = runtime_control.append(root, folder, ids=["small-base"])
    assert len(more["tasks"]) == 2
    assert not (root / "added").exists()


@pytest.mark.usefixtures("lightweight")
def test_external_batch_claim_completion_and_original_resume(
    tmp_path, monkeypatch, request
):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    other_base = tmp_path / "other"
    other_base.mkdir()
    other = batch(other_base)
    receipt = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    assert len(receipt["tasks"]) == 1
    key = receipt["tasks"][0]
    jobs = runtime_control.snapshot(root)["additions"]
    job = jobs[key]
    owner = Path(job["owner_root"])
    expected = compile_launch(other, first_train_seeds=1).output_root
    assert owner == expected
    assert Path(job["root"]) == owner / "trials/medium-base/seed-4101"
    assert queue_admission.snapshot(owner)["queue"] == str(root)
    # Independent original-batch ownership fails before any model can start.
    with pytest.raises(ValueError, match="adopted directory"):
        queue_admission.check_registration(owner)
    assert (
        runtime_control.append(root, other, ids=["medium-base"], first_train_seeds=1)[
            "tasks"
        ]
        == []
    )
    # Simulate the worker's completed native evaluation receipt, then synchronize
    # through the same function used after every real scheduler observation.
    child = Path(job["root"])
    (child / "evaluation").mkdir()
    (child / "evaluation/completed.json").write_text("{}")
    job["status"] = "completed"
    batch_append.synchronize(root, jobs)
    assert "queue" not in queue_admission.snapshot(owner)
    state = json.loads((owner / "batch.json").read_text())
    assert state["files"]["medium-base--seed-4101"]["status"] == "completed"
    # Re-enter from the owning directory; only its other two selected seeds run.
    plan = compile_launch(other, ids=["medium-base"], resume_missing=True)
    result = run_launch(plan)
    assert result["submitted_tasks"] == 2
    assert "medium-base--seed-4101" not in request.getfixturevalue("lightweight")
    assert result["completed_tasks"] == 3


@pytest.mark.usefixtures("lightweight")
def test_single_file_and_parent_directory_resolve_same_owning_batch(
    tmp_path, monkeypatch, request
):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    child = batch(external_parent)
    parent = {"directory": str(folder), "root": str(root)}
    one = batch_append.compile_sources(child / "medium-base.yaml", parent=parent)[0]
    parent_scan = batch_append.compile_sources(
        external_parent, parent=parent, ids=["batch/medium-base"]
    )[0]
    assert one.output_root == parent_scan.output_root
    assert one.selection == parent_scan.selection
    assert one.inventory == parent_scan.inventory
    assert [row["child"] for row in one.files] == [
        row["child"] for row in parent_scan.files
    ]


@pytest.mark.usefixtures("lightweight")
def test_foreign_batch_changed_science_rejected_and_claim_preserved(
    tmp_path, monkeypatch, request
):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    other = batch(external_parent)
    receipt = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    owner = Path(
        runtime_control.snapshot(root)["additions"][receipt["tasks"][0]]["owner_root"]
    )
    path = other / "small-base.yaml"
    path.write_text(path.read_text().replace("train_ticks: 10", "train_ticks: 99"))
    with pytest.raises(ValueError, match="scientific inventory changed"):
        runtime_control.append(root, other / "medium-sl.yaml", first_train_seeds=1)
    assert queue_admission.snapshot(owner)["queue"] == str(root)
    assert len(runtime_control.snapshot(root)["additions"]) == 1


@pytest.mark.usefixtures("lightweight")
def test_concurrent_file_submissions_claim_one_task(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    folder = batch(tmp_path)
    root = open_queue(folder)
    barrier = Barrier(2)

    def submit():
        barrier.wait(timeout=5)
        return runtime_control.append(
            root, folder / "small-base.yaml", first_train_seeds=1
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        receipts = [future.result(timeout=15) for future in futures]
    assert sorted(len(receipt["tasks"]) for receipt in receipts) == [0, 1]
    assert len(runtime_control.snapshot(root)["additions"]) == 1


@pytest.mark.usefixtures("lightweight")
def test_finished_foreign_jobs_do_not_reclaim_or_overwrite_original_batch(
    tmp_path, monkeypatch, request
):
    from smartsom.experiments.locking import exclusive_lock

    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    other = batch(external_parent)
    receipt = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    jobs = runtime_control.snapshot(root)["additions"]
    job = jobs[receipt["tasks"][0]]
    owner = Path(job["owner_root"])
    job["status"] = "completed"
    batch_append.synchronize(root, jobs)
    # The old queue may remain alive for unrelated work. It must no longer take
    # this driver's lock, nor overwrite state after ownership has been returned.
    before = (owner / "batch.json").read_bytes()
    with exclusive_lock(owner / "driver.lock"):
        batch_append.synchronize(root, jobs)
    assert (owner / "batch.json").read_bytes() == before


@pytest.mark.usefixtures("lightweight")
def test_durable_completion_before_claim_release_is_recoverable(tmp_path, monkeypatch):
    from smartsom.experiments.evidence import write_json

    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    other = batch(external_parent)
    receipt = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    jobs = runtime_control.snapshot(root)["additions"]
    job = jobs[receipt["tasks"][0]]
    owner = Path(job["owner_root"])
    job["status"] = "completed"
    state = json.loads((owner / "batch.json").read_text())
    state["files"][job["owner_task"]]["status"] = "completed"
    write_json(owner / "batch.json", state)
    # Simulated interruption after receipt durability but before claim release.
    batch_append.synchronize(root, jobs)
    assert "queue" not in queue_admission.snapshot(owner)


@pytest.mark.usefixtures("lightweight")
def test_explicit_external_root_and_known_ownership_are_selection_independent(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    other_parent = tmp_path / "other"
    other_parent.mkdir()
    other = batch(other_parent)
    destination = tmp_path / "custom-output"
    compiled = compile_launch(
        other, first_train_seeds=1, operations={"output_root": str(destination)}
    )
    author_batch.allocate(compiled)
    first = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1, output_root=destination
    )
    jobs = runtime_control.snapshot(root)["additions"]
    assert {job["owner_root"] for job in jobs.values()} == {str(destination)}
    # Later file/directory/parent selections reuse the already-known owner root.
    again = runtime_control.append(
        root, other, ids=["medium-base"], first_train_seeds=1
    )
    assert again["tasks"] == []
    second = runtime_control.append(
        root, other_parent, ids=["batch/medium-sl"], first_train_seeds=1
    )
    assert len(first["tasks"]) == len(second["tasks"]) == 1
    assert {
        job["owner_root"]
        for job in runtime_control.snapshot(root)["additions"].values()
    } == {str(destination)}
    default = compile_launch(other, first_train_seeds=1).output_root
    assert not default.exists()


@pytest.mark.usefixtures("lightweight")
def test_stale_scheduler_snapshot_cannot_release_newly_accepted_owner_task(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    other_parent = tmp_path / "other"
    other_parent.mkdir()
    other = batch(other_parent)
    first = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    # Scheduler has observed the old worker's terminal state, but not the later
    # accepted append. This is the exact interleaving before _persist/detach.
    stale_jobs = runtime_control.snapshot(root)["additions"]
    old = stale_jobs[first["tasks"][0]]
    old["status"] = "completed"
    owner = Path(old["owner_root"])
    second = runtime_control.append(root, other / "medium-sl.yaml", first_train_seeds=1)
    batch_append.synchronize(root, stale_jobs)
    assert queue_admission.snapshot(owner)["queue"] == str(root)
    with pytest.raises(ValueError, match="adopted directory"):
        queue_admission.check_registration(owner)
    latest = runtime_control.snapshot(root)["additions"]
    assert latest[second["tasks"][0]]["status"] == "queued"
    assert latest[first["tasks"][0]]["status"] == "completed"
    latest[second["tasks"][0]]["status"] = "completed"
    batch_append.synchronize(root, latest)
    assert "queue" not in queue_admission.snapshot(owner)


@pytest.mark.usefixtures("lightweight")
def test_uncertain_retry_keeps_external_owner_claim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = batch(tmp_path)
    root = open_queue(folder)
    other_root = tmp_path / "other-retry"
    other_root.mkdir()
    other = batch(other_root)
    receipt = runtime_control.append(
        root, other / "medium-base.yaml", first_train_seeds=1
    )
    key = receipt["tasks"][0]
    jobs = runtime_control.snapshot(root)["additions"]
    owner = Path(jobs[key]["owner_root"])
    jobs[key].update(status="failed", retry_recovery_required="unknown spawn outcome")
    batch_append.synchronize(root, jobs)
    assert queue_admission.snapshot(owner).get("queue") == str(root)
    jobs[key]["retry_recovery_required"] = None
    batch_append.synchronize(root, jobs)
    assert "queue" not in queue_admission.snapshot(owner)
