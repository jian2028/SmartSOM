import hashlib
import json
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from smartsom.experiments.packaging import (
    export_experiment,
    export_model,
    import_bundle,
    model_locator,
    relocate_reference,
    verify_bundle,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def training(tmp_path):
    root = tmp_path / "original"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "model.zip"
    weights.write_bytes(b"small synthetic model fixture")
    save(
        checkpoint / "checkpoint.json",
        {
            "schema": "smartsom.checkpoint/v1",
            "provider": "sb3.maskable_ppo",
            "projection": {"max_jobs": 4},
            "parameters": {},
            "files": [
                {
                    "path": "model.zip",
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    save(
        root / "checkpoint_algorithm.json",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {"checkpoint": "checkpoint"},
        },
    )
    save(
        root / "manifest.json",
        {"schema": "smartsom.training-manifest/v1", "status": "completed"},
    )
    save(root / "snapshot.json", {"historical_path": str(checkpoint)})
    return root


def test_model_export_import_is_self_contained_after_moving_original(
    training, tmp_path
):
    old = training / "checkpoint"
    original_bytes = {p.name: p.read_bytes() for p in old.iterdir()}
    bundle = export_model(training, tmp_path / "model.zip")
    manifest = verify_bundle(bundle)
    assert manifest["kind"] == "model"
    moved = tmp_path / "elsewhere"
    training.rename(moved)
    imported = import_bundle(bundle, tmp_path / "imported")
    loaded = model_locator(imported)
    assert {p.name: p.read_bytes() for p in loaded.iterdir()} == original_bytes
    assert relocate_reference(imported, old / "model.zip") == loaded / "model.zip"
    assert not any(p.is_symlink() for p in imported.rglob("*"))


def test_full_export_preserves_raw_bytes_and_relocation_overlay(training, tmp_path):
    snapshot = (training / "snapshot.json").read_bytes()
    bundle = export_experiment(training, tmp_path / "experiment.zip")
    imported = import_bundle(bundle, tmp_path / "imported")
    assert (imported / "snapshot.json").read_bytes() == snapshot
    assert (
        relocate_reference(imported, training / "checkpoint/model.zip")
        == imported / "checkpoint/model.zip"
    )
    with pytest.raises(ValueError, match="not included"):
        relocate_reference(imported, "/not/exported/model.zip")
    assert (training / "snapshot.json").read_bytes() == snapshot


def test_materializes_internal_symlinks_as_actual_files(training, tmp_path):
    (training / "latest").symlink_to("checkpoint", target_is_directory=True)
    exported = export_experiment(training, tmp_path / "full.zip")
    imported = import_bundle(exported, tmp_path / "imported")
    assert (imported / "latest/model.zip").read_bytes() == (
        imported / "checkpoint/model.zip"
    ).read_bytes()
    assert not (imported / "latest").is_symlink()


def test_external_symlink_is_not_silently_exported(training, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("not part of the experiment")
    (training / "external").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        export_experiment(training, tmp_path / "full.zip")
    assert not (tmp_path / "full.zip").exists()


def test_rejects_changed_model_before_export(training, tmp_path):
    (training / "checkpoint/model.zip").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest mismatch"):
        export_model(training, tmp_path / "model.zip")
    assert not (tmp_path / "model.zip").exists()


def test_rejects_changed_bundle_before_creating_destination(training, tmp_path):
    original = export_model(training, tmp_path / "model.zip")
    broken = tmp_path / "broken.zip"
    with ZipFile(original) as source, ZipFile(broken, "w") as output:
        for entry in source.infolist():
            payload = source.read(entry)
            if entry.filename.endswith("model.zip"):
                payload = b"X" * len(payload)
            output.writestr(entry, payload)
    with pytest.raises(ValueError, match="digest mismatch"):
        import_bundle(broken, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "payload/../escape",
        "C:/escape",
        "payload\\escape",
        "./payload/a",
    ],
)
def test_unsafe_zip_paths_never_extract(tmp_path, name):
    bundle = tmp_path / "bad.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        import_bundle(bundle, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_archive_symlinks_are_rejected(tmp_path):
    bundle = tmp_path / "symlink.zip"
    entry = ZipInfo("payload/model")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(bundle, "w") as archive:
        archive.writestr(entry, "../../outside")
    with pytest.raises(ValueError, match="regular"):
        verify_bundle(bundle)


def test_import_refuses_existing_destination(training, tmp_path):
    archive = export_model(training, tmp_path / "model.zip")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        import_bundle(archive, existing)
    assert list(existing.iterdir()) == []


def test_export_destination_cannot_change_source_evidence(training):
    with pytest.raises(ValueError, match="outside"):
        export_model(training, training / "export.zip")
    with pytest.raises(ValueError, match="outside"):
        export_model(training, training / "checkpoint/export.zip")
    with pytest.raises(ValueError, match="outside"):
        export_experiment(training, training / "export.zip")


def test_explicit_missing_best_does_not_fall_back_to_last(training):
    with pytest.raises(ValueError, match="unavailable"):
        model_locator(training, "best")


def update_checkpoint(training, number):
    import shutil

    update = training / "checkpoints" / f"update-{number:06d}"
    shutil.copytree(training / "checkpoint", update / "inference")
    save(
        update / "resolved_training.json",
        {
            "schema": "smartsom.resolved-training/v1",
            "resolved": {},
        },
    )
    (update / "rng.pkl").write_bytes(b"opaque training state; never executed")
    save(
        update / "manifest.json",
        {
            "schema": "smartsom.update-checkpoint/v1",
            "status": "complete",
            "files": {
                str(path.relative_to(update)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in update.rglob("*")
                if path.is_file()
            },
        },
    )
    return update


def test_update_selection_and_nested_current_run_preserve_best(training, tmp_path):
    first = update_checkpoint(training, 1)
    second = update_checkpoint(training, 2)
    save(training / "checkpoints/last.json", {"checkpoint": str(second)})
    save(training / "checkpoints/best.json", {"checkpoint": str(first)})
    assert model_locator(training) == second / "inference"
    assert model_locator(training, "best") == first / "inference"
    assert model_locator(first) == first / "inference"
    assert model_locator(training / "checkpoints/best.json") == first / "inference"
    outer = tmp_path / "outer"
    outer.mkdir()
    training.rename(outer / "training")
    # Use relative pointers for the authoring-time move in this fixture.
    training = outer / "training"
    save(training / "checkpoints/last.json", {"checkpoint": "update-000002"})
    save(training / "checkpoints/best.json", {"checkpoint": "update-000001"})
    save(
        outer / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "id": "a",
            "name": "new",
            "kind": "train",
            "status": "completed",
            "paths": {"training": "training"},
        },
    )
    assert (
        model_locator(outer, "best") == training / "checkpoints/update-000001/inference"
    )


def test_updated_checkpoint_digest_and_cycle_rejections(training):
    update = update_checkpoint(training, 1)
    (update / "rng.pkl").write_bytes(b"changed state")
    with pytest.raises(ValueError, match="update checkpoint file digest mismatch"):
        model_locator(update)
    save(training / "checkpoints/last.json", {"checkpoint": "last.json"})
    with pytest.raises(ValueError, match="cyclic"):
        model_locator(training)


def test_update_export_keeps_inference_and_training_recipe_without_resume_claim(
    training, tmp_path
):
    update = update_checkpoint(training, 1)
    save(training / "checkpoints/last.json", {"checkpoint": str(update)})
    archive = export_model(training, tmp_path / "selected.zip")
    imported = import_bundle(archive, tmp_path / "imported")
    assert model_locator(imported) == imported / "checkpoint"
    assert (imported / "resolved_training.json").read_bytes() == (
        update / "resolved_training.json"
    ).read_bytes()
    assert not list(imported.rglob("rng.pkl"))


def test_full_update_bundle_resolves_absolute_pointer_after_original_move(
    training, tmp_path
):
    update = update_checkpoint(training, 1)
    save(training / "checkpoints/last.json", {"checkpoint": str(update)})
    before = (training / "checkpoints/last.json").read_bytes()
    archive = export_experiment(training, tmp_path / "updated.zip")
    training.rename(tmp_path / "moved-training")
    imported = import_bundle(archive, tmp_path / "imported")
    assert model_locator(imported) == imported / "checkpoints/update-000001/inference"
    assert (imported / "checkpoints/last.json").read_bytes() == before


def evaluation_evidence(training, root):
    checkpoint = training / "checkpoint"
    snapshot = training / "resolved_training.json"
    save(snapshot, {"schema": "smartsom.resolved-training/v1", "resolved": {}})
    identity = {
        "path": str(checkpoint),
        "manifest_sha256": hashlib.sha256(
            (checkpoint / "checkpoint.json").read_bytes()
        ).hexdigest(),
        "training_snapshot": str(snapshot),
        "training_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    save(
        root / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "id": "a",
            "name": "evaluation",
            "kind": "evaluate",
            "status": "completed",
            "paths": {"evaluation": "evaluation"},
            "checkpoint": identity,
        },
    )
    save(
        root / "evaluation/run.json",
        {
            "schema": "smartsom.evaluation/v1",
            "checkpoint": identity,
            "results": [{"checkpoint": identity}],
        },
    )
    save(root / "evaluation/plan.json", {"entries": [{"checkpoint": identity}]})
    save(
        root / "evaluation/snapshots/a.json",
        {
            "schema": "smartsom.resolved-run/v1",
            "algorithm": {"algorithm": {"checkpoint": str(checkpoint)}},
        },
    )
    (root / "evaluation/resolved_run.yaml").write_text(
        "schema: smartsom.resolved-run/v1\nalgorithm:\n  algorithm:\n"
        f"    checkpoint: {checkpoint}\n"
    )
    return root, identity


def test_evaluation_bundle_captures_declared_external_model_and_recipe(
    training, tmp_path
):
    from smartsom.experiments.packaging import locate_reference

    original, identity = evaluation_evidence(training, tmp_path / "evaluation")
    original_files = {
        str(p.relative_to(original)): p.read_bytes()
        for p in original.rglob("*")
        if p.is_file()
    }
    (training / "unrelated.txt").write_text("not a declared dependency")
    archive = export_experiment(original, tmp_path / "evaluation.zip")
    verified = verify_bundle(archive)
    assert sum(row["path"].endswith("/model.zip") for row in verified["files"]) == 1
    assert not any("unrelated.txt" in row["path"] for row in verified["files"])
    imported = import_bundle(archive, tmp_path / "imported")
    # Package references must prefer the captured copy even before old files move.
    selected = model_locator(imported)
    assert selected.is_relative_to(imported)
    original.rename(tmp_path / "moved-evaluation")
    training.rename(tmp_path / "moved-model")
    for name, payload in original_files.items():
        assert (imported / name).read_bytes() == payload
    assert model_locator(imported) == selected
    recipe = locate_reference(imported / "run.json", identity["training_snapshot"])
    assert recipe.is_relative_to(imported)
    assert recipe.is_file()
    assert (
        locate_reference(imported / "evaluation/snapshots/a.json", identity["path"])
        == selected
    )
    # Re-exporting an imported experiment retains historical reference resolution.
    again = export_experiment(imported, tmp_path / "again.zip")
    second = import_bundle(again, tmp_path / "second")
    assert model_locator(second).is_relative_to(second)
    assert locate_reference(
        second / "run.json", identity["training_snapshot"]
    ).is_file()


def test_evaluation_dependency_checksum_and_link_escape_fail_before_export(
    training, tmp_path
):
    original, _ = evaluation_evidence(training, tmp_path / "evaluation")
    outside = tmp_path / "other-file"
    outside.write_bytes(b"not a declared dependency")
    (training / "checkpoint/extra-link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        export_experiment(original, tmp_path / "escape.zip")
    (training / "checkpoint/extra-link").unlink()
    (training / "resolved_training.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        export_experiment(original, tmp_path / "drift.zip")
    assert not (tmp_path / "escape.zip").exists()
    assert not (tmp_path / "drift.zip").exists()


def test_export_destination_cannot_modify_declared_external_dependency(
    training, tmp_path
):
    original, _ = evaluation_evidence(training, tmp_path / "evaluation")
    with pytest.raises(ValueError, match="outside source evidence"):
        export_experiment(original, training / "export.zip")
    assert not (training / "export.zip").exists()


def test_relative_external_model_reference_keeps_owner_semantics_after_move(
    training, tmp_path
):
    from smartsom.experiments.packaging import locate_reference

    root = tmp_path / "evaluation"
    save(root / "manifest.json", {"schema": "smartsom.manifest/v1"})
    save(
        root / "checkpoint_algorithm.json",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {"checkpoint": "../original/checkpoint"},
        },
    )
    raw = (root / "checkpoint_algorithm.json").read_bytes()
    archive = export_experiment(root, tmp_path / "relative.zip")
    root.rename(tmp_path / "moved-evaluation")
    training.rename(tmp_path / "moved-model")
    imported = import_bundle(archive, tmp_path / "imported")
    selected = model_locator(imported)
    assert selected.is_relative_to(imported)
    assert (
        locate_reference(
            imported / "checkpoint_algorithm.json", "../original/checkpoint"
        )
        == selected
    )
    assert (imported / "checkpoint_algorithm.json").read_bytes() == raw
    again = export_experiment(imported, tmp_path / "relative-again.zip")
    second = import_bundle(again, tmp_path / "second")
    assert model_locator(second).is_relative_to(second)


def test_legacy_training_without_descriptor_has_only_last(training):
    (training / "checkpoint_algorithm.json").unlink()
    assert model_locator(training) == training / "checkpoint"
    with pytest.raises(ValueError, match="unavailable"):
        model_locator(training, "best")
