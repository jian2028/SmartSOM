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
