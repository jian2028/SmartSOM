import json
import os
from pathlib import Path

import pytest

from smartsom.experiments.catalog import (
    artifact_paths,
    list_runs,
    read_run,
    rebuild_views,
    resolve_run,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def experiment(root, identity="abc123", name="first"):
    save(
        root / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "id": identity,
            "name": name,
            "kind": "train",
            "status": "completed",
            "paths": {"training": "evidence/train"},
            "future_metadata": True,
        },
    )
    training = root / "evidence/train"
    save(
        training / "manifest.json",
        {"schema": "smartsom.training-manifest/v1", "status": "completed"},
    )
    save(
        training / "checkpoint_algorithm.json",
        {"algorithm": {"checkpoint": "checkpoint"}},
    )
    (training / "checkpoint").mkdir()
    (training / "progress.log").write_text("completed\n")
    save(root / "summary.json", {"status": "completed"})
    return root


def test_catalog_recognizes_new_root_once_and_legacy_without_changing_them(tmp_path):
    current = experiment(tmp_path / "runs" / "current")
    old = tmp_path / "old"
    save(old / "manifest.json", {"schema": "smartsom.manifest/v1", "status": "failed"})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    entries = list_runs([tmp_path / "runs", old])
    assert {entry.path for entry in entries} == {current, old}
    assert read_run(old).status == "failed"
    assert resolve_run("abc", tmp_path / "runs") == current
    assert {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


def test_ambiguous_name_requires_explicit_identity(tmp_path):
    experiment(tmp_path / "a", "abc1", "same")
    experiment(tmp_path / "b", "abc2", "same")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_run("same", tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_run("abc", tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        resolve_run("missing", tmp_path)


def test_rebuildable_relative_links_never_replace_or_delete_real_data(tmp_path):
    source = experiment(tmp_path / "runs" / "one")
    views = tmp_path / "views"
    index = rebuild_views(tmp_path / "runs", views)
    links = json.loads(index.read_text())["links"]
    assert set(Path(name).parts[0] for name in links) == {"models", "logs", "reports"}
    for name, target in links.items():
        assert (views / name).is_symlink()
        assert not os.path.isabs(target)
        (views / name).unlink()
    rebuild_views(tmp_path / "runs", views)
    assert all((views / name).exists() for name in links)
    (views / "models" / "user.txt").write_text("keep me")
    rebuild_views([], views)
    assert not any((views / name).is_symlink() for name in links)
    assert (views / "models/user.txt").read_text() == "keep me"
    assert artifact_paths(source)["checkpoint"].is_dir()
    assert (source / "evidence/train/progress.log").read_text() == "completed\n"


def test_cannot_add_index_inside_legacy_or_current_run(tmp_path):
    source = experiment(tmp_path / "run")
    with pytest.raises(ValueError, match="outside"):
        rebuild_views(source, source / "views")
    assert not (source / "views").exists()


@pytest.mark.parametrize("value", ["../elsewhere", "/tmp/elsewhere", 3, ""])
def test_rejects_invalid_declared_paths_before_index_creation(tmp_path, value):
    source = experiment(tmp_path / "run")
    data = json.loads((source / "run.json").read_text())
    data["paths"]["checkpoint"] = value
    save(source / "run.json", data)
    with pytest.raises(ValueError):
        rebuild_views(source, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_never_replace_unrelated_file_with_shortcut(tmp_path):
    source = experiment(tmp_path / "run")
    views = tmp_path / "views"
    index = rebuild_views(source, views)
    name = next(iter(json.loads(index.read_text())["links"]))
    (views / name).unlink()
    (views / name).write_text("mine")
    with pytest.raises(FileExistsError):
        rebuild_views(source, views)
    assert (views / name).read_text() == "mine"


def test_view_parent_symlink_cannot_write_into_source(tmp_path):
    source = experiment(tmp_path / "run")
    views = tmp_path / "views"
    views.mkdir()
    (views / "models").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="group cannot"):
        rebuild_views(source, views)
    assert not (views / "index.json").exists()
