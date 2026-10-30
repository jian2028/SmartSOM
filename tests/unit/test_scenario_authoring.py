"""Human scenes retain physical worlds, identities and explicit inheritance."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import BaseWorkload, generate_scenario, load_inputs
from smartsom.config.codec import primitive, read_document
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.reservation import compile_reservation
from smartsom.config.scenario_authoring import ScenarioFile, load_scene, save_scene

ROOT = Path(__file__).resolve().parents[2]


def old_inputs(scale="small"):
    factory, _ = load_factory_design_file(ROOT / f"configs/factories/{scale}.yaml")
    workload = BaseWorkload.model_validate(
        read_document(ROOT / f"configs/workloads/{scale}.yaml")
    )
    return factory, workload


def scene():
    return ScenarioFile.from_factory(*old_inputs())


@pytest.mark.parametrize("scale", ["small", "medium"])
@pytest.mark.parametrize("condition", ["normal", "low", "medium", "high"])
@pytest.mark.parametrize("seed", [101, 203])
def test_existing_world_is_exactly_preserved(scale, condition, seed):
    old_factory, old_workload = old_inputs(scale)
    source = ScenarioFile.from_factory(old_factory, old_workload)
    factory, workload = source.expand()
    assert primitive(factory.factory) == primitive(old_factory.factory)
    assert workload.model_dump(mode="json") == old_workload.model_dump(mode="json")
    assert generate_scenario(
        factory.factory, workload, seed, condition, 2000
    ) == generate_scenario(old_factory.factory, old_workload, seed, condition, 2000)


def test_defaults_overrides_and_save_do_not_flatten(tmp_path):
    source = scene()
    data = primitive(source)
    machine = data["factory"]["machines"]["items"][0]
    machine["overrides"] = {"processing_rate_multiplier": "0.8"}
    data["factory"]["agvs"]["items"][1]["overrides"] = {"move_cells_per_tick": 2}
    source = ScenarioFile.model_validate(data)
    target = tmp_path / "scenario.yaml"
    digest = save_scene(target, source)
    loaded, actual = load_scene(target)
    assert digest == actual
    assert loaded == source
    assert loaded.factory.machines[0].processing_rate_multiplier == Decimal("0.8")
    assert loaded.factory.agvs[1].move_cells_per_tick == 2
    assert "overrides" not in read_document(target)["factory"]["machines"]["items"][1]
    assert "rotation" not in machine
    with pytest.raises(ValueError, match="already exists"):
        save_scene(target, source)
    target.write_text(target.read_text() + "\n# external edit\n")
    with pytest.raises(ValueError, match="changed externally"):
        save_scene(target, source, expected_digest=digest)


def test_fault_override_leaves_jobs_and_other_fault_streams_unchanged():
    source = scene()
    before_f, before_w = source.expand()
    data = primitive(source)
    data["factory"]["machines"]["items"][0]["overrides"] = {
        "breakdown": {"enabled": False}
    }
    f, w = ScenarioFile.model_validate(data).expand()
    before = generate_scenario(before_f.factory, before_w, 101, "high", 2000)
    after = generate_scenario(f.factory, w, 101, "high", 2000)
    assert before.demands == after.demands
    assert after.outages == tuple(
        o for o in before.outages if o.machine_id != f.factory.machines[0].machine_id
    )


def test_edit_keeps_explicit_equal_override_and_partial_fault_override():
    data = primitive(scene())
    data["factory"]["machines"]["items"][0]["overrides"] = {
        "processing_rate_multiplier": "1",
        "breakdown": {"enabled": False},
    }
    source = ScenarioFile.model_validate(data)
    design = source.factory
    moved = replace(design, name="Changed display name")
    result = source.with_design(moved)
    assert (
        result.factory_spec["machines"]["items"][0]["overrides"]
        == data["factory"]["machines"]["items"][0]["overrides"]
    )
    changed = primitive(result)
    changed["factory"]["machines"]["defaults"]["processing_rate_multiplier"] = "2"
    updated = ScenarioFile.model_validate(changed).factory
    assert updated.machines[0].processing_rate_multiplier == 1
    assert updated.machines[1].processing_rate_multiplier == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("size", {"width": 2, "height": 1}),
        ("rotation", 90),
        ("operation", ["A", "B"]),
        ("position", {"x": 3}),
    ],
)
def test_explicit_machine_contract_rejects_invalid_fields(field, value):
    data = primitive(scene())
    data["factory"]["machines"]["items"][0][field] = value
    with pytest.raises(ValueError):
        ScenarioFile.model_validate(data)


def test_non_single_cell_legacy_geometry_is_not_silently_shrunk():
    envelope, workload = old_inputs()
    first = envelope.factory.machines[0]
    machines = (
        replace(first, footprint=replace(first.footprint, width=2)),
        *envelope.factory.machines[1:],
    )
    envelope = envelope.model_copy(
        update={"factory": replace(envelope.factory, machines=machines)}
    )
    with pytest.raises(ValueError, match="not resized"):
        ScenarioFile.from_factory(envelope, workload)


def test_compile_freeze_reload_and_science_equivalence(tmp_path):
    source = ROOT / "experiments/examples/current-smoke/default/experiment.yaml"
    # This also accepts the migrated example: construct an independent legacy
    # comparison from the same explicit reference files.
    raw = read_document(source)
    raw.pop("scenario", None)
    raw.update(
        factory=str(ROOT / "configs/factories/small.yaml"),
        workload=str(ROOT / "configs/workloads/small.yaml"),
        algorithm=str((source.parent / raw["algorithm"]).resolve()),
    )
    old_path = tmp_path / "old.yaml"
    old_path.write_text(yaml.safe_dump(raw))
    save_scene(tmp_path / "scenario.yaml", scene())
    new = deepcopy(raw)
    new.pop("factory")
    new.pop("workload")
    new["scenario"] = "scenario.yaml"
    new_path = tmp_path / "new.yaml"
    new_path.write_text(yaml.safe_dump(new))
    before = compile_reservation(old_path)
    after = compile_reservation(new_path)
    # Output is operational; both worlds/learners have the same scientific ID.
    assert before.scientific_sha256 == after.scientific_sha256
    target = tmp_path / "frozen"
    target.mkdir()
    entry = after.materialize(target)
    assert (target / "authoring/scenario.yaml").exists()
    (tmp_path / "scenario.yaml").unlink()
    _, f, w, _, _ = load_inputs(entry)
    assert f == scene().factory
    assert w == scene().expand()[1]
    assert "scenario" in after.summary()["source_files"]


def test_explicit_scene_migration_is_create_only_and_preserves_inputs(tmp_path):
    from smartsom.config.scenario_migration import migrate_scene_experiment
    from smartsom.experiments.launch import compile_launch

    raw = {
        "schema": "smartsom.base-experiment/v1",
        "factory": str(ROOT / "configs/factories/small.yaml"),
        "workload": str(ROOT / "configs/workloads/small.yaml"),
        "algorithm": str(ROOT / "configs/algorithms/base-ppo.yaml"),
        "matrix": {"workload.orders": [128, 256]},
    }
    path = tmp_path / "old.yaml"
    original = yaml.safe_dump(raw)
    path.write_text(original)
    out = tmp_path / "new"
    migrate_scene_experiment(path, out)
    assert path.read_text() == original
    migrated = read_document(out / "experiment.yaml")
    assert migrated["matrix"] == {"scenario.jobs.orders.total": [128, 256]}
    assert "factory" not in migrated
    assert load_scene(out / "scenario.yaml")[0].expand()[1] == old_inputs()[1]
    plans = compile_launch(out / "experiment.yaml")
    assert [p["compiled"].orders for p in plans.files] == [128, 256]
    with pytest.raises(ValueError, match="already exists"):
        migrate_scene_experiment(path, out)


def test_default_fault_disable_and_one_machine_enable():
    data = primitive(scene())
    data["factory"]["machines"]["defaults"]["breakdown"]["enabled"] = False
    data["factory"]["machines"]["items"][0]["overrides"] = {
        "breakdown": {"enabled": True}
    }
    factory, workload = ScenarioFile.model_validate(data).expand()
    world = generate_scenario(factory.factory, workload, 101, "high", 2000)
    assert world.outages
    assert {f.machine_id for f in world.outages} == {
        factory.factory.machines[0].machine_id
    }


def test_rename_preserves_fault_override_and_new_entities_inherit():
    data = primitive(scene())
    data["factory"]["machines"]["defaults"]["processing_rate_multiplier"] = "1.7"
    data["factory"]["agvs"]["defaults"]["move_cells_per_tick"] = 3
    data["factory"]["machines"]["items"][0]["overrides"] = {
        "breakdown": {"enabled": False}
    }
    source = ScenarioFile.model_validate(data)
    from smartsom.studio.editing import rename

    old_id = source.factory.machines[0].machine_id
    renamed = rename(source.factory, old_id, "renamed_machine")
    result = source.with_design(renamed, renames={old_id: "renamed_machine"})
    assert result.expand()[1].machine_faults["renamed_machine"].enabled is False
    assert old_id not in result.expand()[1].machine_faults
    # Placement receives ordinary domain defaults, then applies the scene defaults.
    design = source.factory
    machine, agv = design.machines[0], design.agvs[0]
    candidate = replace(
        design,
        machines=(
            replace(machine, processing_rate_multiplier=Decimal(1)),
            *design.machines[1:],
        ),
        agvs=(replace(agv, move_cells_per_tick=1), *design.agvs[1:]),
    )
    inherited = source.inherit_new(candidate, [machine.machine_id, agv.agv_id])
    assert inherited.machines[0].processing_rate_multiplier == Decimal("1.7")
    assert inherited.agvs[0].move_cells_per_tick == 3


def test_omitted_defaults_are_editable_and_agv_size_is_rejected():
    data = primitive(scene())
    for section in ("machines", "agvs"):
        data["factory"][section].pop("defaults")
    data["factory"].pop("buffer_defaults")
    result = ScenarioFile.model_validate(data)
    assert result.factory_spec["agvs"]["defaults"]["job_capacity"] == 1
    assert result.factory_spec["machines"]["defaults"]["breakdown"]["enabled"]
    data["factory"]["agvs"]["items"][0]["size"] = [2, 1]
    with pytest.raises(ValueError, match="unknown fields"):
        ScenarioFile.model_validate(data)


def test_numeric_yaml_defaults_do_not_become_entity_overrides_on_edit():
    data = primitive(scene())
    defaults = data["factory"]["machines"]["defaults"]
    defaults["processing_rate_multiplier"] = 1.0
    for mode in defaults["quality_modes"]:
        mode["time_scale"] = float(mode["time_scale"])
        mode["error_rate"] = float(mode["error_rate"])
    original = ScenarioFile.model_validate(data)
    saved = original.with_design(replace(original.factory, name="New name"))
    assert saved.factory_spec["machines"] == original.factory_spec["machines"]


@pytest.mark.parametrize("storage", [{"mode": "pool"}, {"capacity": 4}, {}])
def test_buffer_storage_preserves_inheritance_and_explicit_pins(storage):
    data = primitive(scene())
    buffer = next(b for b in data["factory"]["buffers"] if b["role"] == "machine_pre")
    buffer["storage"] = storage
    source = ScenarioFile.model_validate(data)
    saved = source.with_design(replace(source.factory, name="Unrelated edit"))
    raw = primitive(saved)
    buffer_id = buffer["buffer_id"]
    authored = next(b for b in raw["factory"]["buffers"] if b["buffer_id"] == buffer_id)
    assert authored.get("storage", {}) == storage
    raw["factory"]["buffer_defaults"]["machine_pre"]["storage"]["capacity"] = 9
    changed = ScenarioFile.model_validate(raw)
    actual = next(b for b in changed.factory.buffers if b.buffer_id == buffer_id)
    assert actual.storage.capacity == (4 if "capacity" in storage else 9)
    design = source.factory
    buffers = tuple(
        replace(b, storage=replace(b.storage, capacity=7))
        if b.buffer_id == buffer_id
        else b
        for b in design.buffers
    )
    edited = source.with_design(replace(design, buffers=buffers))
    assert (
        next(
            b for b in edited.factory.buffers if b.buffer_id == buffer_id
        ).storage.capacity
        == 7
    )


def test_buffer_storage_mode_change_replaces_definition():
    from smartsom.studio.editing import convert_storage

    source = scene()
    design = source.factory
    buffer = next(b for b in design.buffers if b.role == "machine_pre")
    saved = source.with_design(convert_storage(design, buffer.buffer_id, "slots", 2))
    raw = next(
        b for b in saved.factory_spec["buffers"] if b["buffer_id"] == buffer.buffer_id
    )
    assert raw["storage"]["mode"] == "slots"
    assert "capacity" not in raw["storage"]
    assert raw["storage"]["slots"][0]["capacity"] == 2


def test_scene_matrix_rejects_input_reference_axis(tmp_path):
    from smartsom.config.codec import ConfigurationError
    from smartsom.experiments.launch import compile_launch

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"matrix": {"experiment.scenario": ["other.yaml"]}}))
    with pytest.raises(ConfigurationError, match="invalid scientific matrix axis"):
        compile_launch(path)


@pytest.mark.parametrize("multifile", [False, True])
@pytest.mark.parametrize("leaf_axis", [False, True])
def test_scene_migration_copies_matrix_only_components(tmp_path, multifile, leaf_axis):
    import shutil

    from smartsom.config.scenario_migration import migrate_scene_experiment
    from smartsom.experiments.launch import compile_launch

    source = tmp_path / "original"
    source.mkdir()
    algorithm = read_document(ROOT / "configs/algorithms/rule.yaml")
    specs = []
    for name, implementation in [("one", "rule"), ("two", "random")]:
        package = source / name
        package.mkdir()
        entry = package / "policy.py"
        if multifile:
            entry.write_text("from .helper import create_policy\n")
            (package / "helper.py").write_text(
                f"from smartsom.policies.{implementation} import create_policy\n"
            )
            (package / "resource.txt").write_text(name)
        else:
            entry.write_text(
                f"from smartsom.policies.{implementation} import create_policy\n"
            )
        spec = {"path": f"{name}/policy.py", "factory": "create_policy"}
        if multifile:
            spec["source_root"] = name
        specs.append(spec)
    key = "algorithm.production.policies.rule.implementation"
    matrix = {key: specs}
    if leaf_axis:
        # Same bundle and root, but an alternative entry file referenced only
        # by a scalar matrix value rather than a complete component mapping.
        (source / "one/alternate.py").write_text(
            "from smartsom.policies.random import create_policy\n"
        )
        algorithm["production"]["policies"]["rule"]["implementation"] = specs[0]
        matrix = {key + ".path": ["one/policy.py", "one/alternate.py"]}
    (source / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    raw = {
        "schema": "smartsom.base-experiment/v1",
        "factory": str(ROOT / "configs/factories/small.yaml"),
        "workload": str(ROOT / "configs/workloads/small.yaml"),
        "algorithm": "algorithm.yaml",
        "matrix": matrix,
    }
    entry = source / "experiment.yaml"
    entry.write_text(yaml.safe_dump(raw))
    before = compile_launch(entry)
    out = tmp_path / "migrated"
    migrate_scene_experiment(entry, out)
    shutil.rmtree(source)
    after = compile_launch(out / "experiment.yaml")
    assert len(after.files) == len(before.files) == 2
    assert [p["compiled"].scientific_sha256 for p in after.files] == [
        p["compiled"].scientific_sha256 for p in before.files
    ]
    if multifile:
        assert list(out.glob("components/extensions/*/resource.txt"))


def test_scene_migration_preserves_same_entry_with_different_bundle_scopes(tmp_path):
    import shutil

    from smartsom.config.scenario_migration import migrate_scene_experiment
    from smartsom.experiments.launch import compile_launch

    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    (package / "policy.py").write_text(
        "from smartsom.policies.rule import create_policy\n"
    )
    (source / "resource.txt").write_text("outer resource")
    (source / "algorithm.yaml").write_text(
        (ROOT / "configs/algorithms/rule.yaml").read_text()
    )
    entry = source / "experiment.yaml"
    entry.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-experiment/v1",
                "factory": str(ROOT / "configs/factories/small.yaml"),
                "workload": str(ROOT / "configs/workloads/small.yaml"),
                "algorithm": "algorithm.yaml",
                "matrix": {
                    "algorithm.production.policies.rule.implementation": [
                        {
                            "path": "pkg/policy.py",
                            "source_root": root,
                            "factory": "create_policy",
                        }
                        for root in ("pkg", ".")
                    ]
                },
            }
        )
    )
    before = compile_launch(entry)
    target = tmp_path / "migrated"
    migrate_scene_experiment(entry, target)
    shutil.rmtree(source)
    after = compile_launch(target / "experiment.yaml")
    assert [p["compiled"].scientific_sha256 for p in after.files] == [
        p["compiled"].scientific_sha256 for p in before.files
    ]


def test_scene_migration_rejects_correlated_scalar_bundle_relocation(tmp_path):
    from smartsom.config.scenario_migration import migrate_scene_experiment
    from smartsom.experiments.launch import compile_launch

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "policy.py").write_text(
        "from smartsom.policies.rule import create_policy\n"
    )
    algorithm = read_document(ROOT / "configs/algorithms/rule.yaml")
    algorithm["production"]["policies"]["rule"]["implementation"] = {
        "path": "pkg/policy.py",
        "factory": "create_policy",
        "source_root": "pkg",
    }
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    entry = tmp_path / "experiment.yaml"
    entry.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-experiment/v1",
                "factory": str(ROOT / "configs/factories/small.yaml"),
                "workload": str(ROOT / "configs/workloads/small.yaml"),
                "algorithm": "algorithm.yaml",
                "matrix": {
                    "algorithm.production.policies.rule.implementation.source_root": [
                        "pkg",
                        ".",
                    ]
                },
            }
        )
    )
    assert len(compile_launch(entry).files) == 2
    target = tmp_path / "migrated"
    with pytest.raises(ValueError, match="use complete component declarations"):
        migrate_scene_experiment(entry, target)
    assert not target.exists()
