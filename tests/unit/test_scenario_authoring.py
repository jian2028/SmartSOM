"""Human scenes retain physical worlds, identities and explicit inheritance."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import generate_scenario, load_inputs
from smartsom.config.codec import primitive, read_document
from smartsom.config.reservation import compile_reservation
from smartsom.config.scenario_authoring import ScenarioFile, load_scene, save_scene

ROOT = Path(__file__).resolve().parents[2]


def expanded_inputs(scale="small"):
    return load_scene(ROOT / f"configs/scenarios/{scale}.yaml")[0].expand()


def scene():
    return ScenarioFile.from_factory(*expanded_inputs())


@pytest.mark.parametrize("scale", ["small", "medium"])
@pytest.mark.parametrize("condition", ["normal", "low", "medium", "high"])
@pytest.mark.parametrize("seed", [101, 203])
def test_existing_world_is_exactly_preserved(scale, condition, seed):
    old_factory, old_workload = expanded_inputs(scale)
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


def test_multi_capability_and_unlabelled_due_mixture_round_trip():
    source = scene()
    data = primitive(source)
    machine = data["factory"]["machines"]["items"][0]
    operation = machine.pop("operation")
    machine["operations"] = [
        operation,
        next(o for o in data["factory"]["operation_types"] if o != operation),
    ]
    data["jobs"]["due_dates"]["mixture"] = [[0.2, 1.5, 2.5], [0.8, 2.0, 3.0]]
    changed = ScenarioFile.model_validate(data)
    envelope, workload = changed.expand()
    assert len(envelope.factory.machines[0].operation_types) == 2
    restored = ScenarioFile.from_factory(envelope, workload)
    assert restored.expand() == (envelope, workload)
    assert changed.with_design(envelope.factory).factory == envelope.factory
    generated = generate_scenario(envelope.factory, workload, 101, "normal", 2000)
    assert all(not demand.rush for demand in generated.demands)
    assert {len(demand.steps) for demand in generated.demands} == {4, 5}


@pytest.mark.parametrize(
    "scale,machines,vehicles,orders",
    [("small", 8, 10, 300), ("medium", 20, 25, 750), ("large", 40, 50, 1500)],
)
def test_v2_scenes_preserve_workload_distribution_and_balanced_flexibility(
    scale, machines, vehicles, orders
):
    from collections import Counter

    source, _ = load_scene(ROOT / f"configs/scenarios/v2/{scale}.yaml")
    envelope, workload = source.expand()
    factory = envelope.factory
    assert len(factory.machines) == machines and len(factory.agvs) == vehicles
    assert workload.orders == orders
    counts = Counter(
        op for machine in factory.machines for op in machine.operation_types
    )
    assert set(counts) == set("ABCDEFGH")
    assert set(counts.values()) == {machines // 4}
    assert all(len(machine.operation_types) == 2 for machine in factory.machines)
    pair_counts = Counter(
        tuple(sorted(machine.operation_types)) for machine in factory.machines
    )
    assert max(pair_counts.values()) - min(pair_counts.values()) <= 1
    current = generate_scenario(factory, workload, 101, "normal", 2000)
    legacy = generate_scenario(
        factory, workload.model_copy(update={"due_mixture": ()}), 101, "normal", 2000
    )
    assert current.demands == tuple(replace(d, rush=False) for d in legacy.demands)
    assert {len(d.steps) for d in current.demands} == {4, 5}
    assert workload.optional_repeat_probability == 0.5
    assert workload.workload_rho == 0.2


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
    envelope, workload = expanded_inputs()
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
    raw = read_document(source)
    raw["scenario"] = str((source.parent / raw["scenario"]).resolve())
    raw["algorithm"] = str((source.parent / raw["algorithm"]).resolve())
    new_path = tmp_path / "experiment.yaml"
    new_path.write_text(yaml.safe_dump(raw))
    after = compile_reservation(new_path)
    target = tmp_path / "frozen"
    target.mkdir()
    entry = after.materialize(target)
    assert (target / "authoring/scenario.yaml").exists()
    new_path.unlink()
    _, f, w, _, _ = load_inputs(entry)
    assert f == scene().factory
    assert w == scene().expand()[1]
    assert "scenario" in after.summary()["source_files"]


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


def condition_experiment(tmp_path):
    source = primitive(scene())
    source["jobs"]["orders"] = {"total": 4, "initial": 2}
    for index, condition in enumerate(("normal", "low", "medium", "high")):
        variant = deepcopy(source)
        variant["factory"]["machines"]["defaults"]["breakdown"]["repair"]["scale"] = (
            10 + index
        )
        (tmp_path / f"{condition}.yaml").write_text(yaml.safe_dump(variant))
    raw = {
        "schema": "smartsom.base-experiment/v1",
        "scenario": "normal.yaml",
        "condition_scenarios": {
            c: f"{c}.yaml" for c in ("normal", "low", "medium", "high")
        },
        "algorithm": str(ROOT / "configs/algorithms/rule.yaml"),
        "tracking": {"mode": "disabled"},
        "training_conditions": ["normal", "low", "medium", "high"],
        "condition_sampling": "balanced_cycle",
        "training_world_domain": "social-long-v1/world",
        "training_world_prefix": ["train"],
        "training_seed_modulus": None,
        "train_episodes": 1000,
    }
    entry = tmp_path / "experiment.yaml"
    entry.write_text(yaml.safe_dump(raw))
    return entry


def test_condition_scenarios_freeze_sources_and_identity(tmp_path, monkeypatch):
    import json

    from smartsom.experiments import reservation, reservation_runtime

    entry = condition_experiment(tmp_path)
    plan = compile_reservation(entry)
    assert len(plan.summary()["condition_scenarios"]) == 4
    assert len([x for x in plan.source_files if x[0].startswith("condition_")]) == 4
    assert len([x for x in plan.components if x[0].startswith("expanded/")]) == 4
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    compiled = plan.materialize(frozen)
    original = load_inputs(compiled)
    ex, factory, workload, _, paths = original
    assert "_condition_inputs" not in ex.model_dump()
    assert [
        ex.inputs_for(c, factory, workload)[1].repair.scale
        for c in ex.training_conditions
    ] == [10, 11, 12, 13]
    assert set(paths) >= {f"condition_{c}" for c in ex.training_conditions}
    # Author files are not consulted after materialization.
    for c in ex.training_conditions:
        (tmp_path / f"{c}.yaml").unlink()
    assert load_inputs(compiled)[0]._condition_inputs == ex._condition_inputs
    assert compile_reservation(compiled).scientific_sha256 == plan.scientific_sha256
    source = {
        "environment": {"platform": "mock"},
        "implementation_sha256": "test",
        "commit": "test",
    }
    monkeypatch.setattr(reservation_runtime, "source_identity", lambda: source)
    monkeypatch.setattr(
        "smartsom.experiments.evidence.runtime_source_matches", lambda *args: True
    )
    # Existing snapshot/resume validator must bind every condition file.
    root = tmp_path / "run"
    reservation.allocate(plan, root=root)
    reservation.load(root)
    target = root / "inputs/condition_high.yaml"
    target.write_text(target.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="frozen input identity"):
        reservation.load(root)
    document = read_document(frozen / "condition_high.yaml")
    document["factory"]["machines"]["defaults"]["breakdown"]["repair"]["scale"] = 13.0
    (frozen / "condition_high.yaml").write_text(yaml.safe_dump(document))
    assert compile_reservation(compiled).scientific_sha256 == plan.scientific_sha256
    document["factory"]["machines"]["defaults"]["breakdown"]["repair"]["scale"] = 17
    (frozen / "condition_high.yaml").write_text(yaml.safe_dump(document))
    assert compile_reservation(compiled).scientific_sha256 != plan.scientific_sha256
    assert json.loads((root / "plan.json").read_text())["files"][
        "inputs/condition_high.yaml"
    ]


@pytest.mark.parametrize("mutation", ["missing", "unknown", "jobs", "factory"])
def test_condition_scenarios_reject_invalid_scope(tmp_path, mutation):
    entry = condition_experiment(tmp_path)
    raw = read_document(entry)
    if mutation == "missing":
        raw["condition_scenarios"].pop("high")
    elif mutation == "unknown":
        raw["condition_scenarios"]["fifth"] = "high.yaml"
    else:
        variant = read_document(tmp_path / "high.yaml")
        if mutation == "jobs":
            variant["jobs"]["orders"]["total"] += 1
        else:
            variant["factory"]["machines"]["defaults"]["processing_rate_multiplier"] = (
                "2"
            )
        (tmp_path / "high.yaml").write_text(yaml.safe_dump(variant))
    entry.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        compile_reservation(entry)


def test_condition_fixed_cases_bind_selected_workload(tmp_path):
    import json

    from smartsom.config.codec import digest
    from smartsom.config.evaluation_cases import condition_input_identity, load_cases

    entry = condition_experiment(tmp_path)
    ex, factory, workload, _, _ = load_inputs(entry)
    cases = []
    for index in range(40):
        condition = ex.training_conditions[index % 4]
        world = generate_scenario(
            *ex.inputs_for(condition, factory, workload), 50 + index, condition, 2000
        )
        cases.append(
            dict(
                id=f"{condition}-{index}",
                seed=50 + index,
                condition=condition,
                action_seeds=[91],
                scenario=primitive(world),
            )
        )
    data = dict(
        schema="smartsom.fixed-cases/v1",
        split="test",
        factory_sha256=digest(primitive(factory)),
        workload_sha256=digest(workload.model_dump(mode="json")),
        condition_inputs_sha256=condition_input_identity(ex._condition_inputs),
        cases=cases,
    )
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))
    assert (
        load_cases(
            path,
            split="test",
            factory=factory,
            workload=workload,
            condition_inputs=ex._condition_inputs,
        )
        == cases
    )
    raw = read_document(entry)
    raw["case_sets"] = {"test": "test.json"}
    entry.write_text(yaml.safe_dump(raw))
    plan = compile_reservation(entry)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    compiled = plan.materialize(frozen)
    loaded, f, w, _, _ = load_inputs(compiled)
    assert (
        load_cases(
            frozen / loaded.case_sets["test"],
            split="test",
            factory=f,
            workload=w,
            condition_inputs=loaded._condition_inputs,
        )
        == cases
    )
    frozen_inventory = frozen / loaded.case_sets["test"]
    frozen_data = json.loads(frozen_inventory.read_text())
    frozen_data["cases"][0]["action_seeds"] = [999]
    frozen_inventory.write_text(json.dumps(frozen_data))
    with pytest.raises(ValueError, match="inventory digest"):
        compile_reservation(compiled)
    wrong = deepcopy(data)
    wrong["cases"][3]["scenario"] = primitive(
        generate_scenario(factory, workload, 53, "high", 2000)
    )
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="condition scenario/seed"):
        load_cases(
            path,
            split="test",
            factory=factory,
            workload=workload,
            condition_inputs=ex._condition_inputs,
        )
    data.pop("condition_inputs_sha256")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="condition scenarios"):
        load_cases(
            path,
            split="test",
            factory=factory,
            workload=workload,
            condition_inputs=ex._condition_inputs,
        )


def test_condition_training_cycle_keeps_seed_stream_and_one_model(
    tmp_path, monkeypatch
):
    import hashlib
    from collections import Counter
    from types import SimpleNamespace

    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.reservation_session import ReservationSession

    ex, factory, workload, algorithm, _ = load_inputs(condition_experiment(tmp_path))
    seen = []

    def episode(f, w, a, policy, seed, condition, drain, **kwargs):
        seen.append((seed, condition, w.repair.scale, id(policy)))
        return {"status": "completed", "tick": 1}, [0.0], None

    monkeypatch.setattr(runtime, "episode", episode)
    streams = {}
    for seed in (4101, 4102, 4103):
        for arm in ("base", "all"):
            session = ReservationSession.__new__(ReservationSession)
            session.ex, session.factory, session.workload, session.options = (
                ex,
                factory,
                workload,
                algorithm,
            )
            session.args = SimpleNamespace(seed=seed)
            session.rows, session.ticks = [], 0
            session.deadline = float("inf")
            session.drain, session.replay_source = 2000, None
            session.root = tmp_path
            session.policy = SimpleNamespace(
                update=lambda *args, **kw: None, updates=0, last_losses={}
            )
            session.tracking = SimpleNamespace(log=lambda *args: None)
            session.commit = lambda: None
            session.evaluate_periodically = lambda: None
            session.progress = lambda: {"status": "training"}
            seen.clear()
            for _ in range(1000):
                session.step()
            assert Counter(c for _, c, _, _ in seen) == dict.fromkeys(
                ex.training_conditions, 250
            )
            assert len({owner for _, _, _, owner in seen}) == 1
            expected = [
                int.from_bytes(
                    hashlib.sha256(
                        repr((seed, ("social-long-v1/world", "train", i))).encode()
                    ).digest()[:8],
                    "big",
                )
                for i in range(1000)
            ]
            assert [s for s, _, _, _ in seen] == expected
            assert [scale for _, _, scale, _ in seen[:4]] == [10, 11, 12, 13]
            streams[seed, arm] = list(seen)
        assert [x[:3] for x in streams[seed, "base"]] == [
            x[:3] for x in streams[seed, "all"]
        ]


@pytest.mark.parametrize("condition", ["normal", "low", "medium", "high"])
def test_condition_evaluation_selects_frozen_case_inputs(
    tmp_path, monkeypatch, condition
):
    import json
    from types import SimpleNamespace

    from smartsom.config.codec import digest
    from smartsom.config.evaluation_cases import condition_input_identity
    from smartsom.experiments import reservation_runtime as runtime

    entry = condition_experiment(tmp_path)
    ex, factory, workload, _, _ = load_inputs(entry)
    seed = 73
    selected_factory, selected_workload = ex.inputs_for(condition, factory, workload)
    world = generate_scenario(
        selected_factory, selected_workload, seed, condition, 2000
    )
    cases = [
        dict(
            id="one",
            condition=condition,
            seed=seed,
            action_seeds=[81],
            scenario=primitive(world),
        )
    ]
    (tmp_path / "test.json").write_text(
        json.dumps(
            dict(
                schema="smartsom.fixed-cases/v1",
                split="test",
                factory_sha256=digest(primitive(factory)),
                workload_sha256=digest(workload.model_dump(mode="json")),
                condition_inputs_sha256=condition_input_identity(ex._condition_inputs),
                cases=cases,
            )
        )
    )
    raw = read_document(entry)
    raw.update(case_sets={"test": "test.json"}, drain_frozen=True)
    entry.write_text(yaml.safe_dump(raw))
    frozen = tmp_path / "inputs"
    frozen.mkdir()
    compiled = compile_reservation(entry).materialize(frozen)
    for c in ex.training_conditions:
        (tmp_path / f"{c}.yaml").unlink()
    args = SimpleNamespace(
        experiment=str(compiled),
        output=str(tmp_path / "evaluation"),
        method=["rule"],
        checkpoint=None,
        selection="argmax",
        split="test",
        cases=1,
        action_seeds=1,
        seconds=None,
        calibration=None,
        native=True,
        tracking=SimpleNamespace(),
        numerical_threads=1,
    )
    monkeypatch.setattr(
        runtime,
        "source_identity",
        lambda: {"environment": {"python": "test", "packages": {}}},
    )
    monkeypatch.setattr(
        runtime, "freeze", lambda *a, **kw: {"source": {}, "started_unix": 0}
    )
    monkeypatch.setattr(
        runtime, "case_store", lambda *a, **kw: SimpleNamespace(rows={}, cases=[{}])
    )
    monkeypatch.setattr(
        "smartsom.algorithms.base_policy.build_policy", lambda *a, **kw: object()
    )

    class Selected(Exception):
        pass

    def episode(f, w, options, policy, actual_seed, actual_condition, drain, **kwargs):
        assert actual_seed == seed and actual_condition == condition
        assert f == selected_factory and w == selected_workload
        assert kwargs["scenario"] == world
        raise Selected

    monkeypatch.setattr(runtime, "episode", episode)
    with pytest.raises(Selected):
        runtime._evaluate(args)


def test_condition_mapping_enters_comparison_and_reference_identity(tmp_path):
    from smartsom.telemetry.evaluation_summary import (
        reference_definition,
        sl_comparison_identity,
    )

    entry = condition_experiment(tmp_path)
    raw = read_document(entry)
    raw["algorithm"] = str(ROOT / "configs/algorithms/base-dqn.yaml")
    entry.write_text(yaml.safe_dump(raw))
    frozen = tmp_path / "inputs"
    frozen.mkdir()
    compiled = compile_reservation(entry).materialize(frozen)
    ex, f, w, a, paths = load_inputs(compiled)
    before = sl_comparison_identity(a, ex, factory=f, workload=w)
    reference = reference_definition(ex, f, w, a)
    doc = read_document(frozen / "condition_high.yaml")
    doc["factory"]["machines"]["defaults"]["breakdown"]["repair"]["scale"] += 1
    (frozen / "condition_high.yaml").write_text(yaml.safe_dump(doc))
    ex2, f2, w2, a2, paths2 = load_inputs(compiled)
    assert (
        sl_comparison_identity(a2, ex2, factory=f2, workload=w2)["non_sl_sha256"]
        != before["non_sl_sha256"]
    )
    assert reference_definition(ex2, f2, w2, a2)["sha256"] != reference["sha256"]


@pytest.mark.parametrize("one_path", [False, True])
def test_condition_scenes_can_preserve_identical_complete_fault_profiles(
    tmp_path, one_path
):
    entry = condition_experiment(tmp_path)
    source = primitive(scene())
    source["jobs"]["orders"] = {"total": 4, "initial": 2}
    for condition in ("normal", "low", "medium", "high"):
        (tmp_path / f"{condition}.yaml").write_text(yaml.safe_dump(source))
    if one_path:
        raw = read_document(entry)
        raw["condition_scenarios"] = dict.fromkeys(
            raw["condition_scenarios"], "normal.yaml"
        )
        entry.write_text(yaml.safe_dump(raw))
    frozen = tmp_path / "inputs"
    frozen.mkdir()
    compiled = compile_reservation(entry).materialize(frozen)
    ex, factory, workload, _, _ = load_inputs(compiled)
    old_factory, old_workload = expanded_inputs()
    old_workload = old_workload.model_copy(update={"orders": 4, "initial_orders": 2})
    for condition in ex.training_conditions:
        selected = ex.inputs_for(condition, factory, workload)
        assert selected[1].fault_intervals == old_workload.fault_intervals
        assert generate_scenario(*selected, 4101, condition, 2000) == generate_scenario(
            old_factory.factory, old_workload, 4101, condition, 2000
        )


def test_condition_continuation_uses_committed_cycle_and_frozen_inputs(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.reservation_session import ReservationSession

    entry = condition_experiment(tmp_path)
    raw = read_document(entry)
    raw.update(
        algorithm=str(ROOT / "configs/algorithms/base-dqn.yaml"), drain_frozen=True
    )
    entry.write_text(yaml.safe_dump(raw))
    frozen = tmp_path / "inputs"
    frozen.mkdir()
    compiled = compile_reservation(entry).materialize(frozen)
    for condition in raw["condition_scenarios"]:
        (tmp_path / f"{condition}.yaml").unlink()

    class Policy:
        updates = 0
        last_losses = {}

        def update(self, *args, **kwargs):
            self.updates += 1

        def save(self, path):
            path.write_text(json.dumps({"updates": self.updates}))

        def load(self, path):
            self.updates = json.loads(path.read_text())["updates"]

    class Tracking:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *args):
            pass

        def log(self, *args):
            pass

        def close(self, **kwargs):
            pass

    observed = []

    def episode(f, w, a, policy, seed, condition, drain, **kwargs):
        observed.append((seed, condition, w.repair.scale))
        return {"status": "completed", "tick": 1}, [0.0], None

    source = {
        "environment": {"platform": "mock"},
        "implementation_sha256": "test",
        "commit": "test",
    }
    monkeypatch.setattr(runtime, "source_identity", lambda: source)
    monkeypatch.setattr(
        "smartsom.experiments.evidence.runtime_source_matches", lambda *args: True
    )
    monkeypatch.setattr(
        "smartsom.algorithms.base_policy.build_policy", lambda *args, **kwargs: Policy()
    )
    monkeypatch.setattr(
        "smartsom.telemetry.experiment_tracking.ExperimentTracking", Tracking
    )
    monkeypatch.setattr(runtime, "episode", episode)
    monkeypatch.setattr(ReservationSession, "evaluate_periodically", lambda self: None)
    monkeypatch.setattr(
        ReservationSession, "progress", lambda self: {"status": "training"}
    )
    args = SimpleNamespace(
        experiment=compiled,
        output=tmp_path / "run",
        seed=4101,
        ticks=None,
        seconds=None,
        calibration=None,
    )
    first = ReservationSession(args)
    first.step()
    first.step()
    saved_context = json.loads((first.root / "last.pt.json").read_text())[
        "training_context"
    ]
    assert saved_context == first.training_context
    assert saved_context["definition"]["condition_inputs"]
    first.close()
    # A partial write without pointer promotion is not a committed episode.
    partial = first.checkpoints / "partial"
    partial.mkdir()
    (partial / "model.pt").write_text('{"updates": 999}')
    resumed = ReservationSession(args)
    assert resumed.policy.updates == 2 and len(resumed.rows) == 2
    assert resumed.training_context == saved_context
    resumed.step()
    resumed.step()
    resumed.close()
    assert [(c, scale) for _, c, scale in observed] == [
        ("normal", 10),
        ("low", 11),
        ("medium", 12),
        ("high", 13),
    ]
    assert resumed.policy.updates == 4 and len(resumed.rows) == 4
    target = frozen / "condition_high.yaml"
    target.write_text(target.read_text() + "\n# external change\n")
    with pytest.raises(ValueError, match="frozen source/input identity"):
        ReservationSession(args)


def test_condition_scene_cache_does_not_leak_mutable_workloads(tmp_path):
    import hashlib

    from smartsom.config.base_diagnostic import MachineFaults

    entry = condition_experiment(tmp_path)
    raw = read_document(entry)
    raw["condition_scenarios"] = dict.fromkeys(
        raw["condition_scenarios"], "normal.yaml"
    )
    entry.write_text(yaml.safe_dump(raw))
    source_hash = hashlib.sha256((tmp_path / "normal.yaml").read_bytes()).hexdigest()
    scientific = compile_reservation(entry).scientific_sha256
    first, f1, w1, _, _ = load_inputs(entry)
    second, f2, w2, _, _ = load_inputs(entry)
    selected = first.inputs_for("low", f1, w1)[1]
    other = second.inputs_for("low", f2, w2)[1]
    assert selected is not other
    assert selected.fault_intervals is not other.fault_intervals
    scale = other.fault_intervals["low"].scale
    selected.fault_intervals["low"] = selected.fault_intervals["low"].model_copy(
        update={"scale": 777.0}
    )
    selected.machine_faults[f1.machines[0].machine_id] = MachineFaults(enabled=False)
    assert first.inputs_for("high", f1, w1)[1].fault_intervals["low"].scale == scale
    assert not first.inputs_for("high", f1, w1)[1].machine_faults
    assert other.fault_intervals["low"].scale == scale and not other.machine_faults
    third, f3, w3, _, _ = load_inputs(entry)
    reloaded = third.inputs_for("low", f3, w3)[1]
    assert (
        reloaded.fault_intervals["low"].scale == scale and not reloaded.machine_faults
    )
    assert (
        hashlib.sha256((tmp_path / "normal.yaml").read_bytes()).hexdigest()
        == source_hash
    )
    assert compile_reservation(entry).scientific_sha256 == scientific


def test_checkpoint_training_context_validates_sidecar_and_preserves_ood(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from types import SimpleNamespace

    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.telemetry.evaluation_summary import training_context_identity

    entry = condition_experiment(tmp_path)
    raw = read_document(entry)
    raw["algorithm"] = str(ROOT / "configs/algorithms/base-dqn.yaml")
    entry.write_text(yaml.safe_dump(raw))
    ex, f, w, algorithm, paths = load_inputs(entry)
    context = training_context_identity(ex, f, w, algorithm, paths)
    raw["algorithm"] = str(ROOT / "configs/algorithms/sl-dqn.yaml")
    entry.write_text(yaml.safe_dump(raw))
    sl_ex, sl_f, sl_w, sl_a, sl_paths = load_inputs(entry)
    assert (
        training_context_identity(sl_ex, sl_f, sl_w, sl_a, sl_paths)["sha256"]
        == context["sha256"]
    )
    # Independent training profile changes must change context identity.
    variant = read_document(tmp_path / "high.yaml")
    variant["factory"]["machines"]["defaults"]["breakdown"]["repair"]["scale"] += 1
    (tmp_path / "high.yaml").write_text(yaml.safe_dump(variant))
    ex2, f2, w2, a2, p2 = load_inputs(entry)
    assert training_context_identity(ex2, f2, w2, a2, p2)["sha256"] != context["sha256"]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    evidence = {
        "schema": "smartsom.checkpoint-evidence/v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "train_seed": 4101,
        "episodes": 1000,
        "physical_ticks": 1234,
        "source_sha256": "source",
        "run_id": "run",
        "training_context": context,
    }
    sidecar = checkpoint.with_suffix(".pt.json")
    sidecar.write_text(json.dumps(evidence))
    monkeypatch.setattr(
        runtime, "source_identity", lambda: {"implementation_sha256": "source"}
    )
    args = SimpleNamespace(
        checkpoint=checkpoint, experiment=entry, training_progress=None
    )
    runtime.checkpoint_training_evidence(args)
    assert (
        args.training_context == context
    )  # eval environment has changed: OOD remains legal
    assert args.training_progress["train_seed"] == 4101
    with pytest.raises(ValueError, match="training progress"):
        runtime.checkpoint_training_evidence(
            SimpleNamespace(checkpoint=checkpoint, training_progress={"episodes": 1})
        )
    with pytest.raises(ValueError, match="training context"):
        runtime.checkpoint_training_evidence(
            SimpleNamespace(
                checkpoint=checkpoint,
                training_context=training_context_identity(ex2, f2, w2, a2, p2),
            )
        )
    evidence["training_context"]["definition"]["factory"] = "tampered"
    sidecar.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="training context"):
        runtime.checkpoint_training_evidence(SimpleNamespace(checkpoint=checkpoint))
    evidence.pop("training_context")
    sidecar.write_text(json.dumps(evidence))
    legacy = SimpleNamespace(checkpoint=checkpoint)
    runtime.checkpoint_training_evidence(legacy)
    assert legacy.training_context is None
