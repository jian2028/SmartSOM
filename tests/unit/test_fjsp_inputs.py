import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
from test_experiments import ROOT, edit, json_file, run_path
from test_experiments import bundle as bundle

from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, normalize_workload, primitive
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.experiments import run_one
from smartsom.experiments.cli import main
from smartsom.workloads import (
    IntegerRange,
    StaticFJSPProfile,
    generate_fjsp,
    import_fjs,
)


def test_import_keeps_duplicate_pairs_and_is_independent_of_alternative_order(tmp_path):
    path = tmp_path / "input.fjs"
    path.write_text("1 2 9.5\n2 4 2 5 1 3 1 3 1 2 1 2 4\n")
    original = import_fjs(path, instance_id="test")
    assert original.factory == FactorySpec((Machine("M1"), Machine("M2")))
    expected = WorkloadInstance(
        (
            Order(
                "test",
                (
                    Job(
                        "test/job_1",
                        (
                            Operation(
                                "test/job_1/op_1",
                                (
                                    ProcessingMode("m1_t2_v1", "M1", 2),
                                    ProcessingMode("m1_t3_v1", "M1", 3),
                                    ProcessingMode("m1_t3_v2", "M1", 3),
                                    ProcessingMode("m2_t5_v1", "M2", 5),
                                ),
                            ),
                            Operation(
                                "test/job_1/op_2",
                                (ProcessingMode("m2_t4_v1", "M2", 4),),
                                ("test/job_1/op_1",),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    assert original.workload == expected
    assert (
        original.provenance.source_sha256
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert original.provenance.header.average_flexibility == "9.5"
    with pytest.raises(FrozenInstanceError):
        original.provenance.header.jobs = 7
    path.write_text("1 2 9.5\n\n2 4 1 3 1 2 2 5 1 3 1 2 4\n")
    reordered = import_fjs(path, instance_id="test")
    assert reordered.workload == expected
    assert reordered.provenance.source_sha256 != original.provenance.source_sha256
    assert digest(reordered.workload) == digest(original.workload)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "1",
        "1 2 3 4\n1 1 1 2",
        "0 2\n",
        "1.0 2\n1 1 1 2",
        "1 2 nan\n1 1 1 2",
        "1 2 inf\n1 1 1 2",
        "1 2 bad\n1 1 1 2",
        "1 2 -1\n1 1 1 2",
        "1 0\n1 1 1 2",
        "2 2\n1 1 1 2",
        "1 2\n1 1 1 2\n1 1 1 2",
        "1 2\n0",
        "1 2\n1 0",
        "1 2\n1 1 1",
        "1 2\n1 1 1 2 3",
        "1 2\n1 1 3 2",
        "1 2\n1 1 0 2",
        "1 2\n1 1 1 0",
        "1 2\n1 1 1 -2",
        "1 2\n1 1 1 2.0",
        "1 2\n1 true 1 2",
        "1 2\n1 1 1 True",
        "1 2\n1 1 1 2 # comment",
        "1 2\n2 1 1 2",
    ],
)
def test_invalid_fjs_fails_before_export_allocation(tmp_path, content):
    source = tmp_path / "input.fjs"
    source.write_text(content)
    with pytest.raises(ValueError):
        import_fjs(source, instance_id="x")
    output = tmp_path / "new-parent" / "output"
    assert (
        main(
            [
                "import-fjs",
                str(source),
                "--instance-id",
                "x",
                "--output-dir",
                str(output),
                "--factory",
                str(ROOT / "configs/factories/factory_hand.yaml"),
            ]
        )
        == 2
    )
    assert not output.parent.exists()


def test_import_cli_exports_reusable_standard_files_and_refuses_overwrite(bundle):
    from smartsom.api import load_config, prepare
    from smartsom.config.factory_design import load_factory_design
    from smartsom.config.production import WorkloadFile, read_file

    source = bundle / "data/reference/mk01/Mk01.fjs"
    factory = bundle / "configs/factories/mk01.yaml"
    output = bundle / "imported"
    command = [
        "import-fjs",
        str(source),
        "--instance-id",
        "mk01",
        "--output-dir",
        str(output),
        "--factory",
        str(factory),
    ]
    assert main(command) == 0
    contents = {p.name: p.read_bytes() for p in output.iterdir()}
    assert set(contents) == {
        "factory.yaml",
        "workload.yaml",
        "scenario.yaml",
        "algorithm.yaml",
        "run.yaml",
        "source.fjs",
    }
    assert contents["source.fjs"] == source.read_bytes()
    assert main(command) == 1
    assert {p.name: p.read_bytes() for p in output.iterdir()} == contents
    parsed = import_fjs(source, instance_id="mk01")
    assert (
        load_factory_design(output / "factory.yaml")[0]
        == load_factory_design(factory)[0]
    )
    workload = read_file(output / "workload.yaml", WorkloadFile)
    assert workload.provenance == parsed.provenance
    for demand in workload.demands:
        job = next(
            j
            for order in parsed.workload.orders
            for j in order.jobs
            if j.job_id == demand.demand_id
        )
        for step, op in zip(demand.steps, job.operations, strict=True):
            assert dict(step.machine_nominal_ticks) == {
                m.machine_id: m.nominal_ticks for m in op.modes
            }
    config = load_config(output / "run.yaml")
    config.training.max_ticks = 16
    first = prepare(config, training=False)
    assert main(["validate", str(output / "run.yaml")]) == 0
    assert not (bundle / "runs").exists()
    run = run_one(first, verbose=False, output_root=bundle / "runs")
    manifest = json_file(run.run_dir, "run.json")
    assert manifest["inputs"]["scenario"]["demands"] == primitive(workload.demands)
    assert (
        manifest["input_metadata"]["workload_authoring"]["provenance"]["source_sha256"]
        == parsed.provenance.source_sha256
    )
    assert (
        json.loads(first.resolved.workload_json)["provenance"]["source_sha256"]
        == parsed.provenance.source_sha256
    )
    config.seed = 234
    config.algorithm.source = str(bundle / "configs/algorithms/first_feasible.yaml")
    second = prepare(config, training=False)
    assert second.resolved.scenario.demands == first.resolved.scenario.demands
    assert (
        json.loads(second.resolved.workload_json)["provenance"]
        == json.loads(first.resolved.workload_json)["provenance"]
    )


@pytest.mark.parametrize("instance_id", ["", "  ", True, None])
def test_import_requires_explicit_valid_identity(tmp_path, instance_id):
    with pytest.raises(ValueError, match="instance_id"):
        import_fjs(tmp_path / "absent.fjs", instance_id=instance_id)


def test_fjsp_generator_seed_golden_and_frozen_base():
    recipe = resolve_run(ROOT / "configs/runs/generated_fjsp_spt.yaml").resolved
    assert (
        digest(recipe.scenario.demands)
        == "a3a90fdbca591fba5b47d4e6c53ccb159824f010906e2cf041895722f160c7bc"
    )
    assert recipe.scenario.demands == recipe.episode(123).demands
    assert sum(len(d.steps) for d in recipe.scenario.demands) == 7
    for demand in recipe.scenario.demands:
        for step in demand.steps:
            eligible = {
                m.machine_id
                for m in recipe.scenario.factory.machines
                if step.operation_type in m.operation_types
            }
            assert set(dict(step.machine_nominal_ticks)) == eligible
    assert any(
        len(s.machine_nominal_ticks) > 1
        for d in recipe.scenario.demands
        for s in d.steps
    )


def test_generation_ranges_revisits_ids_and_local_rng():
    factory = FactorySpec((Machine("M2"), Machine("M1")))
    state = random.getstate()
    for count in (1, 2):
        profile = StaticFJSPProfile(
            2, 2, IntegerRange(3, 5), IntegerRange(count, count), IntegerRange(1, 7)
        )
        for seed in range(10):
            workload = generate_fjsp(factory, profile, seed)
            assert workload == generate_fjsp(
                replace(factory, machines=tuple(reversed(factory.machines))),
                profile,
                seed,
            )
            assert random.getstate() == state
            for j, order in enumerate(workload.orders, 1):
                assert order.order_id == f"order_{j}"
                for k, job in enumerate(order.jobs, 1):
                    assert job.job_id == f"order_{j}/job_{k}"
                    assert (
                        3 <= len(job.operations) <= 5
                    )  # More operations than machines.
                    for i, op in enumerate(job.operations, 1):
                        assert op.operation_id == f"{job.job_id}/op_{i}"
                        assert (
                            len(op.modes)
                            == len({m.machine_id for m in op.modes})
                            == count
                        )
                        assert all(
                            1 <= m.nominal_ticks <= 7
                            and m.processing_mode_id == f"machine/{m.machine_id}"
                            for m in op.modes
                        )
                        assert op.predecessor_ids == (
                            (job.operations[i - 2].operation_id,) if i > 1 else ()
                        )
            assert normalize_workload(workload) == workload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(generator="static_jsp_v1"),
        lambda d: d["profile"].pop("operation_types"),
        lambda d: d["profile"].update(seed=1),
        lambda d: d["profile"].update(jobs=True),
        lambda d: d["profile"].update(jobs=1.0),
        lambda d: d["profile"].update(operation_types=["missing"]),
        lambda d: d["profile"].update(min_operations=4, max_operations=3),
        lambda d: d["profile"].update(min_operations=0),
        lambda d: d["profile"].update(min_operations=True),
        lambda d: d["profile"].update(max_operations=2.0),
        lambda d: d["profile"].update(nominal_min=0),
        lambda d: d["profile"].update(nominal_min=3, nominal_max=2),
    ],
)
def test_strict_profile_validation_before_simulator_or_directory(
    bundle, mutate, monkeypatch
):
    edit(bundle / "configs/workloads/static_fjsp.yaml", mutate)

    def forbidden(*args, **kwargs):
        pytest.fail("Simulator created during invalid materialization")

    monkeypatch.setattr("smartsom.engine.production.ProductionSimulator", forbidden)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "generated_fjsp_spt"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("name,ticks", [("fast", 32), ("slow", 33)])
def test_hand_configs_match_code_and_selected_modes(bundle, name, ticks):
    from smartsom.engine.production import ProductionSimulator
    from smartsom.trace.production import audit

    prepared = resolve_run(run_path(bundle, f"fjsp_{name}"))
    sim = ProductionSimulator(prepared.resolved.scenario)
    for command in prepared.resolved.algorithm.commands:
        sim.step(command)
    result = run_one(prepared, verbose=False)
    assert result.simulation_result.final_state == sim.snapshot()
    assert result.simulation_result.makespan == ticks
    assert audit(result.run_dir)["status"] == "passed"


def test_generated_export_is_fixed_when_seed_or_provider_changes(bundle):
    path = run_path(bundle, "generated_fjsp_spt")
    first = resolve_run(path).resolved
    output = bundle / "configs/workloads/frozen.yaml"
    output.write_text(first.workload_json)
    edit(
        bundle / "configs/scenarios/generated_fjsp.yaml",
        lambda d: d.update(workload="../workloads/frozen.yaml"),
    )
    edit(
        path, lambda d: d.update(seed=5, algorithm="../algorithms/first_feasible.yaml")
    )
    second = resolve_run(path).resolved
    assert second.scenario.demands == first.scenario.demands
    assert second.workload_json == first.workload_json


def test_generation_and_multimode_replay_do_not_depend_on_cwd_or_hash_seed(tmp_path):
    script = """
import sys
from pathlib import Path
from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.experiments import run_one
root=Path(sys.argv[1])
r=resolve_run(root/'configs/runs/generated_fjsp_spt.yaml')
print(canonical_json(r.resolved.scenario.demands))
r=resolve_run(root/'configs/runs/fjsp_fast.yaml')
print(canonical_json(run_one(r, output_root=sys.argv[2], verbose=False).simulation_result))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script, str(ROOT), str(tmp_path / "runs")],
            cwd=directory,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed, directory in [("1", tmp_path), ("799", ROOT)]
    ]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.0, "1"])
def test_generator_rejects_invalid_effective_seed(seed):
    factory = FactorySpec((Machine("M1"),))
    profile = StaticFJSPProfile(
        1, 1, IntegerRange(2, 2), IntegerRange(1, 1), IntegerRange(3, 3)
    )
    with pytest.raises(ValueError, match="seed"):
        generate_fjsp(factory, profile, seed)


def test_fjsp_import_handles_non_utf8_missing_input_and_empty_destination(tmp_path):
    source = tmp_path / "input.fjs"
    output = tmp_path / "output"
    args = [
        "import-fjs",
        str(source),
        "--instance-id",
        "x",
        "--output-dir",
        str(output),
        "--factory",
        str(ROOT / "configs/factories/factory_hand.yaml"),
    ]
    assert main(args) == 2
    source.write_bytes(b"\xff")
    assert main(args) == 2
    assert not output.exists()
    source.write_text("1 1\n1 1 1 2\n")
    output.mkdir()
    assert main(args) == 1
    assert list(output.iterdir()) == []
