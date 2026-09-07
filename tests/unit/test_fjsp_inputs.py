import hashlib
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
from test_experiments import ROOT, edit, json_file, run_path
from test_experiments import bundle as bundle
from test_fjsp_engine import FAST, SLOW, flexible_case

from smartsom.algorithms import SPTPolicy
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, normalize_workload, read_model
from smartsom.config.models import FactoryFile, InstanceFile
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.engine import Simulator, replay_schedule
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
    result = Simulator(original.factory, expected).run(SPTPolicy())
    assert (
        replay_schedule(original.factory, expected, result.schedule).schedule
        == result.schedule
    )


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
            ]
        )
        == 2
    )
    assert not output.parent.exists()


def test_import_cli_exports_reusable_standard_files_and_refuses_overwrite(bundle):
    source = bundle / "data/reference/mk01/Mk01.fjs"
    output = bundle / "imported"
    command = [
        "import-fjs",
        str(source),
        "--instance-id",
        "mk01",
        "--output-dir",
        str(output),
    ]
    assert main(command) == 0
    contents = {p.name: p.read_bytes() for p in output.iterdir()}
    assert set(contents) == {"factory.yaml", "workload.json"}
    assert main(command) == 1
    assert {p.name: p.read_bytes() for p in output.iterdir()} == contents
    imported = import_fjs(source, instance_id="mk01")
    assert (
        read_model(output / "factory.yaml", FactoryFile)[0].factory == imported.factory
    )
    instance = read_model(output / "workload.json", InstanceFile)[0]
    assert (
        instance.workload == imported.workload
        and instance.provenance == imported.provenance
    )
    scenario = bundle / "configs/scenarios/mk01.yaml"
    edit(
        scenario,
        lambda d: d.update(
            factory="../../imported/factory.yaml",
            workload={"kind": "instance", "path": "../../imported/workload.json"},
        ),
    )
    first = resolve_run(run_path(bundle, "mk01_spt"))
    assert main(["validate", str(run_path(bundle, "mk01_spt"))]) == 0
    assert not (bundle / "runs").exists()
    run = run_one(first)
    manifest = json_file(run.run_dir, "manifest.json")
    assert manifest["generation_provenance"] is None
    assert (
        manifest["import_provenance"]["source_sha256"]
        == imported.provenance.source_sha256
    )
    edit(
        scenario,
        lambda d: d.update(
            workload={
                "kind": "instance",
                "path": str(run.run_dir / "realized_instance.json"),
            }
        ),
    )
    edit(
        run_path(bundle, "mk01_spt"),
        lambda d: d.update(seed=234, algorithm="../algorithms/first_feasible.yaml"),
    )
    second = resolve_run(run_path(bundle, "mk01_spt"))
    assert second.workload == first.workload
    assert second.workload_sha256 == first.workload_sha256 == instance.content_sha256
    assert second.provenance == first.provenance
    assert not any(seed.consumed for seed in second.seeds)


@pytest.mark.parametrize("instance_id", ["", "  ", True, None])
def test_import_requires_explicit_valid_identity(tmp_path, instance_id):
    with pytest.raises(ValueError, match="instance_id"):
        import_fjs(tmp_path / "absent.fjs", instance_id=instance_id)


def test_fjsp_generator_seed_golden_and_original_jsp_golden():
    resolved = resolve_run(ROOT / "configs/runs/generated_fjsp_spt.yaml")
    assert (
        resolved.workload_sha256
        == "35da0e9853412307a15e33162d7fd8eb4070ceb62aca9b8a9f21bcce4afc9db2"
    )
    assert [
        [(m.machine_id, m.nominal_ticks) for m in op.modes]
        for op in resolved.workload.operations
    ] == [
        [("M1", 5), ("M2", 1), ("M3", 1)],
        [("M2", 3)],
        [("M1", 3), ("M2", 3)],
        [("M1", 1), ("M2", 2), ("M3", 3)],
        [("M1", 5), ("M3", 5)],
        [("M1", 1), ("M2", 1), ("M3", 1)],
    ]
    assert resolved.provenance.generator == "static_fjsp_v1"
    assert resolved.provenance.effective_seed == 6941565647359864201
    assert (
        resolve_run(ROOT / "configs/runs/generated.yaml").workload_sha256
        == "c039e0307dc2c71cc3b29d6e1ee3ad466dd7055b44422a2fd254d0678d0ee7a5"
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
            Simulator(factory, workload).run(SPTPolicy())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(generator="static_jsp_v1"),
        lambda d: d["profile"].pop("eligible_machines_per_operation"),
        lambda d: d["profile"].update(seed=1),
        lambda d: d["profile"].update(order_count=True),
        lambda d: d["profile"].update(jobs_per_order=1.0),
        lambda d: d["profile"].update(
            eligible_machines_per_operation={"min": 1, "max": 4}
        ),
        lambda d: d["profile"].update(
            eligible_machines_per_operation={"min": 2, "max": 1}
        ),
        lambda d: d["profile"].update(
            eligible_machines_per_operation={"min": 0, "max": 2}
        ),
        lambda d: d["profile"].update(
            eligible_machines_per_operation={"min": True, "max": 2}
        ),
        lambda d: d["profile"].update(
            eligible_machines_per_operation={"min": 1, "max": 2.0}
        ),
        lambda d: d["profile"].update(nominal_ticks={"min": 0, "max": 2}),
        lambda d: d["profile"].update(operations_per_job={"min": 3, "max": 2}),
    ],
)
def test_strict_profile_validation_before_simulator_or_directory(
    bundle, mutate, monkeypatch
):
    edit(bundle / "configs/workloads/static_fjsp.yaml", mutate)

    def forbidden(*args, **kwargs):
        pytest.fail("Simulator created during invalid materialization")

    monkeypatch.setattr("smartsom.experiments.runner.Simulator", forbidden)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "generated_fjsp_spt"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("name,schedule", [("fast", FAST), ("slow", SLOW)])
def test_hand_configs_match_code_and_selected_modes(bundle, name, schedule):
    factory, workload = flexible_case()
    resolved = resolve_run(run_path(bundle, f"fjsp_{name}"))
    assert resolved.workload == normalize_workload(workload)
    result = run_one(resolved).simulation_result
    assert result == replay_schedule(factory, workload, schedule)


def test_generated_export_is_fixed_when_seed_or_provider_changes(bundle):
    path = run_path(bundle, "generated_fjsp_spt")
    first = resolve_run(path)
    result = run_one(first)
    edit(
        bundle / "configs/scenarios/generated_fjsp.yaml",
        lambda d: d.update(
            workload={
                "kind": "instance",
                "path": str(result.run_dir / "realized_instance.json"),
            }
        ),
    )
    edit(
        path, lambda d: d.update(seed=5, algorithm="../algorithms/first_feasible.yaml")
    )
    second = resolve_run(path)
    assert (
        second.workload == first.workload
        and second.workload_sha256 == first.workload_sha256
    )
    assert second.provenance == first.provenance
    assert not any(seed.consumed for seed in second.seeds)


def test_generation_and_multimode_replay_do_not_depend_on_cwd_or_hash_seed(tmp_path):
    script = """
import json,sys
from pathlib import Path
from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.engine import replay_schedule
from smartsom.domain import ScheduledOperation
root=Path(sys.argv[1])
r=resolve_run(root/'configs/runs/generated_fjsp_spt.yaml')
print(canonical_json(r.workload))
r=resolve_run(root/'configs/runs/mk01_spt.yaml')
s=json.loads((root/'data/reference/mk01/schedule.json').read_text())['schedule']
print(canonical_json(replay_schedule(r.factory,r.workload,[ScheduledOperation(**e) for e in s])))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script, str(ROOT)],
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
    ]
    assert main(args) == 2
    source.write_bytes(b"\xff")
    assert main(args) == 2
    assert not output.exists()
    source.write_text("1 1\n1 1 1 2\n")
    output.mkdir()
    assert main(args) == 1
    assert list(output.iterdir()) == []
