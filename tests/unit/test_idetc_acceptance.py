import importlib
import json
import shutil
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
import yaml

from smartsom.config import resolve_run, resolve_study
from smartsom.config.codec import digest, read_model
from smartsom.config.models import FactoryFile, InstanceFile
from smartsom.experiments import run_one
from smartsom.experiments.evidence import write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("validation.idetc_inputs")


def test_lock_versions_and_verified_raw_inputs(inputs):
    source = inputs.verify_source()
    assert source["input_commit"] == "729bc692c85781be275f42144ce582449b875d98"
    assert source["lock_archive_commit"] == "b668ba6e677e3410c5074a7c281d1c6004e0040a"
    assert source["freeze_record_dirty"] is True
    assert len(source["files"]) == 8
    assert len(source["paper_reference"]["results"]) == 12
    assert (
        source["paper_reference"]["pdf_sha256"]
        == "72023b1162f1b6fc43731191c78cdb45aa4b535e5b11d31b4026da79660bb66f"
    )


@pytest.mark.parametrize(
    "case,operations,max_release",
    [("S00", 283, 149), ("S01", 291, 284), ("S10", 283, 149), ("S11", 291, 284)],
)
def test_conversion_against_raw_fields_and_independent_expectations(
    inputs, case, operations, max_release
):
    factory, workload, arrivals = inputs.convert_case(case)
    raw = yaml.safe_load(
        (inputs.REFERENCE / "raw" / case / "instance.yaml").read_text()
    )["instance_config"]
    jobs = json.loads((inputs.REFERENCE / "raw" / case / "jobs.json").read_text())[
        "jobs"
    ]
    assert len(jobs) == len(workload.orders[0].jobs) == 100
    assert len(workload.operations) == operations
    assert max(a.release_at for a in arrivals.jobs) == max_release
    assert [(a.job_id, a.reveal_at, a.release_at) for a in arrivals.jobs] == [
        (j["job_id"], j["release_time"], j["release_time"]) for j in jobs
    ]
    assert len(factory.machines) == 8
    assert [(a.agv_id, a.initial_node_id) for a in factory.transport.agvs] == [
        (f"t-{i}", f"m-{i}") for i in range(4)
    ]
    assert len(factory.buffers) == 8
    assert all(b.pre_capacity == b.post_capacity == 6 for b in factory.buffers)
    assert (
        factory.holding_buffer.buffer_id,
        factory.holding_buffer.node_id,
        factory.holding_buffer.capacity,
    ) == ("b-hold", "b-hold", 999999)
    assert (
        len(factory.transport.nodes) == 11
        and len(factory.transport.travel_times) == 121
    )
    matrix = raw["logistics"]["specification"].strip().splitlines()
    labels = matrix[0].split("|")
    original = {
        (line.split("|")[0], dest): int(value)
        for line in matrix[1:]
        if line.strip()
        for dest, value in zip(labels, line.split("|")[1].split(), strict=True)
    }
    assert {
        (t.from_node_id, t.to_node_id): t.ticks for t in factory.transport.travel_times
    } == original
    assert [
        (q.quality_mode_id, q.time_scale, q.error_rate)
        for q in factory.quality_speed.default_modes
    ] == [
        ("M0", Decimal("1.2"), Decimal("0.01")),
        ("M1", Decimal("1.0"), Decimal("0.018")),
        ("M2", Decimal("0.8"), Decimal("0.03")),
    ]
    for converted, old in zip(workload.orders[0].jobs, jobs, strict=True):
        assert converted.job_id == old["job_id"]
        for index, (op, previous) in enumerate(
            zip(converted.operations, old["operations"], strict=True), 1
        ):
            assert op.operation_id == f"{old['job_id']}/op_{index:03d}"
            assert op.predecessor_ids == (
                () if index == 1 else (f"{old['job_id']}/op_{index - 1:03d}",)
            )
            eligible = {
                m
                for m, types in raw["routing"]["machine_capabilities"].items()
                if previous["operation_type"] in types
            }
            assert {
                (m.processing_mode_id, m.machine_id, m.nominal_ticks) for m in op.modes
            } == {(m, m, previous["proc_time"]) for m in eligible}
            for scale in ("1.2", "1.0", "0.8"):
                # Independent original Python-round vs contracted decimal half-up.
                assert round(previous["proc_time"] * float(scale)) == int(
                    (Decimal(previous["proc_time"]) * Decimal(scale)).to_integral_value(
                        rounding=ROUND_HALF_UP
                    )
                )
    f = read_model(inputs.REFERENCE / "converted" / case / "factory.yaml", FactoryFile)[
        0
    ].factory
    w = read_model(
        inputs.REFERENCE / "converted" / case / "workload.json", InstanceFile
    )[0].workload
    assert (f, w) == (factory, workload)


def test_export_is_reproducible_source_checked_and_no_overwrite(inputs, tmp_path):
    output = tmp_path / "export"
    inputs.write_bundle(output)
    for path in (inputs.REFERENCE / "converted").rglob("*"):
        if path.is_file():
            assert (
                output / path.relative_to(inputs.REFERENCE / "converted")
            ).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        inputs.write_bundle(output)
    source = tmp_path / "source"
    shutil.copytree(inputs.REFERENCE, source)
    (source / "raw/S00/jobs.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        inputs.write_bundle(tmp_path / "invalid", source)
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="unknown"):
        inputs.convert_case("unknown")


def test_all60_pairing_inputs_and_frozen_identity(inputs, tmp_path):
    study = resolve_study(ROOT / "configs/studies/idetc_spt.yaml")
    assert study.spec.seed == 101 and study.spec.replications == 5
    assert len(study.entries) == 60
    assert (
        study.plan_sha256
        == "e1950c1ae8d93c926961568d62b71be998b1b1285ad5f62779d6836a03171f9b"
    )
    pairs = defaultdict(list)
    for entry in study.entries:
        recipe = entry.resolved.resolved
        scenario = recipe.scenario
        assert entry.variant_id == "control"
        assert recipe.algorithm.provider == "builtin.spt"
        assert recipe.algorithm.quality_mode == entry.algorithm_id.removeprefix("SPT-")
        assert (
            scenario.mode == "dynamic"
            and scenario.quality_probability_visibility == "public"
        )
        assert not scenario.outages and not scenario.processing_samples
        assert len(scenario.factory.agvs) == 4
        raw = yaml.safe_load(
            (inputs.REFERENCE / "raw" / entry.case_id / "instance.yaml").read_text()
        )["instance_config"]
        assert {
            m.machine_id: list(m.operation_types) for m in scenario.factory.machines
        } == raw["routing"]["machine_capabilities"]
        jobs = json.loads(
            (inputs.REFERENCE / "raw" / entry.case_id / "jobs.json").read_text()
        )["jobs"]
        assert len(scenario.demands) == len(jobs) == 100
        for demand, original in zip(scenario.demands, jobs, strict=True):
            assert demand.demand_id == original["job_id"]
            assert demand.release_at == demand.reveal_at == original["release_time"]
            assert [
                (step.operation_type, step.nominal_ticks) for step in demand.steps
            ] == [
                (op["operation_type"], op["proc_time"]) for op in original["operations"]
            ]
        pairs[entry.case_id, entry.replication].append(recipe)
    assert len(pairs) == 20
    for group in pairs.values():
        assert len(group) == 3 and len({r.scenario_json for r in group}) == 1
        assert len({r.scenario.seed for r in group}) == 1
    # Portable current configuration keeps identity; raw historical export still
    # reproduces its frozen matrix bytes in the separate conversion test above.
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    assert (
        resolve_study(tmp_path / "configs/studies/idetc_spt.yaml").plan_sha256
        == study.plan_sha256
    )
    assert not (tmp_path / "artifacts").exists()


@pytest.fixture
def audit_example(inputs, tmp_path):
    result = run_one(
        resolve_run(ROOT / "configs/runs/holding_hand.yaml"),
        output_root=tmp_path,
        verbose=False,
    )
    return importlib.import_module("validation.idetc_audit"), result


def test_audit_small_hand_case(audit_example):
    audit, result = audit_example
    report = audit.audit_run(result.run_dir, expected_jobs=1, expected_operations=2)
    assert report["status"] == "passed" and report["makespan"] == 22
    assert report["holding_trips"] == 1
    assert "semantic_commands" in report["checks"]
    assert report["paper_comparison"] == "incompatible_physics"


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("state", "execution audit"),
        ("trace", "execution audit"),
        ("action", "execution audit"),
        ("summary", "run result differs"),
        ("checksum", "state hash"),
    ],
)
def test_audit_rejects_corrupt_evidence(audit_example, corruption, match):
    from smartsom.trace.production import state_hash

    audit, result = audit_example
    directory = result.run_dir
    if corruption == "summary":
        path = directory / "run.json"
        manifest = json.loads(path.read_text())
        manifest["result"]["return"] += 1
        write_json(path, manifest)
    else:
        path = directory / "trace.jsonl"
        rows = audit.read_jsonl(path)
        if corruption in ("state", "checksum"):
            rows[0]["state"]["return"] += 1
        elif corruption == "trace":
            rows[0]["events"][0]["tick"] += 1
        elif corruption == "action":
            rows[0]["actions"]["agvs"] = [[rows[0]["actions"]["agvs"][0][0], "WAIT"]]
        if corruption != "checksum":
            # Matching hashes must never bypass the independent transition check.
            rows[0]["state_hash"] = state_hash(rows[0]["state"])
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=match):
        audit.audit_run(directory, expected_jobs=1, expected_operations=2)


def test_reporting_failure_denominator_and_paired_sets(inputs):
    module = importlib.import_module("validate_idetc")
    source = inputs.verify_source()
    rows, audits = [], []
    for case in inputs.CASES:
        for replication in range(5):
            for mode in range(3):
                key = f"{case}-{replication}-{mode}"
                rows.append(
                    dict(
                        entry_id=key,
                        case_id=case,
                        algorithm_id=f"SPT-M{mode}",
                        replication=replication,
                        variant_id="control",
                        status="completed",
                        makespan=10,
                        passing_rate=0.9,
                    )
                )
                audits.append(
                    dict(
                        entry_id=key,
                        status="passed",
                        passed_job_ids=list(range(3 - mode)),
                    )
                )
    assert module.summarize(rows, audits, source["paper_reference"]["results"])[
        "accepted"
    ]
    audits[-1]["passed_job_ids"] = [99]
    assert not module.summarize(rows, audits, source["paper_reference"]["results"])[
        "accepted"
    ]
    rows[0].update(status="failed", makespan=None, passing_rate=None)
    result = module.summarize(rows, audits, source["paper_reference"]["results"])
    group = result["groups"][0]
    assert group["completed"] == 4 and group["failed_or_incomplete"] == 1
    assert group["makespan_mean"] == 10 and group["makespan_sample_sd"] == 0
    assert result["completed"] == 59 and not result["accepted"]


def test_frozen_inputs_do_not_require_external_repository(inputs):
    source = inputs.verify_source()
    assert (
        digest(source)
        == json.loads((inputs.REFERENCE / "converted/conversion.json").read_text())[
            "source_sha256"
        ]
    )


def test_failed_attempts_and_invalid_evidence_still_produce_complete_report(
    inputs, tmp_path, monkeypatch
):
    module = importlib.import_module("validate_idetc")
    study = resolve_study(ROOT / "configs/studies/idetc_spt.yaml")
    monkeypatch.setattr(module, "_load_plan", lambda _: list(study.entries))

    def failed(directory, entry):
        if entry.replication == 0:
            raise ValueError("damaged evidence")
        return {"status": "failed", "run_dir": None, "failure_reason": "deadlock"}

    monkeypatch.setattr(module, "_latest", failed)
    result = module.audit_study(tmp_path, workers=1, development=True)
    assert not result["accepted"] and result["completed"] == result["audited"] == 0
    assert len(result["rows"]) == 60 and len(result["groups"]) == 12
    report = next(tmp_path.glob("acceptance/*/report.json"))
    saved = json.loads(report.read_text())
    assert saved["evidence_kind"] == "development"
    assert {r["status"] for r in saved["rows"]} == {"failed", "evidence_invalid"}
    assert "damaged evidence" in report.with_suffix(".md").read_text()
    assert len((report.parent / "runs.csv").read_text().splitlines()) == 61
    assert all(g["makespan_mean"] is None for g in result["groups"])


def test_formal_gate_rejects_unintegrated_scientific_changes(inputs, monkeypatch):
    module = importlib.import_module("validate_idetc")

    def command(args, **kwargs):
        if "--show-current" in args:
            return "main\n"
        if "status" in args:
            return b" M AGENTS.md\0?? docs/papers/draft.md\0 M configs/studies/idetc_spt.yaml\0"
        return "commit\n"

    monkeypatch.setattr(module.subprocess, "check_output", command)
    with pytest.raises(ValueError, match="commit acceptance inputs/code first"):
        module.require_integrated_source()
