"""One frozen IDETC ingress; produce existing domain/config types, not a runtime."""

import json
from pathlib import Path

import yaml

from smartsom.config.arrivals import arrival_rows
from smartsom.config.codec import _UniqueLoader, canonical_json, digest, primitive
from smartsom.config.models import FactoryFile, InstanceFile
from smartsom.domain import (
    AGV,
    ArrivalPlan,
    FactorySpec,
    HoldingBuffer,
    Job,
    JobArrival,
    Machine,
    MachineBuffers,
    MachineLocation,
    Operation,
    Order,
    ProcessingMode,
    TransportSpec,
    TravelTime,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.quality import QualityMode, QualitySpeedSpec
from smartsom.experiments.evidence import _file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/idetc"
CASES = ("S00", "S01", "S10", "S11")
CONVERTER = "smartsom.idetc-frozen/v1"


def verify_source(reference: Path = REFERENCE) -> dict:
    source = json.loads((reference / "source.json").read_text())
    if _file_digest(reference / "raw/bundle.lock.json") != source["lock_sha256"]:
        raise ValueError("frozen lock digest mismatch")
    lock = json.loads((reference / "raw/bundle.lock.json").read_text())
    if (
        lock["git"]["commit"] != source["input_commit"]
        or lock["lifecycle_state"] != "frozen"
    ):
        raise ValueError("unexpected lock identity")
    for row in source["files"]:
        if (
            _file_digest(reference / row["local_path"]) != row["sha256"]
            or row["sha256"] != lock["file_hashes"][row["source_path"]]
        ):
            raise ValueError(f"frozen input digest mismatch: {row['local_path']}")
    return source


def convert_case(case_id: str, reference: Path = REFERENCE):
    if case_id not in CASES:
        raise ValueError("unknown frozen IDETC case")
    raw = yaml.load(
        (reference / "raw" / case_id / "instance.yaml").read_text(),
        Loader=_UniqueLoader,
    )["instance_config"]
    payload = json.loads((reference / "raw" / case_id / "jobs.json").read_text())
    capabilities = raw["routing"]["machine_capabilities"]
    machines = tuple(Machine(key) for key in sorted(capabilities))
    lines = [
        line.strip()
        for line in raw["logistics"]["specification"].splitlines()
        if line.strip()
    ]
    nodes = tuple(lines[0].split("|"))
    times = []
    for line in lines[1:]:
        node, durations = line.split("|")
        durations = [int(value) for value in durations.split()]
        if len(durations) != len(nodes):
            raise ValueError("invalid frozen matrix row")
        times.extend(
            TravelTime(node, target, value)
            for target, value in zip(nodes, durations, strict=True)
        )
    buffers = {entry["role"]: entry for entry in raw["buffer"]}
    transport = TransportSpec(
        nodes,
        tuple(MachineLocation(m.machine_id, m.machine_id) for m in machines),
        buffers["input"]["name"],
        buffers["output"]["name"],
        tuple(
            AGV(f"t-{i}", machines[i % len(machines)].machine_id)
            for i in range(raw["logistics"]["amount"])
        ),
        tuple(times),
    )
    labels = {"quality": "M0", "normal": "M1", "fast": "M2"}
    quality = QualitySpeedSpec(
        tuple(
            QualityMode(labels[m["mode"]], str(m["time_scale"]), str(m["error_rate"]))
            for m in raw["quality"]["modes"]
        )
    )
    holding = buffers["compensation"]
    f = FactorySpec(
        machines,
        transport,
        tuple(
            MachineBuffers(
                m.machine_id,
                raw["machines"]["prebuffer"][0]["capacity"],
                raw["machines"]["postbuffer"][0]["capacity"],
            )
            for m in machines
        ),
        quality,
        HoldingBuffer(holding["name"], holding["name"], holding["capacity"]),
    )
    jobs, arrivals = [], []
    for job in payload["jobs"]:
        operations = []
        for index, operation in enumerate(job["operations"], 1):
            key = f"{job['job_id']}/op_{index:03d}"
            modes = tuple(
                ProcessingMode(machine, machine, operation["proc_time"])
                for machine, types in sorted(capabilities.items())
                if operation["operation_type"] in types
            )
            operations.append(
                Operation(
                    key, modes, (operations[-1].operation_id,) if operations else ()
                )
            )
        jobs.append(Job(job["job_id"], tuple(operations)))
        arrivals.append(
            JobArrival(job["job_id"], job["release_time"], job["release_time"])
        )
    w = WorkloadInstance((Order("idetc", tuple(jobs)),))
    a = ArrivalPlan(tuple(arrivals))
    validate_problem(f, w)
    a.validate(w)
    return f, w, a


def write_bundle(output: Path, reference: Path = REFERENCE) -> dict:
    source = verify_source(reference)
    # Validate all inputs before creating the destination.
    cases = {case: convert_case(case, reference) for case in CASES}
    output.mkdir(parents=True)  # No overwrite, even for an existing empty directory.
    converted = {}
    for case, (factory, workload, arrivals) in cases.items():
        folder = output / case
        folder.mkdir()
        (folder / "factory.yaml").write_text(
            yaml.safe_dump(
                primitive(FactoryFile(schema="smartsom.factory/v1", factory=factory)),
                sort_keys=True,
            )
        )
        write_json(
            folder / "workload.json",
            InstanceFile(
                schema="smartsom.workload-instance/v1",
                workload=workload,
                content_sha256=digest(workload),
            ),
        )
        (folder / "arrivals.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in arrival_rows(arrivals, None))
        )
        scenario = {
            "schema": "smartsom.scenario/v1",
            "factory": "factory.yaml",
            "workload": {"kind": "instance", "path": "workload.json"},
            "arrivals": {"kind": "fixed", "path": "arrivals.jsonl"},
            "transport": {"kind": "fixed_matrix"},
            "buffers": {"kind": "limited"},
            "holding_buffer": {"kind": "shared"},
            "quality": {
                "kind": "independent_operation_v1",
                "probability_visibility": "public",
            },
            "decision_trigger": "dispatch_available",
        }
        (folder / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=True))
        converted[case] = {
            "factory_sha256": digest(factory),
            "workload_sha256": digest(workload),
            "arrivals_sha256": digest(arrivals),
            "jobs": len(arrivals.jobs),
            "operations": len(workload.operations),
            "max_release": max(j.release_at for j in arrivals.jobs),
            "files": {
                name: _file_digest(folder / name)
                for name in (
                    "factory.yaml",
                    "workload.json",
                    "arrivals.jsonl",
                    "scenario.yaml",
                )
            },
        }
    for mode in ("M0", "M1", "M2"):
        (output / f"spt_{mode.lower()}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "smartsom.algorithm/v1",
                    "algorithm": {
                        "provider": "builtin.spt",
                        "parameters": {"quality_mode": mode},
                    },
                }
            )
        )
    study = {
        "schema": "smartsom.study/v1",
        "seed": 101,
        "replications": 5,
        "cases": [{"id": case, "scenario": f"{case}/scenario.yaml"} for case in CASES],
        "algorithms": [
            {"id": f"SPT-{mode}", "config": f"spt_{mode.lower()}.yaml"}
            for mode in ("M0", "M1", "M2")
        ],
        "output_root": "runs",
        "recording": {"observations": "hash", "debug": False},
    }
    (output / "study.yaml").write_text(yaml.safe_dump(study, sort_keys=True))
    manifest = {
        "schema": "smartsom.idetc-conversion/v1",
        "converter": CONVERTER,
        "source_sha256": digest(source),
        "source_input_commit": source["input_commit"],
        "cases": converted,
        "quality_labels": {"quality": "M0", "normal": "M1", "fast": "M2"},
        "operation_identity": "<original job ID>/op_<one-based route index, three digits>",
        "base_mode_identity": "eligible machine ID",
        "arrival_rule": "reveal equals original release",
        "non_execution_metadata": [
            "legacy quality threshold 0.95",
            "old proposal volatility metrics; edits already in routes",
        ],
        "study_sha256": _file_digest(output / "study.yaml"),
    }
    write_json(output / "conversion.json", manifest)
    return manifest
