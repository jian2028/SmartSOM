"""Strict ingress for traditional, serial FJSP text (one job per nonempty line)."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
    validate_problem,
)


@dataclass(frozen=True, slots=True)
class FJSHeader:
    jobs: int
    machines: int
    average_flexibility: str | None = None

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1 for value in (self.jobs, self.machines)
        ):
            raise ValueError("header counts must be positive integers")
        if self.average_flexibility is not None:
            try:
                value = Decimal(self.average_flexibility)
            except (InvalidOperation, TypeError) as exc:
                raise ValueError("invalid header average flexibility") from exc
            if not value.is_finite() or value <= 0:
                raise ValueError(
                    "header average flexibility must be positive and finite"
                )


@dataclass(frozen=True, slots=True)
class ImportProvenance:
    importer: Literal["fjs_v1"]
    importer_version: Literal["1"]
    source_sha256: str
    instance_id: str
    header: FJSHeader

    def __post_init__(self) -> None:
        if self.importer != "fjs_v1" or self.importer_version != "1":
            raise ValueError("unsupported importer version")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("invalid source_sha256")
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be a non-empty string")
        if not isinstance(self.header, FJSHeader):
            raise ValueError("expected FJSHeader")


@dataclass(frozen=True, slots=True)
class ImportedProblem:
    factory: FactorySpec
    workload: WorkloadInstance
    provenance: ImportProvenance


def _positive(token: str, label: str) -> int:
    if not re.fullmatch(r"[0-9]+", token) or int(token) < 1:
        raise ValueError(f"{label} must be a positive integer: {token!r}")
    return int(token)


def import_fjs(path: str | Path, *, instance_id: str) -> ImportedProblem:
    """Read once, retain every alternative, and validate before any export writes."""
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")
    raw = Path(path).read_bytes()
    lines = [line.split() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not lines or len(lines[0]) not in (2, 3):
        raise ValueError("expected header: jobs machines [average_flexibility]")
    header = FJSHeader(
        _positive(lines[0][0], "jobs"),
        _positive(lines[0][1], "machines"),
        lines[0][2] if len(lines[0]) == 3 else None,
    )
    if len(lines) - 1 != header.jobs:
        raise ValueError("job line count does not match header")
    jobs = []
    for job_number, tokens in enumerate(lines[1:], 1):
        cursor = 0

        def take(label: str) -> int:
            nonlocal cursor
            if cursor == len(tokens):
                raise ValueError(f"job {job_number}: missing {label}")
            result = _positive(tokens[cursor], label)
            cursor += 1
            return result

        job_id = f"{instance_id}/job_{job_number}"
        operations = []
        for number in range(1, take("operation count") + 1):
            pairs = []
            for _ in range(take("alternative count")):
                machine, duration = take("machine number"), take("duration")
                if machine > header.machines:
                    raise ValueError(
                        f"job {job_number}: machine number out of range: {machine}"
                    )
                pairs.append((machine, duration))
            occurrences = Counter()
            modes = []
            for machine, duration in sorted(pairs):
                occurrences[machine, duration] += 1
                modes.append(
                    ProcessingMode(
                        f"m{machine}_t{duration}_v{occurrences[machine, duration]}",
                        f"M{machine}",
                        duration,
                    )
                )
            operations.append(
                Operation(
                    f"{job_id}/op_{number}",
                    tuple(sorted(modes, key=lambda mode: mode.processing_mode_id)),
                    (operations[-1].operation_id,) if operations else (),
                )
            )
        if cursor != len(tokens):
            raise ValueError(f"job {job_number}: extra tokens after final operation")
        jobs.append(
            Job(job_id, tuple(sorted(operations, key=lambda op: op.operation_id)))
        )
    factory = FactorySpec(
        tuple(
            sorted(
                (Machine(f"M{index}") for index in range(1, header.machines + 1)),
                key=lambda machine: machine.machine_id,
            )
        )
    )
    workload = WorkloadInstance(
        (Order(instance_id, tuple(sorted(jobs, key=lambda job: job.job_id))),)
    )
    validate_problem(factory, workload)
    return ImportedProblem(
        factory,
        workload,
        ImportProvenance(
            "fjs_v1",
            "1",
            hashlib.sha256(raw).hexdigest(),
            instance_id,
            header,
        ),
    )
