"""Versioned, reusable JSONL timing tables with strict per-row validation."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from smartsom.config.codec import ConfigurationError, _unique_pairs, canonical_json
from smartsom.config.models import ArrivalProvenance, Reference, StrictModel
from smartsom.domain import ArrivalPlan, JobArrival


class ArrivalRow(StrictModel):
    schema_id: Literal["smartsom.job-arrival/v1"] = Field(alias="schema")
    job_id: Reference
    release_at: int
    reveal_at: int
    provenance: ArrivalProvenance | None = None

    @model_validator(mode="after")
    def timing(self):
        JobArrival(self.job_id, self.release_at, self.reveal_at)
        return self


def arrival_rows(plan: ArrivalPlan, provenance: ArrivalProvenance | None):
    return tuple(
        ArrivalRow(
            schema="smartsom.job-arrival/v1",
            job_id=job.job_id,
            release_at=job.release_at,
            reveal_at=job.reveal_at,
            provenance=provenance,
        )
        for job in plan.jobs
    )


def read_arrivals(path: Path) -> tuple[ArrivalPlan, ArrivalProvenance | None, str]:
    try:
        raw = path.read_bytes()
        rows = []
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line, object_pairs_hook=_unique_pairs)
                rows.append(ArrivalRow.model_validate_json(canonical_json(data)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"line {number}: {exc}") from exc
        if not rows:
            raise ValueError("arrival table must not be empty")
        provenance = rows[0].provenance
        if any(row.provenance != provenance for row in rows):
            raise ValueError("arrival provenance must agree across all rows")
        plan = ArrivalPlan(
            tuple(JobArrival(row.job_id, row.release_at, row.reveal_at) for row in rows)
        )
        return plan, provenance, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
