"""Portable Studio drawing annotations, never simulator initial conditions."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from smartsom.config.models import StrictModel


class DrawingProgress(StrictModel):
    total: int = Field(default=0, ge=0, le=100000)
    remaining: int = Field(default=0, ge=0, le=100000)

    @model_validator(mode="after")
    def valid_progress(self):
        if self.remaining > self.total:
            raise ValueError("Remaining ticks cannot exceed total ticks")
        return self


MACHINE_STATES = {
    "IDLE": "Idle",
    "READY": "Ready",
    "PROCESSING": "Processing",
    "BLOCKED": "Blocked",
    "DOWN": "Breakdown",
}


class DrawingMachine(DrawingProgress):
    status: Literal["IDLE", "READY", "PROCESSING", "BLOCKED", "DOWN"] = "IDLE"
    mode: str = "normal"


class DrawingJob(StrictModel):
    order: int = Field(ge=1)
    attempt: int = Field(default=1, ge=1)
    quality: Literal["UNKNOWN", "PASS", "FAIL"] = "UNKNOWN"
    owner: str
    slot: str = ""

    @property
    def identifier(self):
        return f"d{self.order:06d}_a{self.attempt - 1}"


class DrawingAGV(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    conflict: bool = False
    charging: bool = False
    conflict_with: str | None = None

    @property
    def status(self):
        return (
            "CONFLICT" if self.conflict else "CHARGING" if self.charging else "NORMAL"
        )


class DrawingBuffer(DrawingProgress):
    waiting: int = Field(default=0, ge=0)
    delivered: int = Field(default=0, ge=0)
    passing_percent: float = Field(default=100.0, ge=0, le=100)
    throughput: float = Field(default=0.0, ge=0)


class DrawingStation(DrawingProgress):
    inspecting: bool = False


class DrawingState(StrictModel):
    tick: int = Field(default=0, ge=0)
    jobs: tuple[DrawingJob, ...] = ()
    machines: dict[str, DrawingMachine] = Field(default_factory=dict)
    agvs: dict[str, DrawingAGV] = Field(default_factory=dict)
    buffers: dict[str, DrawingBuffer] = Field(default_factory=dict)
    stations: dict[str, DrawingStation] = Field(default_factory=dict)
    disposed: dict[str, int] = Field(default_factory=dict)

    @field_validator("jobs", mode="before")
    @classmethod
    def jobs_sequence(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_jobs(self):
        ids = [job.identifier for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("Each order / attempt must be unique")
        if any(type(n) is not int or n < 0 for n in self.disposed.values()):
            raise ValueError("Disposed counts must be nonnegative integers")
        return self
