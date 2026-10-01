"""Strict authoring contract for paired Job pools, arrival slots and measured V."""

import math
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from smartsom.config.codec import read_model
from smartsom.config.models import StrictModel
from smartsom.config.production import WorkloadProfile
from smartsom.domain.production import Demand


class WorkloadTemplate(StrictModel):
    id: str = Field(min_length=1)
    route: tuple[str, ...]
    times: tuple[float, ...]
    count: int = Field(gt=0)
    novel: bool = False

    @model_validator(mode="after")
    def aligned(self):
        if not 1 <= len(self.route) <= 4 or len(self.route) != len(self.times):
            raise ValueError("templates require 1-4 aligned reference operations")
        if any(not kind for kind in self.route) or any(t <= 0 for t in self.times):
            raise ValueError("operation types and reference seconds must be positive")
        if any(a == b for a, b in zip(self.route, self.route[1:])):
            raise ValueError("adjacent repeated operation types are forbidden")
        return self


class Personalization(StrictModel):
    category_weights: tuple[int, int, int, int] = (1, 2, 1, 1)
    adjustment_min: float = Field(default=0.1, gt=0, lt=1)
    adjustment_max: float = Field(default=0.3, gt=0, lt=1)
    insertion_min: float = Field(default=0.1, gt=0)
    insertion_max: float = Field(default=0.3, gt=0)
    decimal_places: int = Field(default=6, ge=0, le=9)

    @model_validator(mode="after")
    def ranges(self):
        if any(n < 0 for n in self.category_weights) or not sum(self.category_weights):
            raise ValueError("category_weights must be nonnegative with positive sum")
        if self.adjustment_min > self.adjustment_max:
            raise ValueError("adjustment magnitude bounds are reversed")
        if self.insertion_min > self.insertion_max:
            raise ValueError("insertion ratio bounds are reversed")
        return self


class ArrivalSlots(StrictModel):
    interval_seconds: float = Field(default=2, gt=0)
    initial_jobs: int = Field(default=10, ge=0)
    window_jobs: int = Field(default=40, gt=0)
    tick_seconds: float = Field(default=1, gt=0)
    notice_seconds: float = Field(default=0, ge=0, le=0)
    slots_seconds: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.initial_jobs > self.window_jobs:
            raise ValueError("initial_jobs must fit in the first arrival window")
        if self.slots_seconds is not None and (
            not self.slots_seconds
            or any(t < 0 for t in self.slots_seconds)
            or tuple(sorted(self.slots_seconds)) != self.slots_seconds
        ):
            raise ValueError("explicit arrival slots must be nonnegative and sorted")
        return self

    @property
    def window_seconds(self):
        return Fraction(str(self.interval_seconds)) * self.window_jobs


class DueFunction(StrictModel):
    base_seconds: float = Field(default=60, ge=0)
    reference_work_factor: float = Field(default=4, gt=0)
    operation_seconds: float = Field(default=20, gt=0)
    rush_probability: float = Field(default=0.05, ge=0, le=1)
    rush_ratio: float = Field(default=0.7, gt=0, lt=1)


class InitialHistory(StrictModel):
    counts: dict[str, int] | None = None

    @model_validator(mode="after")
    def positive(self):
        if self.counts is not None and (
            not self.counts or any(n < 1 for n in self.counts.values())
        ):
            raise ValueError("history requires positive sample counts")
        return self


class Volatility(StrictModel):
    level: Literal["low", "mid", "high"] = "mid"
    eta: float = Field(default=0.5, gt=0, le=1)
    swap_proposals: int = Field(default=100, gt=0)
    minimum_separation: float = Field(default=0, ge=0, lt=1)


class WorkloadV3(StrictModel):
    schema_id: Literal["smartsom.workload/v3"] = Field(alias="schema")
    templates: tuple[WorkloadTemplate, ...] = ()
    demands: tuple[Demand, ...] | None = None
    profile: WorkloadProfile | None = None
    segments: int = Field(default=1, gt=0)
    personalization: Personalization = Field(default_factory=Personalization)
    arrivals: ArrivalSlots = Field(default_factory=ArrivalSlots)
    due: DueFunction = Field(default_factory=DueFunction)
    history: InitialHistory = Field(default_factory=InitialHistory)
    volatility: Volatility = Field(default_factory=Volatility)
    occurrence_bound_seconds: float | None = Field(default=None, gt=0)
    input_id: str | None = Field(default=None, min_length=1)
    mode: Literal["static", "dynamic", "finite"] = "finite"
    tick_limit: int | None = Field(default=None, gt=0)
    horizon_multiplier: float = Field(default=1, ge=1)

    @field_validator("demands", mode="before")
    @classmethod
    def fixed_duration_tables(cls, data):
        if (
            data is None
            or isinstance(data, tuple)
            and all(isinstance(d, Demand) for d in data)
        ):
            return data
        from smartsom.config.production import normalize_duration_tables

        normalized = normalize_duration_tables({"demands": data})["demands"]
        return TypeAdapter(
            tuple[Demand, ...], config=ConfigDict(extra="forbid")
        ).validate_python(normalized)

    @model_validator(mode="after")
    def paired_contract(self):
        if (
            sum(
                (
                    self.demands is not None,
                    self.profile is not None,
                    bool(self.templates),
                )
            )
            != 1
        ):
            raise ValueError(
                "provide fixed demands, a simple profile, or paired templates"
            )
        if self.demands is not None:
            if not self.demands or self.templates:
                raise ValueError(
                    "fixed workload requires nonempty demands and no templates"
                )
            if len({d.demand_id for d in self.demands}) != len(self.demands):
                raise ValueError("fixed workload requires unique Job IDs")
            if self.tick_limit is not None and any(
                d.release_at >= self.tick_limit for d in self.demands
            ):
                raise ValueError("fixed Job arrival must precede tick_limit")
            return self
        if self.profile is not None:
            return self
        if self.mode == "static":
            raise ValueError("generated workloads require finite or dynamic mode")
        if not self.templates or len({t.id for t in self.templates}) != len(
            self.templates
        ):
            raise ValueError("declare uniquely identified base templates")
        scale = 10**self.personalization.decimal_places
        if any(
            Fraction(str(t)) * scale < 1 or (Fraction(str(t)) * scale).denominator != 1
            for template in self.templates
            for t in template.times
        ):
            raise ValueError(
                "template reference seconds must fit the declared decimal precision"
            )
        divisor = sum(self.personalization.category_weights)
        if any(t.count % divisor for t in self.templates):
            raise ValueError(
                "template counts must realize exact personalization quotas"
            )
        segment_jobs = sum(t.count for t in self.templates)
        if segment_jobs % self.arrivals.window_jobs:
            raise ValueError("each shared segment must contain whole arrival windows")
        common = {t.id: t for t in self.templates if not t.novel}
        if not common:
            raise ValueError(
                "initial history needs independently generated common templates"
            )
        if self.history.counts is not None:
            if not set(self.history.counts) <= set(common):
                raise ValueError(
                    "initial history may reference only declared common templates"
                )
            if any(n % divisor for n in self.history.counts.values()):
                raise ValueError(
                    "history counts must realize exact personalization quotas"
                )
        if (
            self.arrivals.window_seconds / Fraction(str(self.arrivals.tick_seconds))
        ).denominator != 1:
            raise ValueError("arrival window boundaries must align with physical ticks")
        if self.occurrence_bound_seconds is not None:
            if (Fraction(str(self.occurrence_bound_seconds)) * scale).denominator != 1:
                raise ValueError(
                    "occurrence_bound_seconds must fit the declared decimal precision"
                )
            if (
                Fraction(str(self.occurrence_bound_seconds))
                < self.minimum_occurrence_bound
            ):
                raise ValueError(
                    "occurrence_bound_seconds is below the generator support bound"
                )
        total = segment_jobs * self.segments
        slots = self.arrivals.slots_seconds
        if slots is None:
            interval = Fraction(str(self.arrivals.interval_seconds))
            slots = tuple(
                Fraction(0) if i < self.arrivals.initial_jobs else i * interval
                for i in range(total)
            )
        if len(slots) != total:
            raise ValueError("arrival slots must fill exactly the complete common pool")
        window_seconds = self.arrivals.window_seconds
        tick_seconds = Fraction(str(self.arrivals.tick_seconds))
        counts = [0] * (total // self.arrivals.window_jobs)
        for slot in slots:
            seconds = Fraction(str(slot))
            window = int(seconds // window_seconds)
            if window >= len(counts):
                raise ValueError(
                    "arrival slot exceeds the declared complete window series"
                )
            physical_seconds = math.ceil(seconds / tick_seconds) * tick_seconds
            if int(physical_seconds // window_seconds) != window:
                raise ValueError("arrival slot rounding would change its V/rush window")
            counts[window] += 1
        if any(n != self.arrivals.window_jobs for n in counts):
            raise ValueError("each declared arrival window must have window_jobs slots")
        return self

    @property
    def minimum_occurrence_bound(self):
        p = self.personalization
        raw_bound = max(
            max(
                Fraction(str(t)) * (1 + Fraction(str(p.adjustment_max)))
                for template in self.templates
                for t in template.times
            ),
            max(
                sum(Fraction(str(t)) for t in template.times)
                * Fraction(str(p.insertion_max))
                for template in self.templates
            ),
        )

        scale = 10**self.personalization.decimal_places
        return Fraction(math.ceil(raw_bound * scale), scale)

    @property
    def occurrence_bound(self):
        return (
            Fraction(str(self.occurrence_bound_seconds))
            if self.occurrence_bound_seconds
            else self.minimum_occurrence_bound
        )

    @property
    def total_jobs(self):
        if self.demands is not None:
            return len(self.demands)
        return self.segments * sum(t.count for t in self.templates)

    @property
    def default_tick_limit(self):
        p, due, arrivals = self.personalization, self.due, self.arrivals
        largest_allowance = max(
            Fraction(str(due.base_seconds))
            + Fraction(str(due.reference_work_factor))
            * sum(Fraction(str(t)) for t in template.times)
            * (1 + Fraction(str(p.adjustment_max)) + Fraction(str(p.insertion_max)))
            + Fraction(str(due.operation_seconds)) * (len(template.route) + 1)
            for template in self.templates
        )
        end = Fraction(str(arrivals.window_seconds)) * (
            self.total_jobs // arrivals.window_jobs
        )
        return math.ceil(
            (end + largest_allowance)
            * Fraction(str(self.horizon_multiplier))
            / Fraction(str(arrivals.tick_seconds))
        )


def load_workload_v3(path: str | Path) -> WorkloadV3:
    return read_model(Path(path).expanduser().resolve(), WorkloadV3)[0]
