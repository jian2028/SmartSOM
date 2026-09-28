"""Factory fault rules and the declared speed/quality heterogeneity contract.

Fault randomness is keyed by stable machine identity, independent of the factory
name, learner and scheduling policy. The generated half-open intervals include
idle uptime; their realized schedule belongs to a frozen run, not this document.
"""

import math
import random
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.codec import ConfigurationError, canonical_json, primitive
from smartsom.config.models import StrictModel
from smartsom.domain.production import Outage


class UniformUptime(StrictModel):
    distribution: Literal["uniform"] = "uniform"
    min_ticks: int = Field(ge=1)
    max_ticks: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.max_ticks < self.min_ticks:
            raise ValueError("uptime max_ticks must be at least min_ticks")
        return self


class ExponentialUptime(StrictModel):
    distribution: Literal["exponential"] = "exponential"
    mean_ticks: float = Field(gt=0, allow_inf_nan=False)


Uptime = Annotated[
    UniformUptime | ExponentialUptime, Field(discriminator="distribution")
]


class RepairDuration(StrictModel):
    min_ticks: int = Field(ge=1)
    max_ticks: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.max_ticks < self.min_ticks:
            raise ValueError("repair max_ticks must be at least min_ticks")
        return self


class MachineReliability(StrictModel):
    enabled: bool = True
    uptime: Uptime
    repair: RepairDuration


class MachineReliabilityOverride(StrictModel):
    enabled: bool | None = None
    uptime: Uptime | None = None
    repair: RepairDuration | None = None

    @model_validator(mode="after")
    def nonempty(self):
        if all(getattr(self, key) is None for key in ("enabled", "uptime", "repair")):
            raise ValueError("machine reliability override must change a field")
        return self


class FactoryReliability(StrictModel):
    enabled: bool = False
    defaults: MachineReliability | None = None
    machines: dict[str, MachineReliabilityOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identifiers(self):
        if any(not key or not key.strip() for key in self.machines):
            raise ValueError(
                "reliability machines requires stable nonempty Machine IDs"
            )
        return self

    def for_machine(self, machine_id):
        if not self.enabled:
            return None
        override = self.machines.get(machine_id)
        values = self.defaults.model_dump() if self.defaults is not None else {}
        if override is not None:
            values.update(override.model_dump(exclude_none=True))
        if values.get("enabled") is False:
            return None
        if "uptime" not in values or "repair" not in values:
            raise ConfigurationError(
                f"reliability for {machine_id!r} needs uptime and repair rules"
            )
        return MachineReliability.model_validate(values)


def validate_factory_reliability(document):
    reliability = document.reliability
    if reliability is None:
        return
    machine_ids = {machine.machine_id for machine in document.factory.machines}
    unknown = set(reliability.machines) - machine_ids
    if unknown:
        raise ConfigurationError(
            "reliability refers to unknown Machine IDs: " + ", ".join(sorted(unknown))
        )
    for machine_id in sorted(machine_ids):
        reliability.for_machine(machine_id)


def factory_reliability_outages(
    document, seed: int, until_tick: int
) -> tuple[Outage, ...]:
    """Freeze one policy-independent schedule using the supplied data seed.

    Availability starts at tick zero. A sampled uptime starts after the preceding
    repair; a repair reaching the horizon is clipped there. No random failure is
    generated during a repair, and no active-job or policy state is inspected.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ConfigurationError("fault data seed must be an unsigned 64-bit integer")
    if type(until_tick) is not int or until_tick < 1:
        raise ConfigurationError("fault horizon must be a positive integer tick")
    from smartsom.config.production import named_seed

    validate_factory_reliability(document)
    if document.reliability is None or not document.reliability.enabled:
        return ()
    rows = []
    for machine in sorted(document.factory.machines, key=lambda m: m.machine_id):
        rule = document.reliability.for_machine(machine.machine_id)
        if rule is None:
            continue
        rng = random.Random(
            named_seed(seed, f"factory-reliability/v1/{machine.machine_id}")
        )
        repaired_at = 0
        while repaired_at < until_tick:
            if isinstance(rule.uptime, UniformUptime):
                uptime = rng.randint(rule.uptime.min_ticks, rule.uptime.max_ticks)
            else:
                sampled = rng.expovariate(1 / rule.uptime.mean_ticks)
                # Very large finite means may overflow the sampled float. Such
                # an uptime simply lies beyond this finite run's horizon.
                uptime = (
                    until_tick
                    if not math.isfinite(sampled) or sampled >= until_tick
                    else max(1, math.ceil(sampled))
                )
            start = repaired_at + uptime
            if start >= until_tick:
                break
            repaired_at = min(
                until_tick,
                start + rng.randint(rule.repair.min_ticks, rule.repair.max_ticks),
            )
            rows.append(Outage(machine.machine_id, start, repaired_at))
    return tuple(sorted(rows, key=lambda row: (row.start, row.machine_id)))


# Section 4.1 of the approved mathematical model; rates are relative to each
# operation's reference rate, so operation reference work cancels in H.
_REFERENCE = {
    "slow": (1 / 1.2, 0.990),
    "normal": (1.0, 0.970),
    "fast": (1 / 0.8, 0.950),
}


@dataclass(frozen=True, slots=True)
class FactoryHeterogeneity:
    supported: bool
    h_speed: float | None
    h_quality: float | None
    h: float | None
    quality_scale: float = 0.01
    speed_weight: float = 0.5
    components: tuple[dict, ...] = ()
    diagnostic: str | None = None


def factory_heterogeneity(document) -> FactoryHeterogeneity:
    """Compute MD H from static speed and quality tables, never fault events.

    Legacy designs without the MD's three-mode table remain readable and runnable;
    they receive an explicit unsupported diagnostic rather than a fabricated H.
    """
    design = document.factory
    if not design.operation_types:
        return FactoryHeterogeneity(
            False, None, None, None, diagnostic="No operation types for H"
        )
    components = []
    for operation in design.operation_types:
        machines = [m for m in design.machines if operation in m.operation_types]
        if not machines:
            return FactoryHeterogeneity(
                False,
                None,
                None,
                None,
                diagnostic=f"No compatible machine for {operation!r}",
            )
        for mode_id, (reference_rate, reference_quality) in _REFERENCE.items():
            modes = [
                next((q for q in m.quality_modes if q.quality_mode_id == mode_id), None)
                for m in machines
            ]
            if any(mode is None for mode in modes):
                return FactoryHeterogeneity(
                    False,
                    None,
                    None,
                    None,
                    diagnostic="H requires slow, normal and fast modes on every compatible machine",
                )
            rates = [
                float(m.processing_rate_multiplier / q.time_scale)
                for m, q in zip(machines, modes, strict=True)
            ]
            if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ConfigurationError(
                    "H requires representable finite positive rates"
                )
            qualities = [1 - float(q.error_rate) for q in modes]
            total = sum(rates)
            components.append(
                {
                    "operation_type": operation,
                    "mode": mode_id,
                    "machine_count": len(machines),
                    "total_rate": total,
                    "reference_total_rate": len(machines) * reference_rate,
                    "weighted_quality": sum(
                        c * q for c, q in zip(rates, qualities, strict=True)
                    )
                    / total,
                    "reference_quality": reference_quality,
                    "speed_squared": sum(
                        ((c - reference_rate) / reference_rate) ** 2 for c in rates
                    )
                    / len(rates),
                    "quality_squared": sum(
                        c / total * ((q - reference_quality) / 0.01) ** 2
                        for c, q in zip(rates, qualities, strict=True)
                    ),
                }
            )
    speed = math.sqrt(sum(row["speed_squared"] for row in components) / len(components))
    quality = math.sqrt(
        sum(row["quality_squared"] for row in components) / len(components)
    )
    return FactoryHeterogeneity(
        True,
        speed,
        quality,
        math.sqrt(0.5 * speed**2 + 0.5 * quality**2),
        components=tuple(components),
    )


def _condition_structure(design):
    data = primitive(design)
    data.pop("factory_id")
    data.pop("name")
    data["operation_types"].sort()
    for collection in (
        "machines",
        "buffers",
        "inspection_stations",
        "scrap_bins",
        "chargers",
        "ports",
        "agvs",
    ):
        for row in data[collection]:
            row.pop("name", None)
            if collection == "machines":
                row.pop("processing_rate_multiplier", None)
                row.pop("quality_modes", None)
                row["operation_types"].sort()
        identity = {
            "machines": "machine_id",
            "buffers": "buffer_id",
            "inspection_stations": "inspection_station_id",
            "scrap_bins": "scrap_bin_id",
            "chargers": "charger_id",
            "ports": "port_id",
            "agvs": "agv_id",
        }[collection]
        data[collection].sort(key=lambda row: row[identity])
    return canonical_json(data)


def validate_factory_conditions(documents) -> tuple[FactoryHeterogeneity, ...]:
    """Check H conditions within each declared machine-count/grid scale.

    Single or legacy factories stay valid. Multiple MD-compatible conditions at
    the same scale must freeze semantic identities, geometry and capabilities,
    and each condition must retain the MD nominal speed and qualified capacity.
    """
    documents = tuple(documents)
    reports = tuple(factory_heterogeneity(document) for document in documents)
    scales = {}
    for document, report in zip(documents, reports, strict=True):
        design = document.factory
        scale = (
            design.grid.width,
            design.grid.height,
            len(design.machines),
        )
        scales.setdefault(scale, []).append((document, report))
    for group in scales.values():
        if len(group) < 2:
            continue
        structure = _condition_structure(group[0][0].factory)
        if any(
            _condition_structure(document.factory) != structure
            for document, _ in group[1:]
        ):
            raise ConfigurationError(
                "same-scale Factory conditions must preserve geometry, stable IDs and complete compatibility"
            )
        if not all(report.supported for _, report in group):
            if any(report.supported for _, report in group):
                raise ConfigurationError(
                    "same-scale H conditions must share the slow/normal/fast mode interface"
                )
            continue
        for document, report in group:
            for row in report.components:
                if not math.isclose(
                    row["total_rate"],
                    row["reference_total_rate"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ConfigurationError(
                        f"{document.factory.factory_id}: H condition changes nominal rate for {row['operation_type']}/{row['mode']}"
                    )
                if not math.isclose(
                    row["weighted_quality"],
                    row["reference_quality"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ConfigurationError(
                        f"{document.factory.factory_id}: H condition changes capacity-weighted quality for {row['operation_type']}/{row['mode']}"
                    )
    return reports
