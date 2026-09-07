"""Versioned per-machine renewal processes, materialized before execution."""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction

from smartsom.domain.machine_events import MachineOutage, MachineOutagePlan
from smartsom.domain.models import FactorySpec, _identifier
from smartsom.workloads.static_jsp import IntegerRange

GENERATOR_VERSION = "smartsom.machine-events/v1"


@dataclass(frozen=True, slots=True)
class MachineFailureProfile:
    machine_id: str
    mean_uptime_ticks: float
    repair_ticks: IntegerRange

    def __post_init__(self):
        _identifier(self.machine_id, "machine_id")
        value = self.mean_uptime_ticks
        if type(value) not in (int, float):
            raise ValueError("mean_uptime_ticks must be positive and finite")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("mean_uptime_ticks must be positive and finite") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mean_uptime_ticks must be positive and finite")
        if not isinstance(self.repair_ticks, IntegerRange):
            raise ValueError("repair_ticks must be IntegerRange")
        object.__setattr__(self, "mean_uptime_ticks", value)


@dataclass(frozen=True, slots=True)
class MachineEventProfile:
    generation_until_tick: int
    machines: tuple[MachineFailureProfile, ...]

    def __post_init__(self):
        if (
            type(self.generation_until_tick) is not int
            or self.generation_until_tick < 1
        ):
            raise ValueError("generation_until_tick must be a positive integer")
        if (
            not isinstance(self.machines, (tuple, list))
            or not self.machines
            or any(not isinstance(row, MachineFailureProfile) for row in self.machines)
        ):
            raise ValueError("machines must be nonempty MachineFailureProfile values")
        ids = [row.machine_id for row in self.machines]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate machine profile")
        object.__setattr__(
            self,
            "machines",
            tuple(sorted(self.machines, key=lambda row: row.machine_id)),
        )


def machine_seed(seed: int, machine_id: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("machine_events seed must be an unsigned 64-bit integer")
    _identifier(machine_id, "machine_id")
    identity = json.dumps(
        [GENERATOR_VERSION, seed, machine_id], ensure_ascii=False, separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")


def generate_machine_events(
    factory: FactorySpec, profile: MachineEventProfile, seed: int
) -> MachineOutagePlan:
    if not isinstance(profile, MachineEventProfile):
        raise ValueError("expected MachineEventProfile")
    known = {machine.machine_id for machine in factory.machines}
    if any(row.machine_id not in known for row in profile.machines):
        raise ValueError("machine profile references unknown machine")
    rows = []
    for machine in profile.machines:
        rng = random.Random(machine_seed(seed, machine.machine_id))
        repaired_at = 0
        while repaired_at < profile.generation_until_tick:
            # Inverse exponential draw. Rational multiplication avoids overflow
            # and defines ceiling exactly for the two binary64 input values.
            unit = -math.log1p(-rng.random())
            uptime = max(
                1, math.ceil(Fraction(unit) * Fraction(machine.mean_uptime_ticks))
            )
            start = repaired_at + uptime
            if start >= profile.generation_until_tick:
                break
            repaired_at = start + rng.randint(
                machine.repair_ticks.min, machine.repair_ticks.max
            )
            rows.append(MachineOutage(machine.machine_id, start, repaired_at))
    return MachineOutagePlan(tuple(rows))
