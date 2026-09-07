"""Versioned named seeds; inactive domains are derived but never consumed."""

import hashlib
from dataclasses import dataclass

SEED_VERSION = "smartsom.seed/v1"
DOMAINS = (
    "workload",
    "demand",
    "machine_events",
    "processing_time",
    "algorithm",
    "solver",
)


@dataclass(frozen=True, slots=True)
class NamedSeed:
    domain: str
    value: int
    consumed: bool


def derive_seeds(
    root: int, *, generated: bool, solver: bool = False, demand: bool = False
) -> tuple[NamedSeed, ...]:
    if type(root) is not int or not 0 <= root < 2**64:
        raise ValueError("root seed must be an unsigned 64-bit integer")
    return tuple(
        NamedSeed(
            domain,
            int.from_bytes(
                hashlib.sha256(
                    f"{SEED_VERSION}\0{root}\0{domain}".encode("utf-8")
                ).digest()[:8],
                "big",
            ),
            (generated and domain == "workload")
            or (solver and domain == "solver")
            or (demand and domain == "demand"),
        )
        for domain in DOMAINS
    )
