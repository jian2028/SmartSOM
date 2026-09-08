"""Read-only progress notifications, separate from semantic evidence."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunProgress:
    run_dir: Path
    stage: str
    elapsed_seconds: float
    simulation_time: int = 0
    completed_operations: int = 0
    delivered_jobs: int = 0
    inspected_jobs: int = 0
    passed_jobs: int = 0
