"""Versioned numeric fields for detached Base observations.

Historical native-v1 is available only in archived source.
The current layout moves the rejected-target count out of the next-operation one-hot.
This does not claim byte compatibility with separately evolved server/Mac inputs.
"""

from dataclasses import dataclass
from typing import Literal

ObservationVersion = Literal["native-v2"]


@dataclass(frozen=True, slots=True)
class VehicleFields:
    distance: int = 0
    loaded: int = 1
    service_locked: int = 2
    arrival_age: int = 3
    job_start: int = 4
    job_stop: int = 16
    x: int = 16
    y: int = 17
    following_start: int = 18
    following_stop: int = 22

    @property
    def following_operation(self):
        return slice(self.following_start, self.following_stop)


VEHICLE = VehicleFields()
REJECTED_TARGET_COUNT = {"native-v2": 22}


def rejected_target_index(version: ObservationVersion) -> int:
    if version not in REJECTED_TARGET_COUNT:
        raise ValueError("historical observation layout requires archived source")
    return REJECTED_TARGET_COUNT[version]


# Inclusive start / exclusive stop; all rows have FEATURES=48 and trailing zeros.
# These descriptors are constructed once at import, never parsed during rollout.
JOB_FIELDS = {
    "due_slack": 0,
    "remaining_standard_work": 1,
    "queue_age": 2,
    "rush": 3,
    "remaining_operations": 4,
    "public_risk": 5,
    "current_operation": slice(6, 10),
    "service_locked": 10,
    "queued": 11,
    "following_operation": slice(12, 16),
}
SUMMARY_FIELDS = {
    "count": 0,
    "minimum_slack": 1,
    "mean_slack": 2,
    "overdue_count": 3,
    "total_work": 4,
    "mean_work": 5,
    "maximum_work": 6,
    "rush_count": 7,
    "rush_fraction": 8,
    "mean_queue_age": 9,
    "maximum_queue_age": 10,
    "queued_count": 11,
}
STATION_FIELDS = {
    "inventory_kind": 0,
    "quota": 1,
    "commitments": 2,
    "down": 3,
    "processing": 4,
    "blocked": 5,
    "pre_standard_work": 6,
    "jobs": slice(7, 19),
    "pre_capacity": 19,
    "post_capacity": 20,
    "operation_capabilities": slice(21, 25),
}
MACHINE_CANDIDATE_FIELDS = {
    "job": slice(0, 12),
    "nominal_duration": 12,
    "error_rate": 13,
    "time_scale": 14,
    "normal_mode": 15,
}
DISPATCHER_CANDIDATE_FIELDS = {
    "distance": 0,
    "quota": 1,
    "ready_jobs": slice(2, 14),
    "commitments": 14,
    "pre_jobs": slice(15, 27),
    "machine_jobs": slice(27, 39),
    "down": 39,
    "processing": 40,
    "blocked": 41,
    "pre_nominal_work": 42,
    "post_count": 43,
}
LOAD_CANDIDATE_FIELDS = {
    "job": slice(0, 16),
    "distance": 16,
    "loaded": 17,
    "service_locked": 18,
    "arrival_age": 19,
    "x": 20,
    "y": 21,
}
CRITIC_FIELDS = {
    "released_active_jobs": slice(0, 12),
    "completed_count": 12,
    "down_count": 13,
    "blocked_count": 14,
    "processing_count": 15,
    "loaded_vehicle_count": 16,
    "committed_vehicle_count": 17,
}
