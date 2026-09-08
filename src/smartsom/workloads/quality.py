"""Independent operation-stable quality draws, materialized before simulation."""

import hashlib
import json
import random

from smartsom.domain import WorkloadInstance
from smartsom.domain.quality import QUALITY_VERSION, QualityDraw, QualityDrawPlan
from smartsom.domain.validation import _identifier


def operation_draw(seed: int, operation_id: str) -> int:
    _identifier(operation_id, "operation ID")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("quality seed must be an unsigned 64-bit integer")
    identity = json.dumps(
        [QUALITY_VERSION, seed, operation_id], ensure_ascii=False, separators=(",", ":")
    )
    derived = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")
    return random.Random(derived).getrandbits(53)


def generate_quality(workload: WorkloadInstance, seed: int) -> QualityDrawPlan:
    return QualityDrawPlan(
        tuple(
            QualityDraw(op.operation_id, operation_draw(seed, op.operation_id))
            for op in workload.operations
        )
    )
