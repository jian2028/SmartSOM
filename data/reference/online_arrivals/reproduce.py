"""Optional reference capture from a pinned local JobShopLib checkout.

Run with the locked CP environment and a checkout path. JobShopLib is not a
SmartSOM dependency; only the frozen JSON outputs are used in ordinary tests.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

COMMIT = "460510f197744eed1cbcbbdfd6ec3252252f412c"


def main(checkout: Path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError("reference requires the pinned JobShopLib commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout
    ):
        raise ValueError("reference source must have no tracked modifications")
    sys.path.insert(0, str(checkout))
    from job_shop_lib import JobShopInstance, Operation
    from job_shop_lib.constraint_programming import ORToolsSolver

    instance = JobShopInstance(
        [
            [Operation(0, 3), Operation(1, 2)],
            [Operation(1, 2, release_date=2), Operation(0, 1)],
        ],
        name="online_arrivals_reference",
    )
    solver = ORToolsSolver(max_time_in_seconds=10)
    solver.solver.parameters.num_search_workers = 1
    solver.solver.parameters.random_seed = 1
    result = solver(instance)
    rows = sorted(
        [
            {
                "job": op.operation.job_id,
                "position": op.operation.position_in_job,
                "machine": op.machine_id,
                "start": op.start_time,
                "end": op.end_time,
            }
            for machine in result.schedule
            for op in machine
        ],
        key=lambda row: (row["job"], row["position"]),
    )
    captured = {
        "makespan": result.makespan(),
        "metadata": result.metadata,
        "schedule": rows,
    }
    out = Path(__file__).parent
    raw = (json.dumps(captured, indent=2, allow_nan=False) + "\n").encode()
    (out / "direct_solver_result.json").write_bytes(raw)
    source = checkout / "job_shop_lib/constraint_programming/_ortools_solver.py"
    sources = {
        "repository": "https://github.com/Pabloo22/job_shop_lib",
        "commit": commit,
        "source_path": str(source.relative_to(checkout)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "num_workers": 1,
        "random_seed": 1,
        "time_limit_seconds": 10,
        "conversion": "job 0/1 -> A/B; position 0/1 -> operation suffix 1/2; machine 0/1 -> M1/M2; standard mode. Only B's first operation needs a release constraint; precedence propagates it.",
        "scope": "Offline reference for release-constrained feasibility only; reveal policy is verified independently in SmartSOM tests.",
    }
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps(captured))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
