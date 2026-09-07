"""The only external-index boundary for the optional PyJobShop CP-SAT provider."""

from smartsom.algorithms.solver import ScheduleSolution, SolveRequest, SolverStatus
from smartsom.domain import ScheduledOperation


class PyJobShopAdapter:
    def solve(self, request: SolveRequest) -> ScheduleSolution:
        try:
            from pyjobshop import Model
            from pyjobshop.constants import MAX_VALUE
        except ImportError as exc:
            raise ImportError(
                "pyjobshop.cp_sat requires the optional cp extra; install with uv sync --extra cp"
            ) from exc

        operations = sorted(request.workload.operations, key=lambda op: op.operation_id)
        if sum(op.modes[0].nominal_ticks for op in operations) > MAX_VALUE:
            raise ValueError(
                f"instance exceeds PyJobShop's supported horizon {MAX_VALUE}"
            )
        model = Model()
        machines = {
            machine.machine_id: model.add_machine(name=machine.machine_id)
            for machine in sorted(
                request.factory.machines, key=lambda machine: machine.machine_id
            )
        }
        resource_ids = tuple(machines)
        jobs = {}
        operation_jobs = {}
        for job in sorted(
            (job for order in request.workload.orders for job in order.jobs),
            key=lambda job: job.job_id,
        ):
            jobs[job.job_id] = model.add_job(name=job.job_id)
            operation_jobs.update(
                {op.operation_id: job.job_id for op in job.operations}
            )
        tasks = {}
        # Task and mode arrays both follow this explicit, stable semantic mapping.
        mode_ids = []
        for op in operations:
            task = model.add_task(
                job=jobs[operation_jobs[op.operation_id]], name=op.operation_id
            )
            tasks[op.operation_id] = task
            mode = op.modes[0]
            model.add_mode(task, machines[mode.machine_id], mode.nominal_ticks)
            mode_ids.append((op.operation_id, mode.processing_mode_id))
        for op in operations:
            for predecessor in op.predecessor_ids:
                model.add_end_before_start(tasks[predecessor], tasks[op.operation_id])
        raw = model.solve(
            solver="ortools",
            time_limit=request.solver_time_limit_seconds,
            num_workers=1,
            random_seed=request.backend_seed,
            display=False,
        )
        status = SolverStatus[raw.status.name]
        schedule = []
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            if len(raw.best.tasks) != len(operations):
                raise ValueError("solver omitted mandatory tasks")
            for op, task in zip(operations, raw.best.tasks, strict=True):
                if (
                    not task.present
                    or task.idle
                    or task.breaks
                    or len(task.resources) != 1
                ):
                    raise ValueError(
                        f"unsupported solver task allocation: {op.operation_id}"
                    )
                if (
                    not 0 <= task.mode < len(mode_ids)
                    or mode_ids[task.mode][0] != op.operation_id
                ):
                    raise ValueError(
                        f"incorrect solver mode mapping: {op.operation_id}"
                    )
                resource = task.resources[0]
                if not 0 <= resource < len(resource_ids):
                    raise ValueError(f"unknown solver resource index: {resource}")
                schedule.append(
                    ScheduledOperation(
                        op.operation_id,
                        mode_ids[task.mode][1],
                        resource_ids[resource],
                        task.start,
                        task.end,
                    )
                )
        return ScheduleSolution(
            tuple(
                sorted(
                    schedule, key=lambda entry: (entry.start_time, entry.operation_id)
                )
            ),
            status,
            raw.objective if schedule else None,
            raw.lower_bound if schedule else None,
            raw.runtime,
        )
