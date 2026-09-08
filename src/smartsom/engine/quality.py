"""Engine-owned quality state; completion and inspection use the existing clock."""

from dataclasses import replace
from fractions import Fraction

from smartsom.domain import OperationStatus
from smartsom.domain.quality import (
    DRAW_DENOMINATOR,
    JobQualityView,
    OperationQuality,
    QualityModeView,
    QualityResult,
)
from smartsom.engine.invariants import InvariantViolation
from smartsom.trace.records import InspectionRecord, QualityRecord


class QualityExecution:
    def __init__(self, module, probability_visibility):
        self.module = module
        self.probability_visibility = probability_visibility
        self.jobs = {
            job.job_id: job for order in module.workload.orders for job in order.jobs
        }
        self.operation_jobs = {
            op.operation_id: job.job_id
            for job in self.jobs.values()
            for op in job.operations
        }
        self.outcomes = {}
        self.defective_jobs = set()
        self.inspections = {}

    def complete(self, event, trace):
        op = event.operation_id
        if op in self.outcomes:
            raise InvariantViolation("operation quality evaluated twice")
        row = self.module.modes[(op, event.processing_mode_id)]
        draw = self.module.draws[op]
        defective = Fraction(draw, DRAW_DENOMINATOR) < Fraction(row.mode.error_rate)
        job = self.operation_jobs[op]
        self.outcomes[op] = OperationQuality(
            op, event.processing_mode_id, job, event.simulation_time, defective
        )
        if defective:
            self.defective_jobs.add(job)
        trace.append(
            QualityRecord(
                len(trace),
                event.simulation_time,
                self.outcomes[op],
                draw,
                row.mode.error_rate,
                job in self.defective_jobs,
            )
        )

    def inspect_ready(self, state, transport, trace):
        for job_id, job in sorted(self.jobs.items()):
            if job_id in self.inspections:
                continue
            ready = (
                transport.positions[job_id].location.kind == "output"
                if transport is not None
                else all(op.operation_id in self.outcomes for op in job.operations)
            )
            if ready:
                inspection = JobQualityView(
                    job_id, job_id not in self.defective_jobs, state.simulation_time
                )
                self.inspections[job_id] = inspection
                trace.append(
                    InspectionRecord(len(trace), state.simulation_time, inspection)
                )

    def check(self, state, transport):
        completed = {
            key
            for key, op in state.operations.items()
            if op.status == OperationStatus.COMPLETED
        }
        if self.outcomes.keys() != completed:
            raise InvariantViolation(
                "quality outcomes disagree with completed operations"
            )
        if self.defective_jobs != {
            x.job_id for x in self.outcomes.values() if x.defective
        }:
            raise InvariantViolation(
                "sticky quality state disagrees with operation outcomes"
            )
        for key, outcome in self.outcomes.items():
            op = state.operations[key]
            row = self.module.modes[(key, op.processing_mode_id)]
            if (
                outcome.job_id != self.operation_jobs[key]
                or outcome.processing_mode_id != op.processing_mode_id
                or outcome.completion_time != op.completion_time
                or outcome.defective
                != (
                    Fraction(self.module.draws[key], DRAW_DENOMINATOR)
                    < Fraction(row.mode.error_rate)
                )
            ):
                raise InvariantViolation("quality outcome disagrees with executed mode")
        for key, inspection in self.inspections.items():
            if (
                not all(
                    op.operation_id in completed for op in self.jobs[key].operations
                )
                or inspection.passed != (key not in self.defective_jobs)
                or inspection.inspection_time
                < max(
                    self.outcomes[op.operation_id].completion_time
                    for op in self.jobs[key].operations
                )
                or inspection.inspection_time > state.simulation_time
                or (
                    transport is not None
                    and transport.positions[key].location.kind != "output"
                )
            ):
                raise InvariantViolation(
                    "inspection precedes completion/output or disagrees with quality"
                )

    def project(self, context):
        visible = {op.operation_id for op in context.operations}
        views = tuple(
            QualityModeView(
                row.operation_id,
                row.processing_mode_id,
                row.base_processing_mode_id,
                row.mode.quality_mode_id,
                row.machine_id,
                row.mode.time_scale,
                row.nominal_ticks,
                row.mode.error_rate
                if self.probability_visibility == "public"
                else None,
            )
            for row in self.module.plan.modes
            if row.operation_id in visible
        )
        return replace(
            context,
            quality_modes=views,
            job_quality=tuple(
                self.inspections.get(job.job_id, JobQualityView(job.job_id))
                for job in context.jobs
            ),
        )

    def result(self):
        if self.inspections.keys() != self.jobs.keys():
            raise InvariantViolation("terminal quality is missing inspections")
        return QualityResult(
            tuple(self.outcomes[key] for key in sorted(self.outcomes)),
            tuple(self.inspections[key] for key in sorted(self.inspections)),
        )
