"""Measured phase service times and read-only queue completion estimates.

No scheduler, learner, process-control or filesystem dependency lives here.
Monotonic absolute timestamps stay in memory; DTOs contain accumulated durations.
"""

import heapq
import math
from collections import defaultdict, deque
from copy import deepcopy
from uuid import uuid4

SCHEMA = "smartsom.execution-observation/v1"
FINAL = {
    "completed",
    "failed",
    "stopped",
    "interrupted",
    "force_stopped",
    "early_stopped",
}
WAITING = {
    "pending",
    "queued",
    "paused",
    "waiting_resources",
    "waiting for resources",
    "waiting for shared control",
    "stopping",
}
PHASES = {
    "training",
    "validation",
    "evaluation",
    "initial",
    "rule",
    "random",
    "saving",
    "loading",
    "finalization",
}


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def phase_of(row):
    stage, values = row.get("stage", ""), row.get("values", {})
    if row.get("status") in WAITING or stage in WAITING:
        return None
    if stage in {"validation", "validating"}:
        return "validation"
    if stage in {"evaluation", "evaluating", "test"}:
        kind = values.get("evaluation_kind") or "evaluation"
        for control in ("initial", "rule", "random"):
            if control in kind:
                return control
        return "evaluation"
    if stage in {"saving", "checkpointing", "checkpoint", "exporting"}:
        return "saving"
    if stage in {"initializing", "restoring", "loading", "restored", "compiling"}:
        return "loading"
    if stage in {"finalizing", "closing", "cleanup"}:
        return "finalization"
    if row.get("status") in FINAL:
        return None
    if row.get("unit") != "entries" and stage in {
        "sampling",
        "updating",
        "optimizing",
        "learning_metrics",
        "running",
        "training",
    }:
        return "training"
    return None


class PhaseMeter:
    """Only real producer events advance clocks and counters, never rendering."""

    def __init__(self, *, window=120, minimum=10, attempt=None):
        self.window, self.minimum = window, minimum
        self.attempt = attempt or uuid4().hex
        self.last = None
        self.phase = None
        self.sequence = 0
        self.counters = {}
        self.cases = {}
        self.case_done = {}
        self.seconds = defaultdict(float)
        self.work = defaultdict(float)
        self.closed = defaultdict(int)
        self.samples = defaultdict(deque)
        self.eta_samples = defaultdict(deque)
        self.updated_at = None
        self.reason = None

    def observe(self, row, *, now, wall_time, attempt=None):
        if (
            not number(now)
            or not number(wall_time)
            or self.last is not None
            and now < self.last
        ):
            return
        if attempt and attempt != self.attempt:
            self.__init__(window=self.window, minimum=self.minimum, attempt=attempt)
        phase = phase_of(row)
        values = row.get("values", {})
        elapsed = now - self.last if self.last is not None else 0
        deltas = defaultdict(float)
        training = values.get("physical_ticks", row.get("completed"))
        if row.get("unit") in {
            "physical ticks",
            "environment steps",
            "adapter decisions",
            "sampling decisions",
        } and number(training):
            old = self.counters.get("training", training)
            if training >= old:
                deltas["training"] = training - old
                self.counters["training"] = training
            else:
                self.reason = "counter regressed without a new attempt"
        if phase in {"validation", "evaluation", "initial", "rule", "random"}:
            prefix = "validation" if phase == "validation" else "evaluation"
            tick, finished = (
                values.get(prefix + "_tick"),
                values.get(prefix + "_finished", 0),
            )
            if number(tick) and number(finished):
                active = values.get(prefix + "_case_active", True)
                group = (
                    phase,
                    values.get("validation_round") if phase == "validation" else 0,
                )
                case = (
                    *group,
                    values.get("evaluation_attempt") if phase != "validation" else None,
                    values.get("data_seed"),
                    values.get("environment_seed"),
                    values.get("policy_root"),
                    max(0, finished - int(not active)),
                )
                old = self.cases.get(case, tick)
                if tick >= old:
                    deltas[phase] += tick - old
                    self.cases[case] = tick
                else:
                    self.reason = "case counter regressed"
                self.case_done[group] = max(finished, self.case_done.get(group, 0))
        if self.phase is not None and elapsed:
            self.seconds[self.phase] += elapsed
            delta = deltas.get(self.phase, 0)
            self.samples[self.phase].append((self.seconds[self.phase], delta, elapsed))
            self.eta_samples[self.phase].append(
                (self.seconds[self.phase], delta, elapsed)
            )
            while (
                len(self.eta_samples[self.phase]) > 1
                and self.seconds[self.phase] - self.eta_samples[self.phase][0][0]
                > self.window
            ):
                self.eta_samples[self.phase].popleft()
            while (
                len(self.samples[self.phase]) > 1
                and self.seconds[self.phase] - self.samples[self.phase][0][0]
                > self.window
            ):
                self.samples[self.phase].popleft()
        for key, delta in deltas.items():
            # A pause/resume interval has no attributable active work window.
            if self.phase is not None:
                self.work[key] += delta
        if self.phase is not None and self.phase != phase:
            self.closed[self.phase] += 1
        if phase is None:
            self.eta_samples.clear()
        self.last, self.phase, self.updated_at = now, phase, wall_time
        self.sequence += 1

    def snapshot(self):
        phases = {}
        for phase in PHASES:
            samples = self.samples[phase]
            span = sum(row[2] for row in samples)
            rate = (
                sum(row[1] for row in samples) / span if span >= self.minimum else None
            )
            recent = self.eta_samples[phase]
            # Time-weighted exponential smoothing is independent of producer
            # event frequency. Keep the measured window rate unchanged.
            weighted_work = weighted_time = 0.0
            for end, work, seconds in recent:
                decay = math.exp(-(self.seconds[phase] - end) / 30.0)
                weight = (1 - math.exp(-seconds / 30.0)) * decay
                weighted_work += (work / seconds) * weight
                weighted_time += weight
            eta_rate = (
                weighted_work / weighted_time
                if weighted_time and sum(item[2] for item in recent) >= self.minimum
                else None
            )
            phases[phase] = {
                "seconds": self.seconds[phase],
                "work": self.work[phase],
                "rate": rate,
                "eta_rate": eta_rate,
                "sample_seconds": span,
                "closed": self.closed[phase],
                "cases_done": sum(
                    value
                    for (name, _), value in self.case_done.items()
                    if name == phase
                ),
                "mean_seconds": self.seconds[phase] / self.closed[phase]
                if self.closed[phase]
                else None,
            }
        return {
            "schema": SCHEMA,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "phases": phases,
            "training_ticks": self.work["training"],
            "execution_ticks": sum(
                self.work[phase]
                for phase in (
                    "training",
                    "validation",
                    "evaluation",
                    "initial",
                    "rule",
                    "random",
                )
            ),
            "reason": self.reason,
        }


def combined_observation(observations, *, attempt, sequence, updated_at):
    phases = {}
    for phase in PHASES:
        rows = [data.get("phases", {}).get(phase, {}) for data in observations]
        seconds = sum(row.get("seconds", 0) for row in rows)
        work = sum(row.get("work", 0) for row in rows)
        closed = sum(row.get("closed", 0) for row in rows)
        valid = [row for row in rows if row.get("rate") is not None]
        span = sum(row.get("sample_seconds", 0) for row in valid)
        phases[phase] = {
            "seconds": seconds,
            "work": work,
            "closed": closed,
            "cases_done": sum(row.get("cases_done", 0) for row in rows),
            "rate": sum(row["rate"] * row["sample_seconds"] for row in valid) / span
            if span
            else None,
            "sample_seconds": span,
            "mean_seconds": seconds / closed if closed else None,
        }
        eta_rows = [row for row in rows if number(row.get("eta_rate"))]
        eta_span = sum(row.get("sample_seconds", 0) for row in eta_rows)
        if any("eta_rate" in row for row in rows):
            phases[phase]["eta_rate"] = (
                sum(row["eta_rate"] * row.get("sample_seconds", 0) for row in eta_rows)
                / eta_span
                if eta_span
                else None
            )
    return {
        "schema": SCHEMA,
        "attempt": attempt,
        "sequence": sequence,
        "updated_at": updated_at,
        "producer_attempts": sorted(
            {
                identity
                for data in observations
                for identity in data.get("producer_attempts", [data["attempt"]])
            }
        ),
        "phases": phases,
        "training_ticks": sum(data.get("training_ticks", 0) for data in observations),
        "execution_ticks": sum(data.get("execution_ticks", 0) for data in observations),
        "reason": (
            "; ".join(
                sorted({data["reason"] for data in observations if data.get("reason")})
            )
            if observations
            else "source has no execution observations"
        )
        or None,
    }


def remaining_service(row, plan, observation, *, reference=None):
    """Missing required rates remain explicit, rather than becoming zero cost."""
    status, values = row.get("status", "running"), row.get("values", {})
    if (
        observation
        and observation.get("aggregate")
        and observation.get("estimate") is not None
    ):
        result = deepcopy(observation["estimate"])
        result["conditional"] = result.get("conditional", False) or status in WAITING
        return result
    if status in FINAL:
        return {"seconds": 0, "known_seconds": 0, "missing": [], "conditional": False}
    phases = (observation or {}).get("phases", {})
    costs = reference or {}
    training_total = plan.get("training_total", row.get("total", 0)) or 0
    training_done = values.get("physical_ticks", row.get("completed", 0)) or 0
    completed_phases = set((observation or {}).get("completed_phases", []))
    if "training" in completed_phases:
        training_done = training_total
    elif number((observation or {}).get("physical_ticks")):
        training_done = observation["physical_ticks"]
    mode = plan.get("mode", "training")
    obligations = {}
    if mode != "evaluation" and training_total:
        obligations["training"] = max(0, training_total - training_done)
        updates = plan.get("validation_updates")
        completed = values.get("validation_completed_updates")
        if updates is not None and completed is not None:
            left = len(set(updates) - set(completed))
        else:
            left = max(
                0,
                (plan.get("validation_rounds") or 0)
                - (values.get("validation_batches_finished") or 0),
            )
        validation = sum(plan.get("validation_horizons", [])) * left
        if "validation" in completed_phases:
            validation = 0
        pending = values.get("validation_pending_update")
        if pending is None:
            pending = values.get("updates")
        current_unfinished = (
            completed is None or pending is not None and pending not in completed
        )
        if phase_of(row) == "validation" and left and current_unfinished:
            from smartsom.telemetry.workflow import case_credit

            validation -= case_credit(
                values, "validation", plan.get("validation_horizons", [])
            )
        obligations["validation"] = max(0, validation)
    if mode in {"train-evaluate", "study-entry", "evaluation"}:
        horizons = plan.get("evaluation_horizons", [])
        evaluation_phases = (
            [plan.get("evaluation_phase", "evaluation")]
            if mode == "evaluation"
            else ["evaluation", *plan.get("control_names", [])]
        )
        for phase in evaluation_phases:
            if phase == "initial":
                horizons = plan.get(
                    "initial_control_horizons", plan.get("evaluation_horizons", [])
                )
            else:
                horizons = plan.get("evaluation_horizons", [])
            if phase in completed_phases:
                obligations[phase] = 0
                continue
            done = phases.get(phase, {}).get("cases_done", 0)
            if phase == "evaluation" and phase_of(row) == "evaluation":
                done = max(done, values.get("evaluation_finished", 0) or 0)
            remaining = sum(horizons[min(len(horizons), int(done)) :])
            progress = values.get("evaluation_case_progress", {}).get(
                "initial_control" if phase == "initial" else phase
            )
            if progress and number(progress.get("evaluation_remaining_ticks")):
                remaining = progress["evaluation_remaining_ticks"]
            if phase_of(row) == phase and number(
                values.get("evaluation_remaining_ticks")
            ):
                # Verified receipts may cover a sparse set. Do not infer a
                # completed prefix from its length or divide restored work by
                # this attempt's elapsed time.
                remaining = values["evaluation_remaining_ticks"]
            if phase_of(row) == phase and values.get("evaluation_case_active"):
                remaining -= min(
                    values.get("evaluation_tick", 0) or 0,
                    values.get("evaluation_tick_limit", 0)
                    if number(values.get("evaluation_remaining_ticks"))
                    else horizons[min(int(done), len(horizons) - 1)]
                    if horizons
                    else 0,
                )
            obligations[phase] = max(0, remaining)
    known, missing = 0.0, []
    for phase, work in obligations.items():
        if not work:
            continue
        measured = phases.get(phase, {})
        rate = measured.get("eta_rate", measured.get("rate"))
        if rate is None and (
            "eta_rate" not in measured or not measured.get("sample_seconds")
        ):
            reference_phase = costs.get(phase, {})
            rate = reference_phase.get("eta_rate", reference_phase.get("rate"))
            if rate is None:
                rate = reference_phase.get("rate")
        if number(rate) and rate > 0:
            known += work / rate
        else:
            missing.append(
                phase
                + (": no recent progress" if rate == 0 else ": collecting phase rate")
            )
    if status in {"queued", "pending"} or phase_of(row) == "loading":
        load = costs.get("loading", {}).get("mean_seconds")
        if load is None:
            missing.append("loading: collecting startup cost")
        else:
            known += max(0, load - phases.get("loading", {}).get("seconds", 0))
    # Checkpoint costs are priced separately; finalization is never implied by ticks.
    rounds = (
        math.ceil(
            max(0, training_total - training_done)
            / (plan.get("round_size") or max(1, training_total))
        )
        if training_total
        else 0
    )
    if rounds:
        mean = phases.get("saving", {}).get("mean_seconds") or costs.get(
            "saving", {}
        ).get("mean_seconds")
        if mean is None:
            missing.append("saving: collecting checkpoint cost")
        else:
            known += rounds * mean
    final_cost = phases.get("finalization", {}).get("mean_seconds")
    if final_cost is None:
        final_cost = costs.get("finalization", {}).get("mean_seconds")
    if final_cost is None:
        missing.append("finalization: collecting completion cost")
    else:
        known += final_cost
    conditional_states = WAITING - {"queued", "pending"}
    conditional = (
        status in conditional_states
        or row.get("stage") in conditional_states
        or bool(row.get("resource_wait"))
    )
    return {
        "seconds": known if not missing else None,
        "known_seconds": known,
        "missing": missing,
        "conditional": conditional,
        "phase_work": obligations,
    }


def task_work_progress(row, plan, observation=None):
    """Frozen work budgets, including validation and evaluation, not elapsed time."""
    mode = plan.get("mode")
    training = mode != "evaluation"
    testing = mode in {"evaluation", "train-evaluate", "study-entry"}
    if (
        not mode
        or training
        and not number(plan.get("training_total", row.get("total")))
        or plan.get("validation_rounds")
        and not plan.get("validation_horizons")
        or testing
        and not plan.get("evaluation_horizons")
    ):
        return {
            "completed": None,
            "total": None,
            "fraction": 1.0 if row.get("status") == "completed" else None,
        }
    initial = {
        "status": "queued",
        "values": {},
        "completed": 0,
        "total": row.get("total"),
    }
    planned = remaining_service(initial, plan, {}).get("phase_work", {})
    total = sum(planned.values())
    if not total:
        return {
            "completed": None,
            "total": None,
            "fraction": 1.0 if row.get("status") == "completed" else None,
        }
    if row.get("status") == "completed":
        return {"completed": total, "total": total, "fraction": 1.0}
    current = dict(row, status="running")
    # Failure/paused snapshots still carry useful completed work, never 100% success.
    data = dict(observation or {})
    data.pop("aggregate", None)
    left = remaining_service(current, plan, data).get("phase_work", {})
    completed = max(0, min(total, total - sum(left.values())))
    return {
        "completed": completed,
        "total": total,
        "fraction": min(0.99, completed / total),
    }


def batch_remaining(jobs, *, limit):
    """Heap model of the submitted queue, barriers and draining occupied slots."""
    jobs = deepcopy(jobs)
    limit = max(1, int(limit))
    known = all(number(job.get("seconds")) for job in jobs)
    conditional = any(job.get("conditional") for job in jobs)
    done = {job["id"] for job in jobs if job.get("status") in FINAL}
    completion = dict.fromkeys(done, 0.0)
    heap, pending = [], []
    for index, job in enumerate(jobs):
        if job["id"] in done:
            continue
        duration = job.get("seconds") if known else job.get("known_seconds", 0)
        duration = duration if number(duration) else 0
        job["duration"], job["order"] = duration, index
        if job.get("status") in {"running", "staged", "pending", "stopping"}:
            heapq.heappush(heap, (duration, index, job["id"]))
        else:
            pending.append(job)
    now = 0.0
    while heap or pending:
        launched = False
        for job in tuple(pending):
            earlier = {
                other["id"]
                for other in jobs
                if other.get("barrier_group") is not None
                and other.get("barrier_group") == job.get("barrier_group")
                and other.get("stage", 0) < job.get("stage", 0)
            }
            if (
                len(heap) >= limit
                or not (set(job.get("requires", [])) | earlier) <= done
            ):
                continue
            heapq.heappush(heap, (now + job["duration"], job["order"], job["id"]))
            pending.remove(job)
            launched = True
        if heap:
            now, _, identity = heapq.heappop(heap)
            done.add(identity)
            completion[identity] = now
        elif pending and not launched:
            return {
                "seconds": None,
                "known_seconds": now,
                "conditional": True,
                "reason": "dependency or stage completion unavailable",
            }
    return {
        "seconds": now if known and not conditional else None,
        "conditional_seconds": now if known and conditional else None,
        "known_seconds": now,
        "conditional": conditional,
        "reason": None if known else "required phase observations unavailable",
        "completion_seconds": completion if known and not conditional else {},
        "conditional_completion_seconds": completion if known and conditional else {},
    }


def coarse_time(seconds):
    if seconds is None:
        return "采集中"
    if seconds < 60:
        return "<1 min" if seconds else "0 min"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{math.ceil(seconds / 360) / 10:g} h"
