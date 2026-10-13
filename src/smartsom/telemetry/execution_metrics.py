"""Execution observations shared by native, Tune and read-only projections."""

import time
from collections import deque
from copy import deepcopy
from uuid import uuid4

from smartsom.telemetry.estimates import (
    FINAL,
    SCHEMA,
    PhaseMeter,
    batch_remaining,
    coarse_time,
    combined_observation,
    number,
    remaining_service,
)

SINGLE = {"run", "training", "evaluation", "train-evaluate", "tune-native"}


def validate_observation(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("invalid execution observation")
    if not isinstance(data.get("attempt"), str) or not data["attempt"]:
        raise ValueError("invalid execution attempt")
    producers = data.get("producer_attempts")
    if producers is not None and (
        not isinstance(producers, list)
        or any(not isinstance(value, str) or not value for value in producers)
    ):
        raise ValueError("invalid execution producers")
    for key in ("sequence", "updated_at", "training_ticks", "execution_ticks"):
        if data.get(key) is not None and not number(data[key]):
            raise ValueError("invalid execution observation count")
    if not isinstance(data.get("phases"), dict):
        raise ValueError("invalid execution phases")
    for values in data["phases"].values():
        if not isinstance(values, dict) or any(
            value is not None and not number(value) for value in values.values()
        ):
            raise ValueError("invalid phase observation")
    if data.get("estimate") is not None:
        validate_estimate(data["estimate"])
    return deepcopy(data)


def validate_estimate(data):
    if not isinstance(data, dict):
        raise ValueError("invalid execution estimate")
    for key in ("seconds", "known_seconds", "conditional_seconds"):
        if data.get(key) is not None and not number(data[key]):
            raise ValueError("invalid execution estimate duration")


def observation_rows(tasks):
    """Prefer complete aggregates before their repeated child projections."""
    return sorted(
        tasks,
        key=lambda row: bool(
            isinstance(row.get("execution"), dict) and row["execution"].get("aggregate")
        ),
        reverse=True,
    )


def producer_identities(data):
    return set(data.get("producer_attempts") or [data["attempt"]])


class RollingThroughput:
    """A coordinator's monotonic window, baselined by source attempt identity."""

    def __init__(self, *, window=30, minimum=5, stale_after=15):
        self.window, self.minimum, self.stale_after = window, minimum, stale_after
        self.previous, self.seen, self.samples = {}, {}, deque()
        self.last = None
        self.received = {}
        self.payload = {
            "status": "collecting",
            "reason": "warming execution window",
            "total": None,
            "training": None,
            "window_seconds": 0,
        }

    def sample(self, tasks, now, *, wall_time=None):
        if not number(now) or self.last is not None and now <= self.last:
            return self.payload["total"]
        delta, training_delta, identities, unavailable = 0.0, 0.0, set(), []
        for row in observation_rows(tasks):
            if row.get("unit") == "entries" and row.get("execution") is None:
                continue
            data = row.get("execution")
            if data is not None:
                try:
                    data = validate_observation(data)
                except ValueError:
                    unavailable.append("invalid execution observation")
                    continue
                key = (row["id"], data["attempt"])
                # A parent aggregate and child projection of one producer count once.
                producers = producer_identities(data)
                if producers <= identities:
                    continue
                if producers & identities:
                    unavailable.append("overlapping aggregate producers")
                    continue
                identities.update(producers)
                counters = (
                    data.get("execution_ticks", 0),
                    data.get("training_ticks", 0),
                )
                sequence = data.get("sequence", 0)
                if data.get("reason"):
                    unavailable.append(data["reason"])
                before = self.previous.get(key)
                if before is not None and sequence < before[2]:
                    unavailable.append("out-of-order source observation")
                    continue
                if (
                    before is not None
                    and sequence == before[2]
                    and counters != before[:2]
                ):
                    unavailable.append("counter changed without a source event")
                    continue
            else:
                # Older sources can expose training counters; case timing remains
                # unknown until they publish the versioned observation contract.
                values = row.get("values", {})
                ticks = values.get("physical_ticks")
                if not number(ticks):
                    if row.get("status") not in FINAL | {"queued", "pending"}:
                        unavailable.append("source has no execution counters")
                    continue
                key = (
                    row["id"],
                    str(
                        row.get("run")
                        or values.get("training_run_directory")
                        or row.get("started_at", "legacy")
                    ),
                )
                counters = (ticks, ticks)
                sequence = values.get("source_updated_at", row.get("updated_at", now))
                before = self.previous.get(key)
                if row.get("status") not in FINAL:
                    unavailable.append("legacy source has training-only counters")
            if any(not number(value) for value in counters):
                unavailable.append("invalid source counter")
                continue
            fresh = before is None or sequence != before[2]
            if fresh:
                age = (
                    max(0, wall_time - data["updated_at"])
                    if data is not None
                    and wall_time is not None
                    and number(data.get("updated_at"))
                    and before is None
                    else 0
                )
                self.received[key] = now - age
            if now - self.received.get(key, now) > self.stale_after and row.get(
                "status"
            ) not in FINAL | {"queued", "pending"}:
                unavailable.append("source progress is stale")
            if before is not None:
                if counters[0] < before[0] or counters[1] < before[1]:
                    unavailable.append("counter reset; warming new attempt")
                    self.previous.pop(key, None)
                    continue
                delta += counters[0] - before[0]
                training_delta += counters[1] - before[1]
            self.previous[key] = (*counters, sequence)
            self.seen[key] = now
        if self.last is not None and now - self.last <= self.window:
            self.samples.append((self.last, now, delta, training_delta))
        elif self.last is not None:
            self.samples.clear()
            unavailable.append("sampling gap exceeds execution window")
        self.last = now
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()
        self.previous = {
            key: value
            for key, value in self.previous.items()
            if now - self.seen.get(key, now) <= self.window
        }
        span = sum(end - start for start, end, _, _ in self.samples)
        warm = span < self.minimum
        status = "collecting" if warm else "unavailable" if unavailable else "ok"
        total = (
            sum(row[2] for row in self.samples) / span
            if span and not warm and not unavailable
            else None
        )
        training = (
            sum(row[3] for row in self.samples) / span
            if span
            and not warm
            and not any(
                reason != "legacy source has training-only counters"
                for reason in unavailable
            )
            else None
        )
        self.payload = {
            "status": status,
            "reason": "; ".join(sorted(set(unavailable)))
            if unavailable
            else "warming execution window"
            if warm
            else None,
            "total": total,
            "training": training,
            "window_seconds": span,
        }
        return total

    def __getstate__(self):
        return {
            "window": self.window,
            "minimum": self.minimum,
            "stale_after": self.stale_after,
        }

    def __setstate__(self, state):
        self.__init__(**state)  # another process must start a fresh monotonic window

    @property
    def training_rate(self):
        return self.payload["training"]


def resource_fields(snapshot, *, roots=None, birth=None, wall_time=None):
    """Label host load separately from verified batch ownership; never signal."""
    if snapshot is None:
        return {
            "cpus_used": None,
            "owned_cpu_cores": None,
            "cpu_status": "unavailable",
            "cpu_reason": "resource observation unavailable",
        }
    first = "CPU load first sample unavailable" in snapshot.unavailable
    used = snapshot.observations.get(
        "used_cpu_cores", snapshot.cpus * snapshot.cpu_load / 100
    )
    owned, reason = [], None
    if roots is None:
        owned = [row for row in snapshot.processes if row.excluded]
    else:
        from smartsom.experiments import control

        birth = birth or control._birth
        identities = {}
        for root in roots:
            try:
                record = control.read(root)
                if record is None or record.get("schema") != "smartsom.run-control/v1":
                    reason = "owned process registration unavailable"
                    continue
                rows = list(record.get("members", []))
                if record.get("status") in control.ACTIVE:
                    rows.append(record["owner"])
                for row in rows:
                    measured_birth = birth(row["pid"])
                    if measured_birth is None:
                        reason = "owned birth identity unavailable"
                    if measured_birth == row["created"]:
                        identities[row["pid"]] = row["created"]
            except (OSError, ValueError, KeyError, TypeError):
                reason = "owned identity observation failed"
        owned = [row for row in snapshot.processes if row.pid in identities]
        if identities.keys() - {row.pid for row in owned}:
            reason = "owned process CPU rows unavailable"
    if any(row.cpu_cores is not None and not number(row.cpu_cores) for row in owned):
        reason = "owned process CPU observation invalid"
    if any(row.cpu_cores is None for row in owned):
        reason = "owned process first CPU sample unavailable"
    return {
        "cpus_total": snapshot.cpus,
        "cpus_used": None if first else used,
        "owned_cpu_cores": None if reason else sum(row.cpu_cores or 0 for row in owned),
        "cpu_status": "collecting" if first else "ok",
        "cpu_reason": "host CPU first sample unavailable" if first else None,
        "owned_cpu_status": "collecting" if reason else "ok",
        "owned_cpu_reason": reason,
        "cpu_observed_at": time.time() if wall_time is None else wall_time,
        "cpu_age_seconds": 0,
    }


class ExecutionTelemetry:
    """Compose producer clocks, saved observations and pure queue estimates."""

    def __init__(self):
        self.attempt = uuid4().hex
        self.native = PhaseMeter(attempt=self.attempt)
        self.native.last, self.native.phase = time.monotonic(), "loading"
        self.meters, self.observations, self.plans, self.references = {}, {}, {}, {}
        self.sequence = 0
        self.updated_at = None
        self.last_training = 0
        self.throughput = RollingThroughput()
        self.result = None
        self.batch_spec = {}
        self.received = {}

    def configure_batch(self, specs):
        self.batch_spec = {row["id"]: deepcopy(row) for row in specs}

    def observe(self, view, row, event, *, now, wall_time):
        if event.get("execution") is not None:
            try:
                data = validate_observation(event["execution"])
            except ValueError:
                row["execution_reason"] = "invalid source timing observation"
                return
            previous = self.observations.get(row["id"])
            if (
                previous
                and previous["attempt"] == data["attempt"]
                and data["sequence"] < previous["sequence"]
            ):
                return
            row["execution"] = data
            if previous == data:
                return
            self.sequence += 1
            self.updated_at = data.get("updated_at")
            self.observations[row["id"]] = data
            age = (
                max(0, wall_time - data["updated_at"])
                if previous is None and number(data.get("updated_at"))
                else 0
            )
            self.received[row["id"]] = now - age
            return
        if view.kind not in SINGLE:
            # A coordinator poll is not a producer phase boundary. Old sources
            # retain their counters but safely warm the new timing contract.
            return
        self.sequence += 1
        self.updated_at = wall_time
        if row.get("unit") == "entries":
            return
        if view.kind in SINGLE:
            values = deepcopy(row.get("values", {}))
            if row.get("unit") in {
                "physical ticks",
                "environment steps",
                "adapter decisions",
                "sampling decisions",
            }:
                self.last_training = (
                    values.get("physical_ticks", row.get("completed", 0)) or 0
                )
            values["physical_ticks"] = self.last_training
            # Rows retain display facts from earlier cases. Only a fresh case
            # event may move its counter; metadata and phase-boundary events
            # must not reuse the previous validation round's last tick.
            for prefix in ("validation", "evaluation"):
                if prefix + "_tick" not in {
                    **event,
                    **event.get("display_values", {}),
                }:
                    values.pop(prefix + "_tick", None)
            merged = {**row, "unit": "physical ticks", "values": values}
            self.native.observe(merged, now=now, wall_time=wall_time)
            row["execution"] = self.native.snapshot()
        else:
            meter = self.meters.get(row["id"])
            if meter is None:
                meter = self.meters[row["id"]] = PhaseMeter()
            meter.observe(
                row, now=now, wall_time=wall_time, attempt=event.get("display_run")
            )
            row["execution"] = meter.snapshot()
            self.observations[row["id"]] = row["execution"]

    def summarize(self, view, *, now, wall_time):
        if view.kind in SINGLE:
            data = self.native.snapshot()
            rows = list(view.tasks.values())
            training = next(
                (
                    row
                    for row in rows
                    if row.get("id") == "training"
                    or row.get("unit") == "physical ticks"
                ),
                {},
            )
            current = next(
                (
                    row
                    for row in rows
                    if row.get("stage") in {"evaluation", "validation"}
                    and row.get("status") not in FINAL
                ),
                training,
            )
            representative = {
                **training,
                "stage": current.get("stage", training.get("stage")),
                "status": view.status,
                "values": {
                    **training.get("values", {}),
                    **current.get("values", {}),
                    "physical_ticks": self.last_training,
                },
            }
            plan = view.workflow or training.get("values", {}).get("workflow", {})
            result = remaining_service(representative, plan, data)
            data.update(plan=deepcopy(plan), estimate=result)
            self.result = {"experiment": data, "tasks": {}, "batch": result}
            return self.result
        tasks, jobs = {}, []
        for row in view.tasks.values():
            if row.get("unit") == "entries" and row.get("execution") is None:
                continue
            data = row.get("execution") or self.observations.get(row["id"])
            plan = (
                (data or {}).get("plan")
                or row.get("values", {}).get("workflow")
                or view.workflow
                or {}
            )
            group = plan.get("eta_group")
            if data and group:
                phases = data.get("phases", {})
                old = self.references.setdefault(group, {})
                for name, cost in phases.items():
                    if (
                        cost.get("rate") is not None
                        or cost.get("mean_seconds") is not None
                    ):
                        old[name] = deepcopy(cost)
            service = remaining_service(
                row, plan, data, reference=self.references.get(group)
            )
            if (
                data
                and data.get("estimate", {}).get("seconds") == 0
                and row.get("status") not in FINAL
            ):
                service.update(
                    seconds=None,
                    conditional=True,
                    missing=["waiting for verified process exit"],
                )
            if (
                data
                and number(data.get("updated_at"))
                and now - self.received.get(row["id"], now) > 15
                and row.get("status") not in FINAL
            ):
                service.update(
                    seconds=None, conditional=True, missing=["source progress is stale"]
                )
            tasks[row["id"]] = service
            spec = self.batch_spec.get(row["id"], {})
            file_spec = self.batch_spec.get(
                row["id"].split("::", 1)[0].split("__", 1)[0], {}
            )
            spec = spec or file_spec
            jobs.append(
                {
                    "id": row["id"],
                    "status": row["status"],
                    **service,
                    "stage": spec.get("stage", 0),
                    "barrier_group": spec.get("barrier_group"),
                    "requires": spec.get("requires", []),
                }
            )
        by_file = {}
        for job in jobs:
            key = job["id"].split("::", 1)[0].split("__", 1)[0]
            by_file.setdefault(key, []).append(job["id"])
        for job in jobs:
            job["requires"] = [
                identity
                for required in job["requires"]
                for identity in by_file.get(required, [required])
            ]
        tuning = view.tuning or {}
        limit = (
            tuning.get("concurrency_limit")
            or (view.workflow or {}).get("max_concurrent")
            or max(
                1,
                sum(
                    row["status"] in {"running", "staged", "pending", "stopping"}
                    for row in jobs
                ),
            )
        )
        allocations = (view.tuning or {}).get("entries", [])
        resources = (view.tuning or {}).get("resources", {})
        requested = [
            entry.get("requested_cpus")
            for entry in allocations
            if number(entry.get("requested_cpus")) and entry["requested_cpus"] > 0
        ]
        peak_ram = [
            entry.get("requested_peak_ram")
            for entry in allocations
            if number(entry.get("requested_peak_ram"))
            and entry["requested_peak_ram"] > 0
        ]
        if requested and number(resources.get("cpus_available")):
            limit = min(
                limit, max(1, int(resources["cpus_available"] // max(requested)))
            )
        if peak_ram and number(resources.get("memory_available")):
            limit = min(
                limit, max(1, int(resources["memory_available"] // max(peak_ram)))
            )
        batch = batch_remaining(jobs, limit=limit)
        batch["open_queue"] = bool(tuning.get("queue_open"))
        sources, included = [], set()
        for row in observation_rows(view.tasks.values()):
            data = row.get("execution")
            if data is None:
                continue
            producers = producer_identities(data)
            if producers & included:
                continue
            sources.append(data)
            included.update(producers)
        aggregate = combined_observation(
            sources,
            attempt=self.attempt,
            sequence=self.sequence,
            updated_at=self.updated_at,
        )
        aggregate.update(aggregate=True, estimate=deepcopy(batch))
        self.result = {
            "tasks": tasks,
            "batch": batch,
            "experiment": None,
            "aggregate": aggregate,
        }
        return self.result


def estimate_text(estimate):
    if not estimate:
        return "采集中（无分阶段观测）"
    if estimate.get("seconds") is not None:
        return coarse_time(estimate["seconds"])
    if estimate.get("conditional_seconds") is not None:
        return coarse_time(estimate["conditional_seconds"]) + "（恢复/资源就绪后）"
    known = estimate.get("known_seconds", 0)
    missing = estimate.get("missing") or [estimate.get("reason") or "后续阶段采集中"]
    reason = str(missing[0])
    for raw, label in {
        "training": "训练",
        "validation": "验证",
        "evaluation": "最终评估",
        "initial": "初始对照",
        "rule": "规则对照",
        "random": "随机对照",
        "saving": "保存",
        "finalization": "收尾",
    }.items():
        reason = reason.replace(raw + ":", label + "：")
    for raw, label in {
        "collecting phase rate": "速率采集中",
        "no recent progress": "近期无推进",
        "collecting checkpoint cost": "耗时采集中",
        "collecting completion cost": "耗时采集中",
        "required phase observations unavailable": "后续阶段观测不足",
        "source progress is stale": "源进度已过期",
        "waiting for verified process exit": "等待实际进程退出",
    }.items():
        reason = reason.replace(raw, label)
    return ("已知部分" + coarse_time(known) + "；" if known else "采集中；") + reason


def batch_text(view):
    data = (view.execution or {}).get("batch")
    prefix = (
        "当前已提交任务预计剩余 "
        if (view.tuning or {}).get("queue_open")
        else "批次预计剩余 "
    )
    return prefix + estimate_text(data)


def source_execution(snapshot):
    data = snapshot.get("execution") or {}
    return data.get("experiment") or data.get("aggregate")
