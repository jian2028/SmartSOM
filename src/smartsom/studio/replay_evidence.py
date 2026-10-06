"""Compact recording-derived series. No simulator execution or hidden quality."""

from bisect import bisect_right

from smartsom.trace.performance import TaskPerformance, qualified_count, shipment_count


def qualified(state):
    return int(state["metrics"].get("fulfilled", len(state.get("completed", ()))))


def submitted(state):
    # Current metrics are a sparse Counter; historical absent fields are unknown.
    return state["metrics"].get("submitted", 0 if "completed" in state else None)


def outside_count(state):
    return (
        len(state["queue"])
        if "queue" in state
        else state["metrics"].get("external_backlog")
    )


def short_job(identifier):
    if "/attempt/" in identifier:
        demand, attempt = identifier.rsplit("/attempt/", 1)
        return f"{demand.rsplit('_', 1)[-1].lstrip('0') or '0'}:{attempt}"
    if "_a" in identifier:
        demand, attempt = identifier.rsplit("_a", 1)
        return f"{demand.removeprefix('d').lstrip('0') or '0'}:{attempt}"
    return identifier


def job_marks(identifier):
    """Order number and one-based attempt; historical attempts start at zero."""
    label = short_job(identifier)
    if "/attempt/" in identifier or "_a" in identifier:
        order, attempt = label.rsplit(":", 1)
        if attempt.isdigit():
            return order, int(attempt) + int(
                "_a" in identifier and "/attempt/" not in identifier
            )
    return label, 1


def movement_conflicts(row):
    historical = row.get("historical_frame")
    if historical is not None:
        return {
            key
            for key, value in historical.get("agvs", {}).items()
            if value.get("previous_outcome") == "CONFLICT"
            and value.get("previous_action")
            in (0, 1, 2, 3, "UP", "DOWN", "LEFT", "RIGHT")
        }
    actions = dict(
        row.get("actions", {}).get("movers", row.get("actions", {}).get("agvs", ()))
    )
    return {
        key.removeprefix("agv:")
        for key, reason in row.get("rejections", {}).items()
        if reason == "conflict"
        and key.startswith("agv:")
        and actions.get(key.removeprefix("agv:")) in ("UP", "DOWN", "LEFT", "RIGHT")
    }


def resource_conflicts(row):
    historical = row.get("historical_frame")
    if historical is not None:
        return {
            key
            for key, value in historical.get("agvs", {}).items()
            if value.get("previous_outcome") == "CONFLICT"
            and value.get("previous_action") == 4
        }
    actions = dict(
        row.get("actions", {}).get("movers", row.get("actions", {}).get("agvs", ()))
    )
    return {
        key.removeprefix("agv:")
        for key, reason in row.get("rejections", {}).items()
        if reason == "conflict"
        and key.startswith("agv:")
        and actions.get(key.removeprefix("agv:")) == "INTERACT"
    }


def conflict_agvs(row):
    """Rejected AGVs, or nothing when the frame records no movement outcomes."""
    historical = row.get("historical_frame")
    if historical is not None and "agvs" not in historical:
        return set()
    return movement_conflicts(row) | resource_conflicts(row)


def displayed_quality(state, job_id, owner):
    quality = state.get("jobs", {}).get(job_id, {}).get("quality", "UNKNOWN")
    if (
        state.get("machines", {}).get(owner, {}).get("status") == "PROCESSING"
        and quality != "FAIL"
    ):
        return "UNKNOWN"
    return quality


def scenario_reference(recording):
    """Declared bound for a recorded run, or None when its scenario is unavailable."""
    snapshot = getattr(recording, "manifest", {}).get("inputs", {}).get("scenario")
    if not snapshot:
        return None
    from smartsom.config.production import scenario_from_snapshot
    from smartsom.trace.performance import theoretical_reference

    try:
        return theoretical_reference(scenario_from_snapshot(snapshot))
    except (KeyError, TypeError, ValueError):
        return None


def _quantity(value, suffix="", digits=3):
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def performance_rows(performance, tick, window, reference=None):
    """Task-performance display rows: whole run, trailing window, declared bound.

    Text only, so the panel renders what evaluation reports. Missing evidence
    prints as an em dash and never as zero.
    """
    totals = performance.cumulative(tick)
    recent = performance.window(tick, window)
    bound = reference if reference and reference.get("available") else {}
    late, tardy = totals["total_tardiness"], totals["tardy_jobs"]
    late_text = "—" if late is None else f"{late} ticks"
    if late is not None and totals.get("tardiness_censored"):
        late_text += (
            " delivered (censored); fixed-job total unavailable; "
            f"lower bound {totals['total_tardiness_lower_bound']} ticks"
        )
    return [
        {
            "label": "Shipments" if totals["qualified"] is None else "Qualified jobs",
            "total": str(totals["shipped"])
            if totals["qualified"] is None
            else str(totals["qualified"]),
            "recent": "—"
            if recent["deliveries"] is None
            else str(recent["deliveries"]),
            "bound": (
                str(bound["max_qualified_in_horizon"])
                if bound.get("max_qualified_in_horizon") is not None
                else "—"
            ),
        },
        {
            "label": "Throughput",
            "total": _quantity(totals["throughput"]),
            "recent": _quantity(recent["throughput"]),
            "bound": _quantity(bound.get("max_throughput_jobs_per_tick")),
        },
        {
            "label": "Passing rate",
            "total": "—"
            if totals["passing_rate"] is None
            else f"{totals['passing_rate']:.1%}",
            "recent": "—"
            if recent["passing_rate"] is None
            else f"{recent['passing_rate']:.1%}",
            "bound": "—",
        },
        {
            "label": "Total tardiness",
            "total": late_text,
            "recent": "—" if recent["tardiness"] is None else f"+{recent['tardiness']}",
            "bound": "0" if bound else "—",
        },
        {
            "label": "Tardy jobs",
            "total": "—" if tardy is None else str(tardy),
            "recent": "—"
            if recent["tardy_jobs"] is None
            else str(recent["tardy_jobs"]),
            "bound": "0" if bound else "—",
        },
    ]


MARKER_CATEGORIES = (
    ("conflict", "AGV conflicts"),
    ("delivery", "Deliveries"),
    ("scrap", "Scrapped jobs"),
    ("inspection", "Inspection results"),
    ("any", "Any recorded event"),
)
INSPECTION_KINDS = ("inspection_result", "inspection_completed", "quality_revealed")


class ReplayEvidence:
    """One compact sample per integer frame; lookahead is explicitly recorded."""

    def __init__(self, recording, factory=None):
        self.deliveries = []
        self.passing_rates = []
        self.recorded_events = []
        self.arrivals = []
        self.arrival_ticks = []
        self.station_busy = []
        self.inspected = []
        self.disposals = []
        self.outputs = []
        self.output_attempts = []
        self.mode_counts = []
        factory = factory or getattr(recording, "factory", None)
        output_ids = (
            {b.buffer_id for b in factory.buffers if b.role == "system_output"}
            if factory
            else set()
        )
        self.input_ids = (
            [b.buffer_id for b in factory.buffers if b.role == "system_input"]
            if factory
            else []
        )
        demands = (
            getattr(recording, "manifest", {})
            .get("inputs", {})
            .get("scenario", {})
            .get("demands", [])
        )
        self.demand_inputs = {
            d["demand_id"]: d.get("input_id")
            or (self.input_ids[0] if self.input_ids else None)
            for d in demands
        }
        self.input_arrivals = {key: [] for key in self.input_ids}
        # Task performance shares this single pass so the panel and evaluation
        # reports cannot drift apart.
        self.performance = TaskPerformance()
        self.performance.due = {
            d["demand_id"]: d["due_at"] for d in demands if "due_at" in d
        }
        self.conflict_ticks = []
        attempts, modes = {}, {}
        self.has_output_events = False
        self.blind_output = False
        busy, inspected, disposals, outputs = {}, {}, {}, {}
        previous_released = None
        previous_state = {}
        for tick in range(recording.last_tick + 1):
            row = getattr(recording, "summary_row", recording.row)(tick)
            state = row["state"]
            self.blind_output = state.get("output_semantics") == "blind-shipment/v1"
            self.performance.observe(tick, state)
            if conflict_agvs(row):
                self.conflict_ticks.append(tick)
            self.deliveries.append(shipment_count(state))
            denominator = submitted(state)
            self.passing_rates.append(
                qualified_count(state) / denominator
                if denominator and qualified_count(state) is not None
                else None
            )
            for event in row.get("events", ()):
                # Public identities only: event payloads can contain latent quality.
                identities = [
                    str(event[key])
                    for key in (
                        "machine",
                        "machine_id",
                        "station",
                        "station_id",
                        "agv",
                        "agv_id",
                        "job",
                        "job_id",
                        "owner",
                        "demand",
                    )
                    if key in event
                ]
                self.recorded_events.append(
                    (tick, event["kind"], " · ".join(identities))
                )
            if tick == 0:
                self.has_output_events = "historical_frame" not in row or bool(
                    getattr(recording, "events", None)
                )
            released = state["metrics"].get("released")
            if "released" in state:
                released = len(state["released"])
            if (
                previous_released is not None
                and released is not None
                and released > previous_released
            ):
                self.arrivals.append((tick, int(released - previous_released)))
                self.arrival_ticks.append(tick)
            if tick and self.input_ids:
                if len(self.input_ids) == 1:
                    if self.arrivals and self.arrivals[-1][0] == tick:
                        self.input_arrivals[self.input_ids[0]].append(self.arrivals[-1])
                else:
                    counts = {}
                    for demand in set(state.get("released", ())) - set(
                        previous_state.get("released", ())
                    ):
                        owner = self.demand_inputs.get(demand)
                        if owner in self.input_arrivals:
                            counts[owner] = counts.get(owner, 0) + 1
                    for owner, count in counts.items():
                        self.input_arrivals[owner].append((tick, count))
            previous_released = released
            events = row.get("events", ())
            starts = {
                e.get("station", e.get("station_id"))
                for e in events
                if e["kind"] in ("inspection_started", "inspection_start")
            }
            for key in state["stations"]:
                was_busy = bool(
                    previous_state.get("stations", {}).get(key, {}).get("batch")
                )
                busy[key] = busy.get(key, 0) + int(
                    tick > 0
                    and (
                        was_busy
                        or key in starts
                        and "jobs" not in state["stations"][key]
                    )
                )
            drops = {
                e.get("job", e.get("job_id")): e.get("owner")
                for e in events
                if e["kind"] == "drop"
            }
            for event in events:
                kind = event["kind"]
                if kind == "inspection_completed":
                    key = event.get("station", event.get("station_id"))
                    inspected[key] = inspected.get(key, 0) + len(event.get("jobs", ()))
                if kind == "inspection_result":
                    jid = event.get("job_id")
                    key = previous_state.get("jobs", {}).get(jid, {}).get("location")
                    if key in state["stations"]:
                        inspected[key] = inspected.get(key, 0) + 1
                if kind == "trash":
                    key = drops.get(event.get("job_id"))
                    if key:
                        disposals[key] = disposals.get(key, 0) + 1
                if kind == "output":
                    key = drops.get(event.get("job_id"))
                    if key:
                        attempts[key] = attempts.get(key, 0) + 1
                        outputs[key] = outputs.get(key, 0) + int(
                            event.get("quality") == "PASS"
                        )
                if (
                    kind == "drop"
                    and "historical_frame" not in row
                    and event.get("owner") in output_ids
                ):
                    key = event["owner"]
                    attempts[key] = attempts.get(key, 0) + 1
                    outputs[key] = outputs.get(key, 0) + int(
                        self.blind_output
                        or state["jobs"].get(event.get("job"), {}).get("quality")
                        == "PASS"
                    )
                if kind in ("processing_started", "process_start"):
                    key = event.get("machine", event.get("machine_id"))
                    mode = event.get("mode")
                    if isinstance(mode, int):
                        names = getattr(recording, "mode_names", [])
                        mode = names[mode] if 0 <= mode < len(names) else f"mode_{mode}"
                    mode = str(mode) if mode is not None else "unavailable"
                    counts = modes.setdefault(key, {})
                    counts[mode] = counts.get(mode, 0) + 1
            self.output_attempts.append(dict(attempts))
            self.mode_counts.append(
                {key: dict(counts) for key, counts in modes.items()}
            )
            self.station_busy.append(dict(busy))
            self.inspected.append(dict(inspected))
            self.disposals.append(dict(disposals))
            self.outputs.append(dict(outputs))
            previous_state = state

    def marker_ticks(self, category):
        """Sorted ticks a viewer can jump between, for one marker category."""
        if category == "conflict":
            return list(self.conflict_ticks)
        if category == "delivery":
            return [tick for tick, _demand, _late in self.performance.completions]
        if category == "scrap":
            return sorted({t for t, kind, _ in self.recorded_events if kind == "trash"})
        if category == "inspection":
            return sorted(
                {t for t, kind, _ in self.recorded_events if kind in INSPECTION_KINDS}
            )
        return sorted({tick for tick, _kind, _ in self.recorded_events})

    def input_waiting(self, owner, state):
        """Outside jobs targeting this input; historical ownership may be unknown."""
        if len(self.input_ids) == 1:
            return outside_count(state)
        if "queue" not in state:
            return None
        targets = [
            self.demand_inputs.get(state["jobs"][jid].get("demand"))
            for jid in state["queue"]
        ]
        return targets.count(owner) if all(targets) else None

    def input_progress(self, owner, tick):
        """(interval ticks, remaining ticks, batch size), from recorded releases only."""
        arrivals = self.input_arrivals.get(owner, [])
        index = bisect_right(arrivals, (tick, float("inf")))
        if index == len(arrivals):
            return None
        end, count = arrivals[index]
        start = arrivals[index - 1][0] if index else 0
        return end - start, end - tick, count

    def next_arrival(self, tick):
        index = bisect_right(self.arrival_ticks, tick)
        return self.arrivals[index] if index < len(self.arrivals) else None

    def throughput(self, tick, window=100):
        if tick == 0:
            return None
        start = max(0, tick - window)
        return (self.deliveries[tick] - self.deliveries[start]) / (tick - start)

    def output_metrics(self, owner, tick, window=100):
        """Per-exit qualified total, submission pass rate, and rolling throughput."""
        if not self.has_output_events:
            return None, None, None
        passed = self.outputs[tick].get(owner, 0)
        attempts = self.output_attempts[tick].get(owner, 0)
        start = max(0, tick - window)
        throughput = (
            (passed - self.outputs[start].get(owner, 0)) / (tick - start)
            if tick
            else None
        )
        return (
            passed,
            (passed / attempts if attempts and not self.blind_output else None),
            throughput,
        )
