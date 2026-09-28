"""Recording-derived task performance and a declared theoretical reference.

One definition for throughput, passing rate and tardiness, shared by the replay
view and by evaluation reports, so a displayed number and a reported number
cannot drift apart. Nothing here executes the engine or reads latent quality:
every value comes from recorded public state.

Each metric is available as a cumulative value through the displayed tick and as
a rate over a trailing window. Tardiness needs completion identities and demand
due ticks; when a recording stores counts only, it is reported as unavailable
rather than as zero.
"""

from bisect import bisect_right


def qualified_count(state):
    """Qualified demands through this tick, by identity when it is recorded."""
    completed = state.get("completed")
    if completed is not None:
        return len(completed)
    metrics = state.get("metrics", {})
    for key in ("fulfilled", "passed"):
        if key in metrics:
            return int(metrics[key])
    return 0


def submitted_count(state):
    """Output submissions, or None when the recording does not store them."""
    metrics = state.get("metrics", {})
    if "submitted" in metrics:
        return int(metrics["submitted"])
    return 0 if "completed" in state else None


def lateness(completed_at, due_at):
    """Ticks past the due tick; the one definition of lateness in this project."""
    return max(0, int(completed_at) - int(due_at))


def tardiness_totals(pairs):
    """Total, tardy count and mean over (completion tick, due tick) pairs."""
    values = [lateness(when, due) for when, due in pairs]
    return {
        "total_tardiness": sum(values),
        "tardy_jobs": sum(1 for value in values if value > 0),
        "mean_tardiness": (sum(values) / len(values)) if values else None,
    }


def due_ticks(recording):
    """demand_id -> due tick, from the frozen scenario of a run manifest."""
    demands = (
        getattr(recording, "manifest", {})
        .get("inputs", {})
        .get("scenario", {})
        .get("demands", [])
    )
    return {d["demand_id"]: d["due_at"] for d in demands if "due_at" in d}


class TaskPerformance:
    """Cumulative series and trailing-window rates over one recorded run."""

    def __init__(self, recording=None, due=None):
        if due is None:
            due = due_ticks(recording) if recording is not None else {}
        self.due = due
        self.qualified = []
        self.submitted = []
        self.tardiness = []
        self.tardy = []
        self.completions = []
        self.identified = True
        self._done = set()
        if recording is not None:
            for tick in range(recording.last_tick + 1):
                self.observe(tick, recording.row(tick)["state"])

    @property
    def last_tick(self):
        return len(self.qualified) - 1

    def observe(self, tick, state):
        """Add one recorded frame; callers pass consecutive ticks from zero."""
        if tick != len(self.qualified):
            raise ValueError("task performance requires consecutive ticks from zero")
        total = self.tardiness[-1] if self.tardiness else 0
        late = self.tardy[-1] if self.tardy else 0
        completed = state.get("completed")
        if completed is None:
            self.identified = False
        else:
            for demand in sorted(set(completed) - self._done):
                lateness = self.lateness(demand, tick)
                if lateness is None:
                    self.identified = False
                    continue
                self.completions.append((tick, demand, lateness))
                total += lateness
                late += int(lateness > 0)
            self._done = set(completed)
        self.qualified.append(qualified_count(state))
        self.submitted.append(submitted_count(state))
        self.tardiness.append(total)
        self.tardy.append(late)

    def lateness(self, demand, tick):
        """Ticks past the due tick, or None when the due tick is unknown."""
        due = self.due.get(demand)
        return None if due is None else lateness(tick, due)

    def _span(self, tick, window):
        start = max(0, tick - window)
        return start, tick - start

    def cumulative(self, tick):
        """Totals and averages over the whole run through this tick."""
        delivered = self.qualified[tick]
        submitted = self.submitted[tick]
        return {
            "qualified": delivered,
            "submitted": submitted,
            "throughput": delivered / tick if tick else None,
            "passing_rate": delivered / submitted if submitted else None,
            "total_tardiness": self.tardiness[tick] if self.identified else None,
            "tardy_jobs": self.tardy[tick] if self.identified else None,
            "mean_tardiness": (
                self.tardiness[tick] / delivered
                if self.identified and delivered
                else None
            ),
        }

    def window(self, tick, window=100):
        """The same quantities over the trailing window ending at this tick."""
        start, span = self._span(tick, window)
        if not span:
            return {
                "span": 0,
                "throughput": None,
                "passing_rate": None,
                "tardiness": None,
                "deliveries": 0,
            }
        delivered = self.qualified[tick] - self.qualified[start]
        submitted = (
            self.submitted[tick] - self.submitted[start]
            if self.submitted[tick] is not None and self.submitted[start] is not None
            else None
        )
        return {
            "span": span,
            "deliveries": delivered,
            "throughput": delivered / span,
            "passing_rate": delivered / submitted if submitted else None,
            "tardiness": (
                self.tardiness[tick] - self.tardiness[start]
                if self.identified
                else None
            ),
        }

    def completions_until(self, tick):
        index = bisect_right(self.completions, tick, key=lambda row: row[0])
        return self.completions[:index]

    def summary(self, tick=None, window=100):
        """Both views at one tick; the window is named so a rate stays readable."""
        tick = self.last_tick if tick is None else tick
        return {
            "tick": tick,
            "window": window,
            "cumulative": self.cumulative(tick),
            "recent": self.window(tick, window),
            "tardiness_identified": self.identified,
        }


def from_run(source):
    """Task performance for a finished run, or None when it stored no trajectory."""
    from smartsom.trace.production import Playback

    try:
        return TaskPerformance(Playback(source))
    except (OSError, ValueError, KeyError):
        return None


def theoretical_reference(scenario):
    """Declared upper reference for a scenario, ignoring the stated effects.

    This is a bound computed from work content and machine capability, not a
    prediction and not a target. It deliberately ignores transport, buffer
    blocking, inspection, disposal, outages, processing disturbance and
    replacement attempts, so no controller is expected to reach it. The
    assumptions travel with the numbers so a report cannot quote them bare.
    """
    machines = {}
    for machine in scenario.factory.machines:
        for operation in machine.operation_types:
            machines[operation] = machines.get(operation, 0) + 1
    work, routes = {}, []
    for demand in scenario.demands:
        route = 0
        for step in demand.steps:
            work[step.operation_type] = work.get(step.operation_type, 0) + (
                step.nominal_ticks
            )
            route += step.nominal_ticks
        routes.append(route)
    missing = sorted(set(work) - set(machines))
    if missing:
        return {
            "available": False,
            "reason": f"no machine declares operation types: {', '.join(missing)}",
        }
    route_bound = max(routes, default=0)
    load_bound = max(
        (ticks / machines[operation] for operation, ticks in work.items()), default=0
    )
    rate = min(
        (
            machines[operation] * len(scenario.demands) / ticks
            for operation, ticks in work.items()
        ),
        default=None,
    )
    horizon = int(getattr(scenario, "tick_limit", 0) or 0)
    reachable = None
    if rate is not None and horizon:
        reachable = min(
            len(scenario.demands), int(max(0, horizon - route_bound) * rate) + 1
        )
    return {
        "available": True,
        "demands": len(scenario.demands),
        "makespan_lower_bound": max(route_bound, load_bound),
        "longest_route_ticks": route_bound,
        "busiest_operation_ticks": load_bound,
        "max_throughput_jobs_per_tick": rate,
        "max_qualified_in_horizon": reachable,
        "horizon": horizon or None,
        "assumptions": [
            "nominal processing at the default speed mode",
            "no transport, buffer blocking, inspection or disposal time",
            "no outages and no processing disturbance",
            "every demand qualifies on its first attempt",
            "machines are never idle while eligible work exists",
        ],
    }
