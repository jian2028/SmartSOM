"""Read-only budget accounting and wall-clock estimates for a frozen study."""

import json
import math
import time
from collections import deque
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02}m" if hours else f"{minutes}m {seconds:02}s"


def case_work(values, prefix, horizon):
    """Credit ended cases once, and only the active case's partial horizon."""
    ended = values.get(prefix + "_finished", 0) or 0
    requested = values.get(prefix + "_requested", 0) or 0
    fraction = 0
    if ended < requested and values.get(prefix + "_case_active", True):
        tick, limit = (
            values.get(prefix + "_tick", 0),
            values.get(prefix + "_tick_limit"),
        )
        if limit:
            fraction = min(1, max(0, (tick or 0) / limit))
    return min(requested, ended + fraction) * horizon


class StudyWork:
    """Account train/validation/test/control budgets without reading model state.

    A finished case resolves its whole planned horizon, even if it ends early.
    Shared rule/random controls are counted once per H/V/transport condition.
    This is planned-work progress, not a count of actual simulator ticks.
    """

    def __init__(self, directory, plan):
        self.directory, self.plan = Path(directory), plan
        recipe = plan["recipe"]
        self.training = recipe["total_ticks"]
        self.horizon = recipe.get("max_ticks", 4096)
        updates = math.ceil(self.training / recipe.get("ticks_per_update", 256))
        self.validation_cases = (
            updates // recipe.get("validation_every_updates", 4)
        ) * recipe.get("validation_cases", 5)
        self.evaluation_cases = recipe.get("evaluation_cases", 5)
        self.controls = recipe.get("controls", [])
        self.entry_total = self.training + self.horizon * (
            self.validation_cases
            + self.evaluation_cases * (1 + int("initial" in self.controls))
        )
        conditions = {
            (e["H_case"], e["V_case"], e["transport"]) for e in plan["entries"]
        }
        self.shared_total = (
            len(conditions)
            * self.horizon
            * self.evaluation_cases
            * sum(k in self.controls for k in ("rule", "random"))
        )
        self.total = len(plan["entries"]) * self.entry_total + self.shared_total
        self.started = time.monotonic()
        self.samples = deque()
        self.cache = {}

    def ended_cases(self, path, *, manifest=False):
        """Cache only complete, successfully parsed atomic result files."""
        try:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            cached = self.cache.get(path)
            if cached is None or cached[0] != stamp:
                doc = json.loads(path.read_text())
                cases = doc.get("results", []) if manifest else doc
                if not isinstance(cases, list):
                    raise ValueError("invalid case results")
                self.cache[path] = (stamp, len(cases))
            return self.cache[path][1]
        except (OSError, ValueError, TypeError):
            return 0

    def overview(self, tasks, state, *, now=None):
        now = time.monotonic() if now is None else now
        elapsed = max(0, now - self.started)
        completed = training = 0
        shared = {}
        for spec in self.plan["entries"]:
            row = tasks.get(spec["id"], {})
            values = row.get("values", {})
            entry = state["entries"].get(spec["id"], {})
            ticks = min(self.training, values.get("physical_ticks", 0) or 0)
            source = values.get("training_run_directory") or entry.get("run_dir")
            paths = (
                list((Path(source) / "logs").glob("validation-*.json"))
                if source
                else []
            )
            validation = sum(self.ended_cases(p) for p in paths) * self.horizon
            current = f"validation-{values.get('updates', 0):06d}.json"
            if row.get("stage") == "validation" and not any(
                p.name == current for p in paths
            ):
                validation += case_work(values, "validation", self.horizon)
            tested = 0
            if entry.get("evaluation_dir"):
                tested = (
                    self.ended_cases(
                        Path(entry["evaluation_dir"]) / "run.json", manifest=True
                    )
                    * self.horizon
                )
            initial_path = self.directory / "controls" / (spec["id"] + "_initial.json")
            initial = (
                self.ended_cases(initial_path) * self.horizon
                if "initial" in self.controls
                else 0
            )
            if row.get("stage") == "evaluation":
                kind = values.get("evaluation_kind", "evaluation")
                partial = case_work(values, "evaluation", self.horizon)
                if kind == "evaluation":
                    tested = max(tested, partial)
                elif kind == "initial control":
                    initial = max(initial, partial)
                elif kind in {"rule control", "random control"}:
                    control = kind.split()[0]
                    key = f"{spec['H_case']}_{spec['V_case']}_{spec['transport']}_{control}"
                    shared[key] = max(shared.get(key, 0), partial)
            for kind in ("rule", "random"):
                if kind in self.controls:
                    key = (
                        f"{spec['H_case']}_{spec['V_case']}_{spec['transport']}_{kind}"
                    )
                    resolved = (
                        self.ended_cases(self.directory / "controls" / (key + ".json"))
                        * self.horizon
                    )
                    shared[key] = max(shared.get(key, 0), resolved)
            work = (
                ticks
                + min(self.validation_cases * self.horizon, validation)
                + min(self.evaluation_cases * self.horizon, tested)
                + min(self.evaluation_cases * self.horizon, initial)
            )
            values["planned_work_completed"] = work
            values["planned_work_total"] = self.entry_total
            if entry.get("status") == "completed":
                ticks, work = self.training, self.entry_total
                values["planned_work_completed"] = work
            training += ticks
            completed += min(self.entry_total, work)
        completed = min(
            self.total, completed + min(self.shared_total, sum(shared.values()))
        )
        if not self.samples or now - self.samples[-1][0] >= 5:
            self.samples.append((now, completed))
        while len(self.samples) > 2 and self.samples[1][0] < now - 300:
            self.samples.popleft()
        eta = None
        if completed >= self.total:
            eta = 0
        elif elapsed >= 30 and self.samples:
            first_time, first_work = self.samples[0]
            period = now - first_time
            rate = (completed - first_work) / period if period > 0 else 0
            if rate > 0:
                eta = (self.total - completed) / rate
        return {
            "work_completed": completed,
            "work_total": self.total,
            "training_completed": training,
            "training_total": self.training * len(self.plan["entries"]),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "eta_basis": "budget-weighted train/validation/test/control horizons; approximate 5-minute wall-clock rate",
        }
