"""Explicit, versioned comparison protocols; no implicit success-only filtering."""

import math
from dataclasses import asdict, dataclass

from smartsom.telemetry.research_metrics import SCHEMA, fingerprint


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class LearningTarget:
    """A caller-frozen target, not a claim that a policy has mathematically converged."""

    version: str
    reference: str
    maximum_cost: float
    minimum_passing: float
    consecutive: int
    budget_unit: str

    def __post_init__(self):
        if not self.version or not self.reference:
            raise ValueError(
                "target requires version and independent reference identity"
            )
        if (
            not _finite(self.maximum_cost)
            or self.maximum_cost < 0
            or not _finite(self.minimum_passing)
            or not 0 <= self.minimum_passing <= 1
        ):
            raise ValueError("invalid cost or quality target")
        if type(self.consecutive) is not int or self.consecutive < 1:
            raise ValueError("consecutive must be a positive integer")
        if self.budget_unit not in ("physical_ticks", "episodes", "training_seconds"):
            raise ValueError("explicit supported training budget unit required")

    @property
    def sha256(self):
        return fingerprint(asdict(self))


def time_to_target(points, target):
    """Per training seed/decoder fixed validation panel; gaps break persistence.

    Points are supplied in budget order, not sorted to hide malformed histories.
    Budgets refer to training, never evaluation ticks. Duplicated model labels do
    not provide independent persistence evidence. Confirmation uses the last point.
    """
    if not points:
        return {
            "status": "not_recorded",
            "protocol_sha256": target.sha256,
            "budget_unit": target.budget_unit,
        }
    previous = -1
    seen = set()
    streak = []
    lower = None
    identity = None
    evidence_gaps = False
    for point in points:
        budget = point["budget"]
        if not _finite(budget) or budget < 0 or budget <= previous:
            raise ValueError(
                "budgets must be finite, nonnegative and strictly increasing"
            )
        previous = budget
        current_identity = tuple(
            point[k] for k in ("train_seed", "decoder", "panel_sha256")
        )
        if any(v is None or v == "" for v in current_identity):
            raise ValueError("missing learning panel identity")
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError("cannot combine seeds, decoders or validation panels")
        model = point["model_sha256"]
        if not model:
            raise ValueError("missing checkpoint identity")
        cost, passing = point.get("cost"), point.get("passing_rate")
        drained = point.get("all_drained")
        if cost is not None and (not _finite(cost) or cost < 0):
            raise ValueError("invalid cost observation")
        if passing is not None and (not _finite(passing) or not 0 <= passing <= 1):
            raise ValueError("invalid passing observation")
        if drained is not None and type(drained) is not bool:
            raise ValueError("invalid drain observation")
        failed = (
            drained is False
            or (cost is not None and cost > target.maximum_cost)
            or (passing is not None and passing < target.minimum_passing)
        )
        missing = drained is None or cost is None or passing is None
        if failed or missing:
            streak = []
            seen.clear()
            # Unknown evidence breaks persistence but cannot establish a failure.
            if failed:
                lower = budget
            evidence_gaps |= missing
            continue
        if model in seen:
            continue
        seen.add(model)
        streak.append(budget)
        if len(streak) == target.consecutive:
            return {
                "status": "reached",
                "panel_identity": dict(
                    zip(
                        ("train_seed", "decoder", "panel_sha256"), identity, strict=True
                    )
                ),
                "budget_unit": target.budget_unit,
                "first_qualifying_interval": [lower, streak[0]],
                "first_qualifying_budget": streak[0],
                "evidence_gaps": evidence_gaps,
                "confirmation_budget": budget,
                "protocol_sha256": target.sha256,
            }
    return {
        "status": "censored",
        "panel_identity": dict(
            zip(("train_seed", "decoder", "panel_sha256"), identity, strict=True)
        ),
        "budget_unit": target.budget_unit,
        "observed_budget": previous,
        "pending_confirmation": bool(streak),
        "evidence_gaps": evidence_gaps,
        "protocol_sha256": target.sha256,
    }


def acceleration(base, sl):
    """No finite ratio for censored runs or zero-cost initial attainment."""
    if base.get("protocol_sha256") != sl.get("protocol_sha256") or base.get(
        "budget_unit"
    ) != sl.get("budget_unit"):
        raise ValueError("learning protocol and units must match")
    identities = [r.get("panel_identity") for r in (base, sl)]
    if all(identities) and identities[0] != identities[1]:
        raise ValueError("learning comparisons require paired seed, decoder and panel")
    if any(r.get("status") == "not_recorded" for r in (base, sl)):
        return {"status": "not_recorded", "ratio": None}
    if not all(identities):
        raise ValueError("learning comparisons require paired seed, decoder and panel")
    if base.get("status") != "reached" or sl.get("status") != "reached":
        return {"status": "censored", "ratio": None}
    a, b = base["first_qualifying_budget"], sl["first_qualifying_budget"]
    if min(a, b) == 0:
        return {
            "status": "not_applicable",
            "ratio": None,
            "reason": "initial_attainment",
        }
    return {"status": "observed", "ratio": a / b, "budget_unit": base["budget_unit"]}


def _detail(row):
    detail = row.get("research_metrics", {})
    if detail.get("schema") != SCHEMA or detail.get("level") != "detailed":
        return None
    return detail


def fixed_window(row, start, end):
    """Exact (start,end] counts from recorded order timestamps, never interpolated."""
    if not _finite(start) or not _finite(end) or not 0 <= start < end:
        raise ValueError("invalid physical window")
    detail = _detail(row)
    if detail is None:
        return {"status": "not_recorded"}
    drained = (
        row.get("completed") == row.get("total_orders")
        and row.get("total_orders", 0) > 0
        and not row.get("wall_clock_fragment", False)
    )
    if start < detail["start_tick"] or (end > detail["end_tick"] and not drained):
        return {"status": "censored", "reason": "window_outside_recording"}
    orders = detail["orders"]
    completed = [
        o
        for o in orders
        if o["completion_tick"] is not None and start < o["completion_tick"] <= end
    ]
    return {
        "status": "observed",
        "start_tick": start,
        "end_tick": end,
        "completed": len(completed),
        "qualified": sum(o["final_pass"] is True for o in completed),
        "throughput": len(completed) / (end - start),
        "qualified_throughput": sum(o["final_pass"] is True for o in completed)
        / (end - start),
    }


def _paired(nominal, perturbed):
    a, b = _detail(nominal), _detail(perturbed)
    if a is None or b is None:
        return {"status": "not_recorded"}
    if a["fingerprints"]["exogenous"] != b["fingerprints"]["exogenous"]:
        raise ValueError("external orders, physics or random realization differ")
    keys = (
        "checkpoint_sha256",
        "method",
        "learning",
        "selection",
        "policy_seed",
        "source_sha256",
        "runtime_sha256",
        "algorithm_sha256",
    )
    ia, ib = (
        nominal.get("evaluation_identity", {}),
        perturbed.get("evaluation_identity", {}),
    )
    if any(k not in ia or k not in ib for k in keys):
        return {"status": "not_recorded", "reason": "evaluation_identity_missing"}
    if any(ia[k] != ib[k] for k in keys):
        raise ValueError(
            "paired perturbations require the same policy and action sampling domain"
        )
    if ia["learning"] and ia["checkpoint_sha256"] is None:
        return {"status": "not_recorded", "reason": "model_identity_missing"}
    return None


def robustness(nominal, perturbed, start, end):
    missing = _paired(nominal, perturbed)
    if missing:
        return missing
    a, b = fixed_window(nominal, start, end), fixed_window(perturbed, start, end)
    if a["status"] != "observed" or b["status"] != "observed":
        return {"status": "censored", "nominal": a, "perturbed": b}
    return {
        "status": "observed" if a["completed"] else "not_applicable",
        "reason": None if a["completed"] else "zero_reference",
        "nominal": a,
        "perturbed": b,
        "throughput_ratio": b["completed"] / a["completed"] if a["completed"] else None,
        "qualified_throughput_ratio": b["qualified"] / a["qualified"]
        if a["qualified"]
        else None,
    }


@dataclass(frozen=True)
class RecoveryProtocol:
    version: str
    reference: str
    window_ticks: int
    consecutive_windows: int
    fraction: float = 0.95

    def __post_init__(self):
        if not self.version or not self.reference:
            raise ValueError("recovery protocol requires version and reference")
        if any(
            type(v) is not int or v <= 0
            for v in (self.window_ticks, self.consecutive_windows)
        ):
            raise ValueError("positive integer window and persistence required")
        if not _finite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("invalid recovery fraction")

    @property
    def sha256(self):
        return fingerprint(asdict(self))


def recovery(nominal, perturbed, *, machine_id, fault_tick, repair_tick, protocol):
    """Repair-anchored, nonoverlapping windows; confirm, do not backdate recovery.

    Caller selects a preregistered event. Repeated/overlapping disturbances censor;
    zero reference or exhausted nominal demand never produces spurious recovery.
    """
    if (
        any(type(v) is not int for v in (fault_tick, repair_tick))
        or not 0 <= fault_tick < repair_tick
    ):
        raise ValueError("invalid fault/repair time")
    missing = _paired(nominal, perturbed)
    if missing:
        return missing
    a, b = _detail(nominal), _detail(perturbed)
    result = {
        "protocol_sha256": protocol.sha256,
        "machine_id": machine_id,
        "fault_tick": fault_tick,
        "repair_tick": repair_tick,
        "status": "censored",
        "adaptability": None,
    }
    # Require actual disturbance evidence; avoid treating an ordinary window as recovery.
    events = b["events"]
    if not any(
        e["kind"] == "breakdown"
        and e["tick"] == fault_tick
        and e.get("machine") == machine_id
        for e in events
    ) or not any(
        e["kind"] == "repair"
        and e["tick"] == repair_tick
        and e.get("machine") == machine_id
        for e in events
    ):
        return {**result, "status": "not_recorded", "reason": "event_boundary_missing"}
    if "outage_realization" not in b:
        return {
            **result,
            "status": "not_recorded",
            "reason": "outage_realization_missing",
        }
    for outage in b["outage_realization"]:
        selected = (outage["machine_id"], outage["start"], outage["end"]) == (
            machine_id,
            fault_tick,
            repair_tick,
        )
        if (
            not selected
            and outage["start"] <= repair_tick
            and outage["end"] > fault_tick
        ):
            return {**result, "reason": "overlapping_fault", "observed_repair_ticks": 0}
    later_faults = [
        e["tick"]
        for e in events
        if e["kind"] == "breakdown" and e["tick"] > repair_tick
    ]
    end = min(a["end_tick"], b["end_tick"], min(later_faults, default=math.inf))
    if end <= repair_tick:
        return {**result, "reason": "overlapping_fault", "observed_repair_ticks": 0}
    streak = 0
    for start in range(
        repair_tick, int(end) - protocol.window_ticks + 1, protocol.window_ticks
    ):
        finish = start + protocol.window_ticks
        aa, bb = (
            fixed_window(nominal, start, finish),
            fixed_window(perturbed, start, finish),
        )
        if aa["status"] != "observed" or bb["status"] != "observed":
            return {**result, "reason": "window_outside_recording"}
        if not aa["completed"]:
            return {
                **result,
                "status": "not_applicable",
                "reason": "zero_reference",
                "observed_repair_ticks": start - repair_tick,
            }
        if all(
            o["completion_tick"] is not None and o["completion_tick"] <= finish
            for o in a["orders"]
        ):
            return {
                **result,
                "reason": "demand_exhausted",
                "observed_repair_ticks": start - repair_tick,
            }
        streak = (
            streak + 1 if bb["completed"] >= protocol.fraction * aa["completed"] else 0
        )
        if streak >= protocol.consecutive_windows:
            since_repair = finish - repair_tick
            return {
                **result,
                "status": "threshold_confirmed",
                "confirmation_tick": finish,
                "fault_to_confirmation_ticks": finish - fault_tick,
                "repair_to_confirmation_ticks": since_repair,
                "adaptability": 1 / since_repair,
            }
    return {
        **result,
        "reason": "new_fault"
        if later_faults and end == min(later_faults)
        else "observation_ended",
        "observed_repair_ticks": max(0, end - repair_tick),
    }


def relative_gain(baseline, social, *, direction):
    """Report both the proposal's raw SIG and a direction-aware improvement.

    Values must already be matched by budget, panel and decoder. Costs have lower
    is better semantics; a negative raw SIG then denotes improvement, not harm.
    """
    if direction not in ("lower", "higher"):
        raise ValueError("explicit metric direction required")
    if baseline is None or social is None:
        return {"status": "not_recorded", "raw_sig": None, "improvement": None}
    if not _finite(baseline) or not _finite(social) or min(baseline, social) < 0:
        raise ValueError("gain requires finite nonnegative metrics")
    if baseline == 0:
        return {
            "status": "not_applicable",
            "reason": "zero_reference",
            "raw_sig": None,
            "improvement": None,
        }
    raw = (social - baseline) / baseline
    return {
        "status": "observed",
        "raw_sig": raw,
        "improvement": -raw if direction == "lower" else raw,
        "direction": direction,
    }


def compare_cost_curves(base, social, *, budget_unit):
    """Trapezoidal cost AUC on exactly matched observed budgets; no extrapolation.

    AUC is not convergence. Censored panels remain in the accounting and suppress
    the cost comparison. Replicates must first be combined within a training seed.
    """
    if budget_unit not in ("episodes", "physical_ticks", "training_seconds"):
        raise ValueError("unsupported budget unit")
    if len(base) != len(social) or len(base) < 2:
        raise ValueError("at least two matched points required")
    previous = -1
    identity = None
    censored = False
    for a, b in zip(base, social, strict=True):
        if (
            not _finite(a["budget"])
            or a["budget"] < 0
            or a["budget"] <= previous
            or a["budget"] != b["budget"]
        ):
            raise ValueError("curves require identical strictly increasing budgets")
        previous = a["budget"]
        for point in (a, b):
            current = tuple(point[k] for k in ("train_seed", "decoder", "panel_sha256"))
            if any(v is None or v == "" for v in current) or (
                identity is not None and current != identity
            ):
                raise ValueError("curves require paired seed, decoder and panel")
            identity = current
            censored |= (
                point.get("all_drained") is not True or point.get("cost") is None
            )
            if point.get("cost") is not None and (
                not _finite(point["cost"]) or point["cost"] < 0
            ):
                raise ValueError("invalid cost")
    result = {
        "budget_unit": budget_unit,
        "budget_range": [base[0]["budget"], base[-1]["budget"]],
        "points": len(base),
    }
    if censored:
        return {**result, "status": "censored", "reason": "incomplete_cost_panel"}

    def area(points):
        return sum(
            (b["budget"] - a["budget"]) * (a["cost"] + b["cost"]) / 2
            for a, b in zip(points, points[1:])
        )

    aa, bb = area(base), area(social)
    span = base[-1]["budget"] - base[0]["budget"]
    return {
        **result,
        "status": "observed",
        "base_auc": aa,
        "social_auc": bb,
        "base_mean_cost": aa / span,
        "social_mean_cost": bb / span,
        "gain": relative_gain(aa, bb, direction="lower"),
    }
