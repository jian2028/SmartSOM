"""Export explicitly selected, identity-checked paired evaluation evidence."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from smartsom.config.codec import StrictModel, read_model
from smartsom.config.evaluation_protocol import WindowAnalysis
from smartsom.telemetry.research_comparison import (
    RecoveryProtocol,
    _paired,
    _window_robustness,
    fixed_window,
    recovery,
)
from smartsom.telemetry.research_metrics import aggregate


class CaseSelection(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line: int = Field(gt=0)


class Window(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class WindowGrid(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    width: int | None = Field(default=None, gt=0)
    step: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_grid(self):
        if self.width is None or self.step is None:
            return self
        if self.end - self.start < self.width:
            raise ValueError("window grid must contain at least one full window")
        if 1 + (self.end - self.start - self.width) // self.step > 10000:
            raise ValueError("window grid exceeds 10000 windows")
        return self


class Fault(StrictModel):
    machine_id: str = Field(min_length=1)
    fault_tick: int = Field(ge=0)
    repair_tick: int = Field(gt=0)


class BaselineSettings(StrictModel):
    window_ticks: int = Field(gt=0, strict=True)
    baseline_windows: int = Field(ge=1, strict=True)
    minimum_window_deliveries: int = Field(gt=0, strict=True)
    warmup_end_tick: int = Field(ge=0, strict=True)
    maximum_cv: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    maximum_relative_trend: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def thresholds(self):
        if self.baseline_windows > 1 and (
            self.maximum_cv is None or self.maximum_relative_trend is None
        ):
            raise ValueError(
                "multiple baseline windows require explicit CV and trend thresholds"
            )
        if self.baseline_windows == 1 and (
            self.maximum_cv is not None or self.maximum_relative_trend is not None
        ):
            raise ValueError("single-window baseline cannot assess CV or trend")
        return self


class RecoverySettings(StrictModel):
    semantics: Literal[
        "paired_repair_anchored", "same_run_pre_fault", "configured_same_run"
    ]
    duration_from: Literal["fault", "repair"] | None = None
    version: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    window_ticks: int | None = Field(default=None, gt=0)
    consecutive_windows: int | None = Field(default=None, gt=0)
    fraction: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    baseline: BaselineSettings | None = None

    @model_validator(mode="after")
    def valid_baseline(self):
        if self.semantics == "configured_same_run":
            if any(
                x is not None
                for x in (
                    self.baseline,
                    self.duration_from,
                    self.consecutive_windows,
                    self.fraction,
                )
            ):
                raise ValueError(
                    "configured same-run parameters must come from experiment YAML"
                )
            return self
        if any(
            x is None
            for x in (self.duration_from, self.consecutive_windows, self.fraction)
        ):
            raise ValueError("recovery duration, windows and fraction are required")
        if self.semantics == "same_run_pre_fault":
            if self.baseline is None or self.duration_from != "repair":
                raise ValueError(
                    "same-run mode requires explicit baseline and repair origin"
                )
        elif self.baseline is not None:
            raise ValueError("paired mode does not use same-run baseline settings")
        return self


class ExportSettings(StrictModel):
    schema_id: Literal["smartsom.paired-export/v1"] = Field(alias="schema")
    experiment: str | None = None
    nominal: CaseSelection | None = None
    perturbed: CaseSelection
    windows: list[Window] = Field(default_factory=list, max_length=10000)
    window_grid: WindowGrid | None = None
    overall_window: Window
    fault: Fault
    recovery: RecoverySettings

    @model_validator(mode="after")
    def one_window_specification(self):
        if bool(self.windows) == (self.window_grid is not None):
            raise ValueError("provide exactly one of windows or window_grid")
        if (self.recovery.semantics == "paired_repair_anchored") != (
            self.nominal is not None
        ):
            raise ValueError("nominal case is required only in paired mode")
        return self


class _EvaluationView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    window_analysis: WindowAnalysis


class _ExperimentView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    evaluation: _EvaluationView
    run: dict = Field(default_factory=dict)


def _load_case(selection, config_directory):
    path = (config_directory / selection.path).resolve()
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != selection.sha256:
        raise ValueError(f"case file hash mismatch: {path}")
    lines = data.splitlines()
    if selection.line > len(lines):
        raise ValueError(f"case line out of range: {path}")
    row = json.loads(lines[selection.line - 1])
    if not isinstance(row, dict):
        raise ValueError("selected case must be a JSON object")
    return row, {"path": str(path), "sha256": actual, "line": selection.line}


def _observed_window(row, start, end):
    detail = row.get("research_metrics", {})
    if detail.get("level") == "detailed" and end > detail.get("end_tick", -1):
        return {"status": "censored", "reason": "window_outside_recording"}
    return fixed_window(row, start, end)


def _outage_pair(nominal, perturbed, fault):
    nominal_detail = nominal.get("research_metrics", {})
    perturbed_detail = perturbed.get("research_metrics", {})
    normal_events = nominal_detail.get("events")
    fault_events = perturbed_detail.get("events")
    if isinstance(normal_events, list) and any(
        event.get("kind") == "breakdown"
        and event.get("machine") == fault.machine_id
        and event.get("tick") == fault.fault_tick
        for event in normal_events
    ):
        raise ValueError("nominal case contains the selected fault event")
    left = nominal_detail.get("outage_realization")
    right = perturbed_detail.get("outage_realization")
    if left is None or right is None:
        return "outage_realization_missing"
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("outage realization must be a list")

    def identity(outages):
        return Counter(
            json.dumps(x, sort_keys=True, separators=(",", ":")) for x in outages
        )

    selected = {
        "machine_id": fault.machine_id,
        "start": fault.fault_tick,
        "end": fault.repair_tick,
    }
    selected_key = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    baseline = identity(left)
    if baseline[selected_key] or identity(right) != baseline + Counter([selected_key]):
        raise ValueError(
            "paired outage plans must differ by exactly the selected fault"
        )
    if not isinstance(normal_events, list) or not isinstance(fault_events, list):
        return "event_log_missing"
    if not any(
        event.get("kind") == "breakdown"
        and event.get("machine") == fault.machine_id
        and event.get("tick") == fault.fault_tick
        for event in fault_events
    ):
        return "selected_fault_event_missing"
    return None


def _identity_present(row):
    identity = row.get("evaluation_identity")
    detail = row.get("research_metrics", {})
    if not isinstance(identity, dict) or not isinstance(detail, dict):
        return False
    fingerprint = detail.get("fingerprints", {}).get("exogenous")
    strings = (
        "method",
        "selection",
        "source_sha256",
        "runtime_sha256",
        "algorithm_sha256",
    )
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    if any(not isinstance(identity.get(k), str) or not identity[k] for k in strings):
        return False
    if (
        type(identity.get("learning")) is not bool
        or type(identity.get("policy_seed")) is not int
    ):
        return False
    checkpoint = identity.get("checkpoint_sha256")
    return (not identity["learning"] and checkpoint is None) or (
        isinstance(checkpoint, str) and bool(checkpoint)
    )


def _outcomes(row):
    detail = row.get("research_metrics", {})
    summary = detail.get("summary", {})
    total, accrued = row.get("total_orders"), row.get("observed_overdue")
    observed = (
        accrued / total
        if type(total) is int and total > 0 and type(accrued) in (int, float)
        else None
    )
    return {
        "scope": "own_observation_horizon_not_common_window",
        "start_tick": detail.get("start_tick"),
        "end_tick": detail.get("end_tick"),
        "recorded_sample_interval_ticks": detail.get("sample_interval_ticks"),
        "performance": aggregate(row),
        "observed_tardiness_per_order": observed,
        "observed_tardiness_scope": "lower_bound_if_incomplete",
        "on_time_coverage": summary.get("on_time_coverage"),
        "conditional_on_time_rate": summary.get("conditional_on_time_rate"),
        "resources": {
            "machine_state_ticks": summary.get("machine_state_ticks"),
            "machine_state_fractions": summary.get("machine_state_fractions"),
            "available_processing_fraction": summary.get(
                "available_processing_fraction"
            ),
            "machines": detail.get("machines"),
            "vehicles": detail.get("vehicles"),
            "buffers": detail.get("buffer_entities"),
            "definitions": {
                "machine_fraction": "state ticks / (machine count * observed ticks)",
                "available_processing_fraction": "processing / (machine ticks - down ticks)",
                "vehicle_states": "loaded/empty and moving/stationary are overlapping partitions",
                "vehicle_denominator": "per-vehicle observed ticks; each partition is separate",
                "buffer_occupancy": "resident item-ticks / observed ticks; excludes reservations",
            },
        },
    }


def _same_run_recovery(row, fault, settings, width):
    """Repair-anchored recovery against an explicitly validated pre-fault baseline."""
    from smartsom.telemetry.recovery_protocol import platform

    if fault.fault_tick >= fault.repair_tick:
        raise ValueError("repair must follow fault")
    detail = row.get("research_metrics", {})
    result = {"status": "not_recorded", "adaptability": None}
    events, outages = detail.get("events"), detail.get("outage_realization")
    if (
        detail.get("level") != "detailed"
        or detail.get("schema") != "smartsom.research-metrics/v1"
        or not isinstance(detail.get("orders"), list)
        or not isinstance(events, list)
        or not isinstance(outages, list)
    ):
        return {**result, "reason": "detailed_fault_evidence_missing"}
    if fault.repair_tick > detail.get("end_tick", -1):
        return {**result, "status": "censored", "reason": "observation_too_short"}
    if not all(
        any(
            e.get("kind") == kind
            and e.get("machine") == fault.machine_id
            and e.get("tick") == tick
            for e in events
        )
        for kind, tick in (
            ("breakdown", fault.fault_tick),
            ("repair", fault.repair_tick),
        )
    ):
        return {**result, "reason": "event_boundary_missing"}
    selected = {
        "machine_id": fault.machine_id,
        "start": fault.fault_tick,
        "end": fault.repair_tick,
    }
    if outages.count(selected) != 1:
        return {**result, "reason": "selected_outage_missing_or_ambiguous"}
    baseline_start = (
        fault.fault_tick
        - settings.baseline.window_ticks * settings.baseline.baseline_windows
    )
    if baseline_start < detail["start_tick"] or fault.repair_tick > detail["end_tick"]:
        return {**result, "status": "censored", "reason": "observation_too_short"}
    if baseline_start < settings.baseline.warmup_end_tick:
        return {
            **result,
            "status": "insufficient_baseline",
            "reason": "baseline_overlaps_configured_warmup",
        }
    other = [o for o in outages if o != selected]
    if any(o["start"] < fault.repair_tick and o["end"] > baseline_start for o in other):
        return {**result, "status": "censored", "reason": "overlapping_fault"}
    if settings.baseline.baseline_windows == 1:
        base_window = _observed_window(row, baseline_start, fault.fault_tick)
        if base_window["status"] != "observed":
            return {
                **result,
                "status": "censored",
                "reason": "baseline_window_not_observed",
            }
        if not base_window["completed"]:
            return {**result, "status": "not_applicable", "reason": "zero_baseline"}
        baseline = {
            "status": "valid"
            if base_window["completed"] >= settings.baseline.minimum_window_deliveries
            else "insufficient_baseline",
            "reason": None
            if base_window["completed"] >= settings.baseline.minimum_window_deliveries
            else "insufficient_output",
            "mean_throughput": base_window["throughput"],
            "window_ticks": settings.baseline.window_ticks,
            "window_rates": [base_window["throughput"]],
            "stability": "unverified_single_window",
        }
    else:
        pre = [
            _observed_window(
                row,
                fault.fault_tick
                - (settings.baseline.baseline_windows - i)
                * settings.baseline.window_ticks,
                fault.fault_tick
                - (settings.baseline.baseline_windows - i - 1)
                * settings.baseline.window_ticks,
            )
            for i in range(settings.baseline.baseline_windows)
        ]
        if all(w.get("status") == "observed" and w["completed"] == 0 for w in pre):
            return {**result, "status": "not_applicable", "reason": "zero_baseline"}
        baseline = platform(row, settings.baseline, fault.fault_tick)
        baseline["stability"] = (
            "passes_configured_checks"
            if baseline["status"] == "valid"
            else "not_established"
        )
    if baseline["status"] != "valid":
        return {
            **result,
            "status": "insufficient_baseline",
            "reason": baseline.get("reason", "unstable"),
            "baseline": baseline,
        }
    reference = baseline["mean_throughput"]
    if reference <= 0:
        return {
            **result,
            "status": "not_applicable",
            "reason": "zero_baseline",
            "baseline": baseline,
        }
    fault_window = _observed_window(row, fault.fault_tick, fault.repair_tick)
    fault_ratio = {
        "status": "observed",
        "ratio": fault_window["throughput"] / reference,
        "meaning": "fault_period_mean_to_pre_fault_baseline_not_steady_state",
    }
    end = min(
        [detail["end_tick"]]
        + [o["start"] for o in other if o["start"] >= fault.repair_tick]
    )
    streak = 0
    for start in range(fault.repair_tick, end - width + 1, width):
        finish = start + width
        # A drained finite-demand run has no observed sustained supply afterwards.
        orders = detail.get("orders", [])
        if orders and all(
            o["completion_tick"] is not None and o["completion_tick"] <= finish
            for o in orders
        ):
            return {
                "status": "censored",
                "reason": "demand_exhausted",
                "baseline": baseline,
                "fault_period_mean_ratio": fault_ratio,
                "adaptability": None,
            }
        window = _observed_window(row, start, finish)
        if window["status"] != "observed":
            return {
                "status": "censored",
                "reason": "window_outside_recording",
                "baseline": baseline,
                "fault_period_mean_ratio": fault_ratio,
                "adaptability": None,
            }
        streak = (
            streak + 1 if window["throughput"] >= settings.fraction * reference else 0
        )
        if streak >= settings.consecutive_windows:
            elapsed = finish - fault.repair_tick
            return {
                "status": "threshold_confirmed",
                "confirmation_tick": finish,
                "repair_to_confirmation_ticks": elapsed,
                "adaptability": 1 / elapsed,
                "baseline": baseline,
                "fault_period_mean_ratio": fault_ratio,
            }
    return {
        "status": "censored",
        "reason": "new_fault" if end < detail["end_tick"] else "observation_ended",
        "observed_repair_ticks": max(0, end - fault.repair_tick),
        "baseline": baseline,
        "fault_period_mean_ratio": fault_ratio,
        "adaptability": None,
    }


def _configured_recovery(
    row, fault, settings, sample_ticks, recovery_ticks, *, force_fallback=False
):
    return {
        "machine_id": fault.machine_id,
        "fault_tick": fault.fault_tick,
        "repair_tick": fault.repair_tick,
        **_configured_recovery_impl(
            row,
            fault,
            settings,
            sample_ticks,
            recovery_ticks,
            force_fallback=force_fallback,
        ),
    }


def _configured_recovery_impl(
    row, fault, settings, sample_ticks, recovery_ticks, *, force_fallback=False
):
    """Manual or configured plateau selection, followed by repair-anchored recovery."""
    from bisect import bisect_right

    from smartsom.telemetry.plateau_analysis import select_periods

    detail = row.get("research_metrics", {})
    invalid = {"status": "not_recorded", "adaptability": None}
    if fault.fault_tick >= fault.repair_tick:
        raise ValueError("repair must follow fault")
    events, outages = detail.get("events"), detail.get("outage_realization")
    if (
        detail.get("schema") != "smartsom.research-metrics/v1"
        or detail.get("level") != "detailed"
        or not isinstance(detail.get("orders"), list)
        or not isinstance(events, list)
        or not isinstance(outages, list)
    ):
        return {**invalid, "reason": "detailed_fault_evidence_missing"}
    if fault.repair_tick > detail.get("end_tick", -1):
        return {**invalid, "status": "censored", "reason": "observation_too_short"}
    if not all(
        any(
            e.get("kind") == kind
            and e.get("machine") == fault.machine_id
            and e.get("tick") == tick
            for e in events
        )
        for kind, tick in (
            ("breakdown", fault.fault_tick),
            ("repair", fault.repair_tick),
        )
    ):
        return {**invalid, "reason": "event_boundary_missing"}
    selected = {
        "machine_id": fault.machine_id,
        "start": fault.fault_tick,
        "end": fault.repair_tick,
    }
    if outages.count(selected) != 1:
        return {**invalid, "reason": "selected_outage_missing_or_ambiguous"}
    other = [o for o in outages if o != selected]
    periods = select_periods(
        row,
        fault.fault_tick,
        fault.repair_tick,
        sample_ticks,
        settings,
        force_fallback=force_fallback,
    )
    pre, fault_period = periods["pre_fault"], periods["fault_period"]
    if settings.mode == "auto":
        missing = [
            name
            for name, part in (("before_fault", pre), ("fault_period", fault_period))
            if part.get("plateau_not_found")
        ]
        periods["plateau_not_found"] = (
            "unknown"
            if any(
                p.get("reason")
                in {
                    "windows_unobserved_or_limit",
                    "plateau_work_limit",
                    "aggregate_plateau_work_limit",
                }
                for p in (pre, fault_period)
            )
            else "both"
            if len(missing) == 2
            else (missing[0] if missing else "none")
        )
    result = {**invalid, "period_selection": periods}
    if pre["status"] == "unavailable":
        return {**result, "status": "insufficient_baseline", "reason": pre["reason"]}
    start = pre["selected"][0]["start_tick"]
    if any(o["start"] < fault.repair_tick and o["end"] > start for o in other):
        return {**result, "status": "censored", "reason": "overlapping_fault"}
    reference = pre["mean_throughput"]
    if reference <= 0:
        return {**result, "status": "not_applicable", "reason": "zero_baseline"}
    fault_average = _observed_window(row, fault.fault_tick, fault.repair_tick)
    result["fault_period_mean_ratio"] = {
        "status": "observed",
        "ratio": fault_average["throughput"] / reference,
        "meaning": "entire_fault_period_mean_not_steady_state",
    }
    if settings.mode == "auto":
        result["robustness"] = (
            {
                "status": "plateau"
                if pre["status"] == "plateau" and fault_period["status"] == "plateau"
                else "preset",
                "baseline_source": pre["status"],
                "fault_source": fault_period["status"],
                "ratio": fault_period["mean_throughput"] / reference,
                "meaning": "plateau_candidate_ratio"
                if fault_period["status"] == "plateau" and pre["status"] == "plateau"
                else "preset_window_ratio_not_steady_state",
            }
            if fault_period["status"] != "unavailable"
            else {
                "status": "unavailable",
                "reason": fault_period["reason"],
                "ratio": None,
            }
        )
    else:
        result["robustness"] = result["fault_period_mean_ratio"]
    end = min(
        [detail["end_tick"]]
        + [o["start"] for o in other if o["start"] >= fault.repair_tick]
    )
    streak = 0
    result["recovery_windows"] = []
    if max(0, (end - fault.repair_tick) // recovery_ticks) > 10000:
        return {**result, "status": "not_recorded", "reason": "recovery_window_limit"}
    orders = detail["orders"]
    times = sorted(
        o["completion_tick"] for o in orders if o["completion_tick"] is not None
    )
    qualified = sorted(
        o["completion_tick"]
        for o in orders
        if o["completion_tick"] is not None and o["final_pass"] is True
    )
    for begin in range(fault.repair_tick, end - recovery_ticks + 1, recovery_ticks):
        finish = begin + recovery_ticks
        if orders and len(times) == len(orders) and times[-1] <= finish:
            return {**result, "status": "censored", "reason": "demand_exhausted"}
        completed = bisect_right(times, finish) - bisect_right(times, begin)
        passed = bisect_right(qualified, finish) - bisect_right(qualified, begin)
        point = {
            "status": "observed",
            "start_tick": begin,
            "end_tick": finish,
            "completed": completed,
            "qualified": passed,
            "throughput": completed / recovery_ticks,
            "qualified_throughput": passed / recovery_ticks,
        }
        result["recovery_windows"].append(point)
        streak = (
            streak + 1 if point["throughput"] >= settings.fraction * reference else 0
        )
        if streak >= settings.consecutive_windows:
            origin = (
                fault.fault_tick
                if settings.duration_from == "fault"
                else fault.repair_tick
            )
            duration = finish - origin
            return {
                **result,
                "status": "threshold_confirmed",
                "confirmation_tick": finish,
                "duration_from": settings.duration_from,
                "duration_ticks": duration,
                "adaptability": 1 / duration,
            }
    return {
        **result,
        "status": "censored",
        "reason": "new_fault" if end < detail["end_tick"] else "observation_ended",
        "observed_repair_ticks": max(0, end - fault.repair_tick),
    }


def same_run_episode(report, analysis, metric_interval):
    """Bound per-episode analysis and retain reasons instead of blocking training."""
    detail = report.get("research_metrics", {})
    outages = detail.get("outage_realization")
    if detail.get("level") != "detailed" or not isinstance(outages, list):
        result = {
            "status": "not_recorded",
            "reason": "detailed_fault_evidence_missing",
            "events": [],
        }
    else:
        observed = [o for o in outages if o["start"] <= detail["end_tick"]]
        projected = len(observed) * (
            1
            + (detail["end_tick"] - detail["start_tick"])
            // min(metric_interval, analysis.recovery_window_ticks)
        )
        if (
            len(observed) > analysis.same_run.maximum_fault_events
            or projected > 100_000
        ):
            result = {
                "status": "not_recorded",
                "reason": "fault_event_limit"
                if len(observed) > analysis.same_run.maximum_fault_events
                else "aggregate_window_limit",
                "observed_fault_count": len(observed),
                "events": [],
            }
        else:
            plateau_width = (
                analysis.same_run.auto.plateau_windows if analysis.same_run.auto else 0
            )
            force = bool(plateau_width and projected * plateau_width > 2_000_000)
            result = {
                "status": "observed",
                "events": [
                    _configured_recovery(
                        report,
                        Fault(
                            machine_id=o["machine_id"],
                            fault_tick=o["start"],
                            repair_tick=o["end"],
                        ),
                        analysis.same_run,
                        metric_interval,
                        analysis.recovery_window_ticks,
                        force_fallback=force,
                    )
                    for o in observed
                ],
            }
    result["settings"] = analysis.same_run.model_dump(mode="json")
    result["sample_window_ticks"] = metric_interval
    result["recovery_window_ticks"] = analysis.recovery_window_ticks
    return result


def _window_specs(settings, analysis):
    if settings.window_grid is None:
        return settings.windows
    grid = settings.window_grid
    width = (
        grid.width
        if grid.width is not None
        else (analysis.throughput_width_ticks if analysis else None)
    )
    step = (
        grid.step
        if grid.step is not None
        else (analysis.throughput_step_ticks if analysis else None)
    )
    if width is None or step is None:
        raise ValueError(
            "window grid width and step require experiment settings or explicit values"
        )
    if analysis is not None and (
        width != analysis.throughput_width_ticks
        or step != analysis.throughput_step_ticks
    ):
        raise ValueError("window grid differs from experiment")
    grid = WindowGrid(start=grid.start, end=grid.end, width=width, step=step)
    return [
        Window(start=start, end=start + grid.width)
        for start in range(grid.start, grid.end - grid.width + 1, grid.step)
    ]


def export(config):
    """Return explicit protocol, source identities, raw cases and paired metrics.

    No evaluation or model execution occurs. Incompatible identities raise;
    missing evidence and censoring retain the comparison functions' statuses.
    """
    path = Path(config).expanduser().resolve()
    settings, config_sha256 = read_model(path, ExportSettings)
    experiment_source = None
    analysis = None
    if settings.experiment is not None:
        experiment_path = (path.parent / settings.experiment).resolve()
        experiment, experiment_sha256 = read_model(experiment_path, _ExperimentView)
        analysis = experiment.evaluation.window_analysis
        experiment_source = {"path": str(experiment_path), "sha256": experiment_sha256}
    recovery_width = settings.recovery.window_ticks
    if analysis is not None:
        if (
            recovery_width is not None
            and recovery_width != analysis.recovery_window_ticks
        ):
            raise ValueError("recovery window differs from experiment")
        recovery_width = analysis.recovery_window_ticks
    if recovery_width is None:
        raise ValueError("recovery window must be provided by experiment or protocol")
    perturbed, perturbed_source = _load_case(settings.perturbed, path.parent)
    window_specs = _window_specs(settings, analysis)
    if settings.recovery.semantics == "configured_same_run":
        if analysis is None or analysis.same_run is None:
            raise ValueError(
                "configured same-run requires experiment window_analysis.same_run"
            )
        from smartsom.config.base_diagnostic import ReservationRun

        sample_ticks = experiment.run.get(
            "metric_interval", ReservationRun.model_fields["metric_interval"].default
        )
        if type(sample_ticks) is not int or sample_ticks <= 0:
            raise ValueError(
                "configured same-run requires explicit run.metric_interval"
            )
        observed_interval = perturbed.get("research_metrics", {}).get(
            "sample_interval_ticks"
        )
        if observed_interval != sample_ticks:
            result = {
                "status": "not_recorded",
                "reason": "record_interval_missing_or_mismatch",
                "adaptability": None,
            }
        else:
            result = _configured_recovery(
                perturbed,
                settings.fault,
                analysis.same_run,
                sample_ticks,
                recovery_width,
            )
        windows = []
        for window in window_specs:
            if window.start >= window.end:
                raise ValueError("window start must precede end")
            windows.append(
                {
                    "start": window.start,
                    "end": window.end,
                    "perturbed": _observed_window(perturbed, window.start, window.end),
                }
            )
        overall = settings.overall_window
        if overall.start >= overall.end:
            raise ValueError("overall window start must precede end")
        return {
            "schema": "smartsom.same-run-export-result/v1",
            "config_sha256": config_sha256,
            "settings": settings.model_dump(mode="json", by_alias=True),
            "experiment_source": experiment_source,
            "experiment_source_role": "analysis_window_configuration_not_case_provenance",
            "resolved_same_run": analysis.same_run.model_dump(mode="json"),
            "sample_window_ticks": sample_ticks,
            "resolved_recovery_window_ticks": recovery_width,
            "sources": {"perturbed": perturbed_source},
            "cases": {"perturbed": perturbed},
            "outcomes": {"perturbed": _outcomes(perturbed)},
            "windows": windows,
            "overall": {
                "start": overall.start,
                "end": overall.end,
                "perturbed": _observed_window(perturbed, overall.start, overall.end),
            },
            "recovery": result,
        }
    if settings.recovery.semantics == "same_run_pre_fault":
        result = _same_run_recovery(
            perturbed, settings.fault, settings.recovery, recovery_width
        )
        windows = []
        for window in window_specs:
            if window.start >= window.end:
                raise ValueError("window start must precede end")
            windows.append(
                {
                    "start": window.start,
                    "end": window.end,
                    "perturbed": _observed_window(perturbed, window.start, window.end),
                }
            )
        overall = settings.overall_window
        if overall.start >= overall.end:
            raise ValueError("overall window start must precede end")
        return {
            "schema": "smartsom.same-run-export-result/v1",
            "config_sha256": config_sha256,
            "settings": settings.model_dump(mode="json", by_alias=True),
            "resolved_recovery_window_ticks": recovery_width,
            "experiment_source": experiment_source,
            "experiment_source_role": "analysis_window_configuration_not_case_provenance",
            "sources": {"perturbed": perturbed_source},
            "cases": {"perturbed": perturbed},
            "outcomes": {"perturbed": _outcomes(perturbed)},
            "windows": windows,
            "overall": {
                "start": overall.start,
                "end": overall.end,
                "perturbed": _observed_window(perturbed, overall.start, overall.end),
            },
            "window_basis": "exact completion ticks; (start,end]; no interpolation",
            "recovery": result,
        }
    nominal, nominal_source = _load_case(settings.nominal, path.parent)
    protocol = RecoveryProtocol(
        version=settings.recovery.version,
        reference=settings.recovery.reference,
        window_ticks=recovery_width,
        consecutive_windows=settings.recovery.consecutive_windows,
        fraction=settings.recovery.fraction,
    )
    outage_problem = _outage_pair(nominal, perturbed, settings.fault)
    has_identity = _identity_present(nominal) and _identity_present(perturbed)

    def compare_window(window):
        if window.start >= window.end:
            raise ValueError("window start must precede end")
        baseline = _observed_window(nominal, window.start, window.end)
        treatment = _observed_window(perturbed, window.start, window.end)
        if has_identity:
            missing = _paired(nominal, perturbed)
            comparison = missing or _window_robustness(baseline, treatment)
        else:
            comparison = {
                "status": "not_recorded",
                "reason": "evaluation_identity_missing",
            }
        if outage_problem and comparison["status"] != "not_recorded":
            comparison = {
                "status": "not_recorded",
                "reason": outage_problem,
            }
        if comparison["status"] != "not_recorded" and (
            baseline["status"] == "censored" or treatment["status"] == "censored"
        ):
            comparison = {
                "status": "censored",
                "nominal": baseline,
                "perturbed": treatment,
            }
        return {
            "start": window.start,
            "end": window.end,
            "nominal": baseline,
            "perturbed": treatment,
            "comparison": comparison,
        }

    windows = [compare_window(window) for window in window_specs]
    overall = compare_window(settings.overall_window)
    if not has_identity:
        result = {"status": "not_recorded", "reason": "evaluation_identity_missing"}
    elif outage_problem:
        result = {"status": "not_recorded", "reason": outage_problem}
    else:
        result = recovery(
            nominal,
            perturbed,
            machine_id=settings.fault.machine_id,
            fault_tick=settings.fault.fault_tick,
            repair_tick=settings.fault.repair_tick,
            protocol=protocol,
        )
    if result.get("adaptability") is not None:
        result["adaptability_basis"] = "repair_to_confirmation_ticks"
    delay = result.get(
        "fault_to_confirmation_ticks"
        if settings.recovery.duration_from == "fault"
        else "repair_to_confirmation_ticks"
    )
    selected_adaptability = (
        1 / delay if result.get("status") == "threshold_confirmed" and delay else None
    )
    return {
        "schema": "smartsom.paired-export-result/v1",
        "config_sha256": config_sha256,
        "settings": settings.model_dump(mode="json", by_alias=True),
        "resolved_recovery_window_ticks": recovery_width,
        "experiment_source": experiment_source,
        "experiment_source_role": "analysis_window_configuration_not_case_provenance",
        "sources": {"nominal": nominal_source, "perturbed": perturbed_source},
        "cases": {"nominal": nominal, "perturbed": perturbed},
        "outcomes": {"nominal": _outcomes(nominal), "perturbed": _outcomes(perturbed)},
        "windows": windows,
        "window_basis": "exact completion ticks; (start,end]; no interpolation",
        "overall": overall,
        "recovery": result,
        "selected_adaptability": {
            "duration_from": settings.recovery.duration_from,
            "ticks": delay,
            "value_per_tick": selected_adaptability,
            "status": result["status"],
        },
    }


def write_export(path, payload):
    """Publish a complete JSON document exclusively; preserve any existing path."""
    target = Path(path).expanduser()
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        os.unlink(name)
