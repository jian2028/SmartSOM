"""Explicit, offline throughput-window selection; no simulator side effects."""

from bisect import bisect_right
from statistics import mean, stdev


def _windows(row, start, end, width, *, anchor_end):
    count = max(0, (end - start) // width)
    if count > 10000:
        return None
    detail = row["research_metrics"]
    if start < detail["start_tick"] or end > detail["end_tick"]:
        return None
    times = sorted(
        o["completion_tick"]
        for o in detail["orders"]
        if o["completion_tick"] is not None
    )
    qualified = sorted(
        o["completion_tick"]
        for o in detail["orders"]
        if o["completion_tick"] is not None and o["final_pass"] is True
    )
    first = end - count * width if anchor_end else start
    result = []
    for i in range(count):
        left = first + i * width
        right = left + width
        completed = bisect_right(times, right) - bisect_right(times, left)
        passed = bisect_right(qualified, right) - bisect_right(qualified, left)
        point = {
            "status": "observed",
            "start_tick": left,
            "end_tick": right,
            "completed": completed,
            "qualified": passed,
            "throughput": completed / width,
            "qualified_throughput": passed / width,
        }
        result.append(point)
    return result


def _select(
    windows, spec, *, manual_count=None, allow_zero=False, force_fallback=False
):
    if windows is None:
        return {
            "status": "unavailable",
            "reason": "windows_unobserved_or_limit",
            "windows": [],
        }
    if manual_count is not None:
        selected = windows[-manual_count:]
        if len(selected) != manual_count:
            return {
                "status": "unavailable",
                "reason": "insufficient_manual_windows",
                "windows": windows,
            }
        return {
            "status": "manual",
            "selected": selected,
            "mean_throughput": mean(w["throughput"] for w in selected),
            "windows": windows,
        }
    count = spec.plateau_windows
    if force_fallback or max(0, len(windows) - count + 1) * count > 2_000_000:
        selected = windows[-spec.preset_windows :]
        reason = (
            "aggregate_plateau_work_limit" if force_fallback else "plateau_work_limit"
        )
        if len(selected) != spec.preset_windows:
            return {
                "status": "unavailable",
                "reason": reason,
                "windows": windows,
            }
        return {
            "status": "preset",
            "reason": reason,
            "selected": selected,
            "mean_throughput": mean(w["throughput"] for w in selected),
            "windows": windows,
        }
    positions = range(max(0, len(windows) - count + 1))
    if spec.search_order == "latest":
        positions = reversed(positions)
    for i in positions:
        selected = windows[i : i + count]
        rates = [w["throughput"] for w in selected]
        avg = mean(rates)
        if avg == 0 and not allow_zero:
            continue
        center = (count - 1) / 2
        slope = sum(
            (j - center) * (value - avg) for j, value in enumerate(rates)
        ) / sum((j - center) ** 2 for j in range(count))
        cv = stdev(rates) / avg if avg else None
        relative = slope / avg if avg else None
        if avg == 0 or (
            cv <= spec.maximum_cv
            and abs(relative) <= spec.maximum_relative_trend
            and relative <= spec.maximum_upward_trend
        ):
            return {
                "status": "plateau",
                "selected": selected,
                "mean_throughput": avg,
                "cv": cv,
                "relative_trend_per_window": relative,
                "zero_output_constant": avg == 0,
                "windows": windows,
            }
    selected = windows[-spec.preset_windows :]
    if len(selected) != spec.preset_windows:
        return {
            "status": "unavailable",
            "reason": "plateau_not_found_and_insufficient_preset_windows",
            "plateau_not_found": True,
            "windows": windows,
        }
    return {
        "status": "preset",
        "reason": "plateau_not_found",
        "plateau_not_found": True,
        "selected": selected,
        "mean_throughput": mean(w["throughput"] for w in selected),
        "windows": windows,
    }


def select_periods(
    row, fault_tick, repair_tick, sample_ticks, settings, *, force_fallback=False
):
    """Select pre-fault and fault windows, retaining all counts and preset-window evidence."""
    detail = row["research_metrics"]
    start = max(detail["start_tick"], settings.warmup_end_tick)
    pre = _windows(row, start, fault_tick, sample_ticks, anchor_end=True)
    during = _windows(row, fault_tick, repair_tick, sample_ticks, anchor_end=True)
    if settings.mode == "manual":
        base = _select(pre, None, manual_count=settings.baseline_windows)
        fault = {"status": "not_requested", "windows": during or []}
    else:
        base = _select(pre, settings.auto, force_fallback=force_fallback)
        fault = _select(
            during, settings.auto, allow_zero=True, force_fallback=force_fallback
        )
    return {
        "pre_fault": base,
        "fault_period": fault,
        "sample_window_ticks": sample_ticks,
        "fault_tick": fault_tick,
        "repair_tick": repair_tick,
        "interval": "(start,end]",
    }
