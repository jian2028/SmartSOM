"""Frozen baseline and threshold recovery; never infer equilibrium from storage."""

from statistics import mean, stdev

from smartsom.telemetry.research_comparison import fixed_window


def platform(row, candidate, fault_tick):
    width, count = candidate.window_ticks, candidate.baseline_windows
    windows = [
        fixed_window(
            row, fault_tick - (count - i) * width, fault_tick - (count - i - 1) * width
        )
        for i in range(count)
    ]
    if any(w["status"] != "observed" for w in windows):
        return {"status": "insufficient_baseline", "reason": "window_not_observed"}
    if any(w["completed"] < candidate.minimum_window_deliveries for w in windows):
        return {"status": "insufficient_baseline", "reason": "insufficient_output"}
    rates = [w["throughput"] for w in windows]
    average = mean(rates)
    center = (count - 1) / 2
    slope = sum((i - center) * (v - average) for i, v in enumerate(rates)) / sum(
        (i - center) ** 2 for i in range(count)
    )
    cv, trend = stdev(rates) / average, abs(slope) / average
    valid = cv <= candidate.maximum_cv and trend <= candidate.maximum_relative_trend
    return {
        "status": "valid" if valid else "insufficient_baseline",
        "mean_throughput": average,
        "cv": cv,
        "relative_trend_per_window": trend,
        "window_rates": rates,
        "window_ticks": width,
    }


def recovery_evidence(nominal, perturbed, candidate, fault):
    from smartsom.telemetry.research_comparison import _paired

    paired = _paired(nominal, perturbed)
    if paired is not None:
        return paired
    baseline = platform(nominal, candidate, fault.start_tick)
    own_baseline = platform(perturbed, candidate, fault.start_tick)
    result = {
        "baseline": baseline,
        "perturbed_baseline": own_baseline,
        "fault_tick": fault.start_tick,
        "repair_tick": fault.start_tick + fault.duration_ticks,
        "window_ticks": candidate.window_ticks,
        "fraction": 0.95,
        "consecutive_windows": 3,
        "adaptability": None,
    }
    if baseline["status"] != "valid" or own_baseline["status"] != "valid":
        return {**result, "status": "insufficient_baseline"}
    if nominal.get("last_arrival", 0) < candidate.horizon_ticks:
        return {**result, "status": "insufficient_supply"}
    reference = own_baseline["mean_throughput"]
    streak, first, degraded = 0, None, False
    end = min(nominal["tick"], perturbed["tick"], candidate.horizon_ticks)
    windows = []
    for start in range(
        fault.start_tick, end - candidate.window_ticks + 1, candidate.window_ticks
    ):
        finish = start + candidate.window_ticks
        value = fixed_window(perturbed, start, finish)
        control = fixed_window(nominal, start, finish)
        if value["status"] != "observed" or control["status"] != "observed":
            break
        windows.append(
            {
                "start": start,
                "end": finish,
                "throughput": value["throughput"],
                "normal_throughput": control["throughput"],
            }
        )
        if value["throughput"] < 0.95 * reference:
            degraded, streak, first = True, 0, None
        elif degraded:
            if streak == 0:
                first = finish
            streak += 1
            if streak == 3:
                duration = first - fault.start_tick
                return {
                    **result,
                    "status": "recovered",
                    "first_qualifying_tick": first,
                    "confirmation_tick": finish,
                    "fault_to_recovery_ticks": duration,
                    "repair_to_recovery_ticks": max(0, first - result["repair_tick"]),
                    "recovered_before_repair": first < result["repair_tick"],
                    "adaptability": 1 / duration,
                    "windows": windows,
                }
    return {
        **result,
        "status": "not_recovered" if degraded else "no_observed_degradation",
        "windows": windows,
        "observed_end": end,
    }
