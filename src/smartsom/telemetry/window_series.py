"""Exact post-episode throughput windows, independent of sampled telemetry."""

from bisect import bisect_right


def throughput_series(row, settings):
    detail = row.get("research_metrics", {})
    width, step = settings.throughput_width_ticks, settings.throughput_step_ticks
    base = {"width_ticks": width, "step_ticks": step, "interval": "(start,end]"}
    if detail.get("level") != "detailed" or "orders" not in detail:
        return {**base, "status": "not_recorded", "reason": "detailed_orders_missing"}
    start, end = detail["start_tick"], detail["end_tick"]
    if end - start >= width and 1 + (end - start - width) // step > 10000:
        return {
            **base,
            "status": "not_recorded",
            "reason": "window_count_exceeds_10000",
        }
    orders = detail["orders"]
    times = sorted(
        o["completion_tick"] for o in orders if o["completion_tick"] is not None
    )
    qualified = sorted(
        o["completion_tick"]
        for o in orders
        if o["completion_tick"] is not None and o["final_pass"] is True
    )
    windows = []
    for left in range(start, end - width + 1, step):
        right = left + width
        done = bisect_right(times, right) - bisect_right(times, left)
        passed = bisect_right(qualified, right) - bisect_right(qualified, left)
        windows.append(
            {
                "start": left,
                "end": right,
                "completed": done,
                "qualified": passed,
                "throughput": done / width,
                "qualified_throughput": passed / width,
            }
        )
    return {
        **base,
        "status": "observed" if windows else "insufficient_horizon",
        "observation_start": start,
        "observation_end": end,
        "partial_final_window": "omitted",
        "windows": windows,
    }
