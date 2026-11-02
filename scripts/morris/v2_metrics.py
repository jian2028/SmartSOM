"""Summaries for the v2 evaluation protocol.

usage:
  python v2_metrics.py RUN_DIR [RUN_DIR ...]            # per-run summary
  python v2_metrics.py --paired CONTROL_DIR BREAKDOWN_DIR  # adaptability / robustness
"""
import glob
import json
import statistics as st
import sys

TWINS = [("1", "8"), ("2", "7"), ("3", "6"), ("4", "5")]
W = 300  # rolling throughput window (ticks)


def rows(run):
    out = {}
    for p in glob.glob(f"{run}/**/evidence/case-*/result.json", recursive=True):
        r = json.load(open(p))
        if r.get("evaluation_metrics"):
            seed = p.split("/entries/")[1][:10] if "/entries/" in p else "x"
            out[(seed, r["replication"], r["seed"])] = r
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def summary(run):
    rs = list(rows(run).values())
    if not rs:
        return {"episodes": 0}
    util = [r["evaluation_metrics"]["utilisation"] for r in rs]
    gaps = {
        f"{a}/{b}": mean(
            abs(u["machine"][f"machine_00{a}"] - u["machine"][f"machine_00{b}"])
            for u in util
        )
        for a, b in TWINS
    }
    return {
        "episodes": len(rs),
        "completed": sum(r["status"] == "completed" for r in rs),
        "throughput_per_1000": mean(
            1000 * r["delivered"] / u["duration_ticks"] for r, u in zip(rs, util)
        ),
        "makespan": mean(
            r.get("fixed_job_makespan") for r in rs if r["status"] == "completed"
        ),
        "mean_tardiness": mean(r["mean_tardiness"] for r in rs),
        "late_jobs": mean(r.get("tardy_jobs") for r in rs),
        "machine_util": mean(u["machine_mean"] for u in util),
        "agv_util": mean(u["agv_mean"] for u in util),
        "agv_loaded_waiting": mean(u.get("agv_loaded_waiting_mean") for u in util),
        "twin_gap": gaps,
    }


def rolling(ticks, t):
    return sum(t - W <= x < t for x in ticks) * 1000 / W


def intervals(outs, horizon):
    """(machine, down, up) per outage; each repair closes that machine's breakdown."""
    open_, rows = {}, []
    for o in sorted(outs, key=lambda o: (o["tick"], o["kind"] != "repair")):
        if o["kind"] == "breakdown":
            open_[o["machine"]] = o["tick"]
        elif o["machine"] in open_:
            rows.append((o["machine"], open_.pop(o["machine"]), o["tick"]))
    rows += [(m, t, None) for m, t in open_.items()]  # not repaired before the end
    return sorted(rows, key=lambda r: r[1])


def paired(control, broken):
    """Every outage in the breakdown run is scored against the same window of the
    paired no-breakdown run. Recovery is searched only until the next breakdown
    starts (or the episode ends); otherwise the event is censored."""
    a, b = rows(control), rows(broken)
    keys = sorted(set(a) & set(b))
    adapt, recover, censored, dmake, dtard, events = [], [], 0, [], [], 0
    for k in keys:
        ra, rb = a[k], b[k]
        ca, cb = ra["evaluation_metrics"]["delivery_ticks"], rb["evaluation_metrics"]["delivery_ticks"]
        end = max(ca + cb + [0]) + 1
        spans = intervals(rb["evaluation_metrics"]["outages"], end)
        starts = [s for _, s, _ in spans]
        for _, tb, tr in spans:
            events += 1
            stop = tr if tr is not None else end
            base = sum(tb <= x < stop for x in ca)
            if base:
                adapt.append(sum(tb <= x < stop for x in cb) / base)
            if tr is None:
                censored += 1
                continue
            limit = min([s for s in starts if s >= tr] + [end])
            hit = None
            for t in range(tr, limit, 10):
                if all(
                    rolling(cb, s) >= 0.95 * rolling(ca, s)
                    for s in range(t, min(t + W, end), 10)
                ):
                    hit = t - tr
                    break
            if hit is None:
                censored += 1
            else:
                recover.append(hit)
        if spans:
            if ra.get("fixed_job_makespan") and rb.get("fixed_job_makespan"):
                dmake.append(rb["fixed_job_makespan"] - ra["fixed_job_makespan"])
            dtard.append(rb["mean_tardiness"] - ra["mean_tardiness"])
    return {
        "pairs": len(keys),
        "outage_events": events,
        "adaptability_mean": mean(adapt),
        "adaptability_pass_rate": mean([x >= 0.95 for x in adapt]),
        "recovery_ticks_mean": mean(recover),
        "recovery_censored": censored,
        "makespan_increase": mean(dmake),
        "tardiness_increase": mean(dtard),
    }


def fmt(d):
    return json.dumps(d, default=lambda x: round(x, 3) if isinstance(x, float) else x)


if __name__ == "__main__":
    if sys.argv[1] == "--paired":
        print(fmt({k: (round(v, 3) if isinstance(v, float) else v) for k, v in paired(sys.argv[2], sys.argv[3]).items()}))
    else:
        for run in sys.argv[1:]:
            s = summary(run)
            s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
            if isinstance(s.get("twin_gap"), dict):
                s["twin_gap"] = {k: round(v, 2) for k, v in s["twin_gap"].items()}
            print(run.rstrip("/").split("/")[-1][:45], fmt(s))
