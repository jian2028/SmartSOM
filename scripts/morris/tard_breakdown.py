"""Split each shipped job's time (release -> ship) into where it was, from a recorded trace.

usage: python tard_breakdown.py CASE_DIR...   (dirs holding trace.jsonl + result.json)
Categories: source (waiting first pickup), pre (queue before a machine), machine,
post (done, waiting pickup), agv (on a vehicle), inspection, other; rework = time on attempt >= 2.
"""
import collections
import json
import sys


def where(loc):
    if loc is None:
        return "other"
    if loc.startswith("agv"):
        return "agv"
    if loc.startswith("machine"):
        return "machine"
    if loc.startswith("inspection"):
        return "inspection"
    if loc == "buffer_001":
        return "source"
    if loc.startswith("buffer_"):
        n = int(loc.split("_")[1])
        if n == 2:
            return "output"
        return "pre" if n % 2 else "post"
    return "other"


def case(path):
    due, cat, rework, modes = {}, collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter()
    ship = {}
    for line in open(f"{path}/trace.jsonl"):
        r = json.loads(line)
        for dec in r.get("decisions") or []:
            for job, row in ((dec.get("observation") or {}).get("jobs") or {}).items():
                if "due_at" in row:
                    due.setdefault(row["demand"], row["due_at"])
        bs = r["boundary_state"]
        ship = bs.get("shipment_times", ship)
        for job, row in bs["jobs"].items():
            d = row["demand"]
            if d in ship:
                continue
            c = where(row.get("location"))
            cat[d][c] += 1
            if not job.endswith("/attempt/1"):
                rework[d] += 1
        for m in bs["machines"].values():
            if m.get("job") and m.get("mode"):
                modes[m["mode"]] += 1
    return due, ship, cat, rework, modes


def main(paths):
    keys = ["source", "agv", "pre", "machine", "post", "inspection", "output", "other"]
    allj, tardy = collections.Counter(), collections.Counter()
    n_all = n_tardy = 0
    tard_total, rw_all, rw_tardy, modes = 0, 0, 0, collections.Counter()
    for p in paths:
        due, ship, cat, rework, m = case(p)
        modes += m
        for d, t in ship.items():
            if d not in due:
                continue
            late = max(0, t - due[d])
            allj.update(cat[d]); n_all += 1; rw_all += rework[d]
            if late > 0:
                tardy.update(cat[d]); n_tardy += 1; rw_tardy += rework[d]; tard_total += late
    print(f"cases={len(paths)} shipped={n_all} tardy={n_tardy} mean_lateness_of_tardy={tard_total / max(n_tardy, 1):.0f}")
    print("ticks/job       " + " ".join(f"{k:>8}" for k in keys) + "   rework")
    print("all jobs        " + " ".join(f"{allj[k] / max(n_all, 1):8.0f}" for k in keys) + f" {rw_all / max(n_all, 1):8.0f}")
    print("tardy jobs      " + " ".join(f"{tardy[k] / max(n_tardy, 1):8.0f}" for k in keys) + f" {rw_tardy / max(n_tardy, 1):8.0f}")
    tot = sum(modes.values()) or 1
    print("machine mode share:", {k: round(v / tot, 2) for k, v in modes.items()})


main(sys.argv[1:])
