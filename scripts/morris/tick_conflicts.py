"""Count same-tick dispatch conflicts in a recorded v3 trace.

Empty AGVs: a pick of source S conflicts when, counting vehicles already
heading to S (not re-decided this tick) plus earlier picks this tick, S has
no supply left. Loaded AGVs: a pick of a capacity-limited destination
conflicts when its free slots are already booked the same way.
"""
import collections
import json
import sys


def count(path):
    stats = collections.Counter()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            picks = [d for d in row["decisions"] if d["role"] == "dispatcher"]
            if not picks:
                continue
            view = picks[0]["observation"]
            agvs, sources = view["agvs"], view.get("sources", {})
            decided = {d["owner"] for d in picks}
            taken = collections.Counter()
            per_target = collections.Counter()
            for d in picks:
                loaded = agvs[d["owner"]]["job"] is not None
                owner = d["proposal"]["owner"]
                heading = sum(
                    k not in decided
                    and (s["job"] is not None) == loaded
                    and bool(s["target"])
                    and s["target"]["owner"] == owner
                    for k, s in agvs.items()
                )
                if loaded:
                    cand = next(c for c in d["candidates"] if c["identity"] == d["candidate"])
                    f = cand["features"]
                    if not f[9]:
                        stats["loaded_picks"] += 1
                        continue
                    left = round(f[8] * 100) - round(f[7] * 100)
                    kind = "loaded"
                else:
                    s = sources.get(owner, {})
                    left = max(len(s.get("ready", [])), s.get("supply", 0))
                    kind = "empty"
                stats[f"{kind}_picks"] += 1
                if not loaded:
                    def free(o):
                        src = sources.get(o, {})
                        h = sum(
                            k not in decided and s2["job"] is None and bool(s2["target"])
                            and s2["target"]["owner"] == o
                            for k, s2 in agvs.items()
                        )
                        return max(len(src.get("ready", [])), src.get("supply", 0)) - h - taken[("empty", o)]
                    if not any(free(c["action"]["owner"]) > 0 for c in d["candidates"] if c["legal"]):
                        stats["empty_forced"] += 1
                if heading + taken[(kind, owner)] >= left:
                    stats[f"{kind}_conflicts"] += 1
                    if not loaded and any(
                        free(c["action"]["owner"]) > 0 for c in d["candidates"] if c["legal"]
                    ):
                        stats["empty_avoidable"] += 1
                    if taken[(kind, owner)] > 0:
                        stats[f"{kind}_same_tick_conflicts"] += 1
                taken[(kind, owner)] += 1
                per_target[(kind, owner)] += 1
            stats["ticks_with_dispatch"] += 1
            stats["ticks_two_plus_same_target"] += any(v >= 2 for v in per_target.values())
    return stats


for path in sys.argv[1:]:
    s = count(path)
    print(path.split("/")[-4][:40])
    for kind in ("empty", "loaded"):
        n = s[f"{kind}_picks"]
        print(f"  {kind}: picks {n}, conflicts {s[f'{kind}_conflicts']} ({100*s[f'{kind}_conflicts']/max(n,1):.1f}%), "
              f"of which same-tick {s[f'{kind}_same_tick_conflicts']}")
    print(f"  empty picks with no open source at all (forced): {s['empty_forced']}; avoidable empty conflicts: {s['empty_avoidable']}")
    print(f"  ticks with 2+ vehicles picking the same target: {s['ticks_two_plus_same_target']} / {s['ticks_with_dispatch']}")
