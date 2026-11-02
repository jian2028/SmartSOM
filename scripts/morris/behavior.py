"""Behaviour metrics of one recorded episode (same as the Demo slide analysis)."""
import collections
import json
import sys


def analyze(path):
    busy = collections.Counter(); ticks = 0
    hist = collections.defaultdict(list); moves = osc = 0
    swarm_ticks = 0; worst_swarm = (0, None, None)
    raw_peak = 0; pick_bad = pick_total = 0
    last_t, prev_t, flips = {}, {}, collections.Counter()
    loaded_switch = 0
    drops = collections.Counter(); prev_load = {}
    pre = {f"buffer_{2*k+1:03d}": f"machine_{k:03d}" for k in range(1, 9)}
    with open(path) as f:
        for line in f:
            r = json.loads(line); b = r["boundary_state"]; t = b["tick"]; ticks += 1
            for m, s in b["machines"].items():
                busy[m] += s["job"] is not None
            sup = b["source_supply"]
            raw_peak = max(raw_peak, len(sup.get("buffer_001", {}).get("ready", [])))
            for d in r["decisions"]:
                if d["role"] != "dispatcher":
                    continue
                a = d["owner"]; st = b["agvs"][a]; owner = d["proposal"]["owner"]
                if st["job"] is None:
                    pick_total += 1
                    legal = [c for c in d["candidates"] if c["legal"]]
                    if any(sup.get(c["action"]["owner"], {}).get("ready") for c in legal) and not sup.get(owner, {}).get("ready"):
                        pick_bad += 1
                elif st["target"] and st["target"]["owner"] != owner:
                    loaded_switch += 1
                if a in last_t and last_t[a] != owner:
                    if prev_t.get(a) == owner:
                        flips[a] += 1
                    prev_t[a] = last_t[a]
                last_t[a] = owner
            tgt = collections.Counter(s["target"]["owner"] for s in b["agvs"].values() if s["job"] is None and s["target"])
            for o, k in tgt.items():
                s = sup.get(o, {}); avail = max(len(s.get("ready", [])), s.get("supply", 0))
                if k >= 2 and k > max(avail, 1):
                    swarm_ticks += 1
                    if k > worst_swarm[0]:
                        worst_swarm = (k, t, o)
                    break
            for a, s in b["agvs"].items():
                c = tuple(s["cell"]); h = hist[a]
                if not h or h[-1] != c:
                    h.append(c); moves += len(h) > 1; osc += len(h) >= 3 and h[-3] == c
                k = prev_load.get(a)
                if k and k[0] is not None and s["job"] != k[0] and k[1] in pre:
                    drops[pre[k[1]]] += 1
                prev_load[a] = (s["job"], s["target"]["owner"] if s["target"] else None)
    twins = [("machine_001", "machine_008"), ("machine_002", "machine_007"), ("machine_003", "machine_006"), ("machine_004", "machine_005")]
    return {
        "ticks": ticks,
        "swarm_pct": round(100 * swarm_ticks / ticks, 1),
        "worst_swarm": worst_swarm,
        "raw_peak": raw_peak,
        "bad_pick_pct": round(100 * pick_bad / max(pick_total, 1), 1),
        "reversal_pct": round(100 * osc / max(moves, 1), 1),
        "target_flips_per_agv": round(sum(flips.values()) / 8, 1),
        "loaded_switches": loaded_switch,
        "twin_util": [(f"{a[-1]}/{b[-1]}", f"{100*busy[a]//ticks}%/{100*busy[b]//ticks}%") for a, b in twins],
        "twin_drops": [(f"{a[-1]}/{b[-1]}", f"{drops[a]}/{drops[b]}") for a, b in twins],
    }


for p in sys.argv[1:]:
    print(p.split("/replays/")[1].split("/")[0], json.dumps(analyze(p), ensure_ascii=False))
