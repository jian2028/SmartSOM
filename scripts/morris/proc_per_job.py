"""Machine processing ticks per delivered job (completed episodes): proxy for quality-mode choice."""
import glob, json, sys, statistics as st
for run in sys.argv[1:]:
    xs, down = [], []
    for p in glob.glob(f"{run}/**/evidence/case-*/result.json", recursive=True):
        r = json.load(open(p)); m = r.get("evaluation_metrics")
        if not m or r["status"] != "completed": continue
        s = m["utilisation"]["machine_states"]
        xs.append(sum(c.get("processing", 0) for c in s.values()) / r["delivered"])
    print(run.rstrip("/").split("/")[-1][17:50], len(xs), round(st.mean(xs), 1) if xs else None)
