#!/bin/zsh
# launch_chain.sh cfg:label ...  Launch each config after the previous run finishes.
cd /Users/deng/code/SmartSOM-integrate
S=/Users/deng/code/SmartSOM-tools
for spec in "$@"; do
  cfg=${spec%%:*}; label=${spec##*:}
  R=$(uv run --no-sync smartsom run $cfg --background 2>&1 | grep -o '/Users/deng/code/SmartSOM/runs/central-01/[^/"]*' | head -1)
  echo "$(date '+%F %T') launched $label $R"
  [ -z "$R" ] && { echo "launch failed for $label"; exit 1; }
  (cd $S && nohup /Users/deng/code/SmartSOM/.venv/bin/python wandb_bridge.py $R --label $label > bridges/$label.log 2>&1 &)
  sleep 20
  P=$(pgrep -f "smartsom.experiments.background.*$(basename $R)" | head -1)
  [ -n "$P" ] && (nohup caffeinate -i -w $P >/dev/null 2>&1 &)
  (nohup $S/disk_watchdog_i.sh 50 $R > $S/disk_watchdog_$label.log 2>&1 &)
  until python3 -c "
import json, sys
p = json.load(open('$R/logs/progress.json'))
sys.exit(0 if p.get('tasks') and all(t['status'] in ('completed','failed','interrupted','stopped','force_stopped') for t in p['tasks']) else 1)" 2>/dev/null; do sleep 120; done
  echo "$(date '+%F %T') finished $label"
done
