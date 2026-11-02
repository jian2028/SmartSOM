#!/bin/zsh
# disk_watchdog.sh MIN_FREE_GB RUN_DIR: stop the run if free disk drops below MIN_FREE_GB.
MIN=$1; RUN=$2
cd /Users/deng/code/SmartSOM-integrate2
while true; do
  done_=$(python3 -c "
import json,sys
try: p=json.load(open('$RUN/logs/progress.json'))
except Exception: sys.exit(1)
print(int(bool(p.get('tasks')) and all(t['status'] in ('completed','failed','interrupted','stopped','force_stopped') for t in p['tasks'])))" 2>/dev/null)
  [ "$done_" = "1" ] && { echo "$(date '+%F %T') run finished; watchdog exiting"; exit 0; }
  free=$(df -g /Users/deng | tail -1 | awk '{print $4}')
  if [ "$free" -lt "$MIN" ]; then
    echo "$(date '+%F %T') free ${free}GB < ${MIN}GB: stopping $RUN"
    uv run --no-sync smartsom stop "$RUN" --timeout 120
    exit 1
  fi
  sleep 60
done
