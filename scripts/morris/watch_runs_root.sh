#!/bin/zsh
# usage: watch_runs.sh PATTERN COUNT — wait until COUNT run dirs matching PATTERN have a final summary, or any errors.
R=${ROOT:-/Users/deng/code/SmartSOM/runs/v2}
while true; do
  for d in $(ls -d $R/* | grep -E "$1"); do
    if grep -qiE "traceback|digest changed|execution failed" $d/logs/stderr.log 2>/dev/null; then echo "ERROR $d"; tail -3 $d/logs/stderr.log | cut -c1-300; exit 0; fi
  done
  n=$(for d in $(ls -d $R/* | grep -E "$1"); do ls $d/entries/*/runs/final-evaluation/summary.json 2>/dev/null; done | wc -l)
  [ $n -ge $2 ] && { echo "done $n $(date)"; exit 0; }
  sleep 600
done
