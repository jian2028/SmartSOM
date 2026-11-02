#!/bin/zsh
# Run the v2 rule baselines in parallel from integrate3; append a line per finished run.
cd /Users/deng/code/SmartSOM-integrate3
L=/Users/deng/code/SmartSOM-tools/v2_logs
for n in "$@"; do
  ( uv run --no-sync smartsom run configs/runs/central01/v2/$n.yaml --progress off > $L/$n.log 2>&1
    echo "$(date -u +%H:%M) $n exit=$?" >> $L/rules_done.txt ) &
done
wait
