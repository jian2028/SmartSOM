#!/bin/zsh
cd /Users/deng/code/SmartSOM-integrate3; L=/Users/deng/code/SmartSOM-tools/v2_logs
for f in configs/runs/central01/v2/c1yt4eval/c1yt4_*.yaml; do
  n=$(basename $f .yaml)
  ( uv run --no-sync smartsom run $f --progress off > $L/$n.log 2>&1
    echo "$(date -u +%H:%M) $n exit=$?" >> $L/rules_done.txt ) &
done
wait
