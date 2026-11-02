#!/bin/zsh
cd /Users/deng/code/SmartSOM-integrate2
n=$(basename $1 .yaml)
uv run --no-sync smartsom run $1 --progress off > /Users/deng/code/SmartSOM-tools/det_logs/$n.log 2>&1
echo "$(date '+%T') done $n $(grep -o '"status": "completed"' /Users/deng/code/SmartSOM-tools/det_logs/$n.log | wc -l)"
