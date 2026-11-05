#!/bin/zsh
cd /Users/deng/code/SmartSOM-phaseE
n=$(basename $1 .yaml); L=/Users/deng/code/SmartSOM-tools/v2_logs/e1eval; mkdir -p $L
env -u OUTAGE_PROBE -u COMMIT_PROBE .venv/bin/smartsom run $1 --progress off > $L/$n.log 2>&1
echo "$(date) $n exit=$?"
