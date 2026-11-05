#!/bin/zsh
# Launch E1g once E1a's final evaluation has all 5 summaries.
R=/Users/deng/code/SmartSOM/runs/phaseE/20261102T061812Z-e1a_train-980e63f087
setopt nullglob
until [ $(ls $R/entries/*/runs/final-evaluation/summary.json | wc -l) -ge 5 ]; do sleep 60; done
echo "$(date) E1a done; launching E1g"
cd /Users/deng/code/SmartSOM-phaseE && env -u OUTAGE_PROBE -u COMMIT_PROBE .venv/bin/smartsom run configs/runs/central01/v2/e1g_train.yaml --progress off
echo "$(date) E1g launcher exit=$?"
