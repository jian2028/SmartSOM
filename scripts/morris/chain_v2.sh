#!/bin/zsh
# Rule baselines first, then the C1 central training (avoids CPU contention).
T=/Users/deng/code/SmartSOM-tools
$T/launch_v2_rules.sh rules_justin_low rules_justin_balanced rules_justin_low_bdm3 rules_justin_balanced_bdm3 rules_random_clearance_low rules_random_clearance_balanced
cd /Users/deng/code/SmartSOM-integrate3
uv run --no-sync smartsom run configs/runs/central01/v2/c1_train.yaml --progress off > $T/v2_logs/c1_train.log 2>&1
echo "$(date -u +%H:%M) c1_train exit=$?" >> $T/v2_logs/rules_done.txt
