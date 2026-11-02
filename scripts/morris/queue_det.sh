#!/bin/zsh
T=/Users/deng/code/SmartSOM-tools
until grep -q "finished r5_h1_exclusive_long" $T/launch_chain_tonight.log; do sleep 120; done
echo "$(date '+%F %T') R5-long finished; starting deterministic eval"
cd $T && cat det_jobs.txt | xargs -P 5 -L 1 zsh det_eval.sh
echo "$(date '+%F %T') deterministic eval finished"
