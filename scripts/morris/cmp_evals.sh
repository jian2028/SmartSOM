#!/bin/zsh
# Run C1y and Justin+yield bd_random evals, 6 at a time, from SmartSOM-phaseE.
xargs -P 6 -n 1 /bin/zsh /Users/deng/code/SmartSOM-tools/e1_eval_one.sh < /Users/deng/code/SmartSOM-tools/v2_logs/cmp_eval_jobs.txt
echo "$(date) ALL EVALS DONE"
