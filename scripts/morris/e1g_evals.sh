#!/bin/zsh
# Run E1g evaluation configs plus bd_random evals for E1/E1a, 6 at a time, from SmartSOM-phaseE.
xargs -P 6 -n 1 /bin/zsh /Users/deng/code/SmartSOM-tools/e1_eval_one.sh < /Users/deng/code/SmartSOM-tools/v2_logs/e1g_eval_jobs.txt
echo "$(date) ALL EVALS DONE"
