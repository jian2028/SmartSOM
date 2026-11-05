#!/bin/zsh
# Run E1/E1a evaluation configs, 4 at a time, from SmartSOM-phaseE.
xargs -P 4 -n 1 /bin/zsh /Users/deng/code/SmartSOM-tools/e1_eval_one.sh < /Users/deng/code/SmartSOM-tools/v2_logs/e1_eval_jobs.txt
echo "$(date) ALL EVALS DONE"
