#!/bin/zsh
# Extra Justin+yield seeds (102-105) for balanced / bdm3 / bd_random, 6 at a time.
xargs -P 6 -n 1 /bin/zsh /Users/deng/code/SmartSOM-tools/e1_eval_one.sh < /Users/deng/code/SmartSOM-tools/v2_logs/justin_seed_jobs.txt
echo "$(date) ALL EVALS DONE"
