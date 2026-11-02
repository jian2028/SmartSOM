#!/bin/zsh
cd /Users/deng/code/SmartSOM-integrate
T=$1; C=$2; O=$3
mkdir -p $O
uv run --no-sync smartsom evaluate $T --checkpoint $C --replications 5 --no-deterministic --output-root $O --progress off > $O.log 2>&1
echo "$(date '+%T') done $C $O"
