#!/bin/zsh
# After C1 training: softmax + argmax, without and with the m3 outage, per model.
# Eval seed stays 101 so the cases match the rule baselines (paired by replication).
I=/Users/deng/code/SmartSOM-integrate3; R=$(ls -d /Users/deng/code/SmartSOM/runs/v2/*-c1_train-*); L=/Users/deng/code/SmartSOM-tools/v2_logs
cd $I; mkdir -p configs/algorithms/v2eval configs/runs/central01/v2/c1eval
i=0
for e in $R/entries/entry-000{1..5}; do
  i=$((i+1)); seed=$((100+i))
  src=$(ls -d $e/runs/*/checkpoints/update-001024/controllers/central)
  alg=configs/algorithms/v2eval/c1_seed$seed.yaml
  sed '/^controller:/,$d' configs/algorithms/central01_v2_c1.yaml > $alg
  printf 'controller:\n  kind: model\n  model:\n    source: %s\n' $src >> $alg
  for v in "balanced samp batch02_small_h1 false" "balanced det batch02_small_h1 true" "bdm3 samp batch02_small_h1_bd_m3 false" "bdm3 det batch02_small_h1_bd_m3 true"; do
    set -- ${=v}; n=c1_${1}_${2}_seed$seed
    sed -e "s#central01_rules_justin.yaml#v2eval/c1_seed$seed.yaml#" -e "s#batch02_small_h1_bd_m3.yaml#$3.yaml#" \
        -e "s#replications: 20#replications: 20\n  deterministic: $4#" \
        -e "s#name: rules_justin_balanced_bdm3#name: $n#" configs/runs/central01/v2/rules_justin_balanced_bdm3.yaml \
      | sed 's/\\n/\n/' > configs/runs/central01/v2/c1eval/$n.yaml
    ( uv run --no-sync smartsom run configs/runs/central01/v2/c1eval/$n.yaml --progress off > $L/$n.log 2>&1
      echo "$(date -u +%H:%M) $n exit=$?" >> $L/rules_done.txt ) &
  done
done
wait
