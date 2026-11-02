#!/bin/zsh
# Live progress of the re-evaluation (refreshes every 30 s; Ctrl+C to quit).
setopt nullglob
T=/Users/deng/code/SmartSOM-tools
M=/Users/deng/code/SmartSOM/runs/reeval/models
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-22 18:19" +%s)
while true; do
  clear
  now=$(date +%s)
  eps=$(find $M -name result.json 2>/dev/null | wc -l | tr -d ' ')
  jobs=$(grep -c ' done ' $T/reeval2_all.log 2>/dev/null)
  echo "SmartSOM 重新評估進度    $(date '+%m/%d %H:%M:%S')"
  echo "=================================================="
  echo "模型評估：$jobs / 35 完成      episodes：$eps / 700"
  if [ $eps -gt 0 ]; then
    rate=$(( eps * 3600 / (now - START) ))
    left=$(( (700 - eps) * (now - START) / eps ))
    echo "速度：約 $rate episodes/小時   預計剩餘：約 $(( left / 3600 )) 小時 $(( left % 3600 / 60 )) 分"
  fi
  echo
  printf "%-16s %s\n" "模型" "已完成 episodes（每個 seed 20 場）"
  for m in untrained_ab untrained_excl r2 r2long r4 r4long r5; do
    line=""
    for s in 101 102 103 104 105; do
      n=$(find $M/*-${m}_seed${s}-* -name result.json 2>/dev/null | wc -l | tr -d ' ')
      line="$line  s$s:$(printf '%2d' $n)"
    done
    printf "%-16s%s\n" $m "$line"
  done
  echo
  for r in justin random; do
    d=$(ls -d /Users/deng/code/SmartSOM/runs/reeval/*rules_${r}_yield* 2>/dev/null | tail -1)
    n=$( [ -n "$d" ] && find $d -name result.json | wc -l | tr -d ' ' || echo 0)
    printf "規則 %-11s %2d / 20\n" $r $n
  done
  echo
  pgrep -f "xargs -P 11" >/dev/null && echo "狀態：執行中" || echo "狀態：全部模型評估已結束"
  pgrep -f caffeinate >/dev/null && echo "caffeinate：開啟（不會自動休眠，但請不要闔上筆電）" || echo "caffeinate：未開啟！"
  echo "\n（每 30 秒更新；按 Ctrl+C 離開，不影響評估）"
  sleep 30
done
