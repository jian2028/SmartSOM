# Step 3b — Central PPO（拆 mover）＋ task reward ＋ progress shaping（progress_weight 0.004）

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（含 Justin 分支 squash 合併、ADR 0040 本地實作）
- 設定檔：`configs/runs/central01/split/step3b_task_shape.yaml`（與 Step 1 只差 algorithm 檔的 reward.team）
- 指令：`uv run --no-sync smartsom run configs/runs/central01/split/step3b_task_shape.yaml --background`

| 項目 | 設定 |
|---|---|
| 與 Step 1 相同 | 工廠、64 件工作量、data_seed 909、grid、seeds 101–105、65,536 ticks、num_envs 2、validation 關閉、評估 last ckpt 5 cases sampling、keep_continuations 2、mover 規則 shortest_path_yield |
| Reward | task（0.5／0.3／0.2，ref 128／4096），gamma 1.0 ＋ **builtin.production_potential**（ADR 0040，PBRS） |
| Shaping 參數 | weight 0、empty_weight 0（移動交給規則，不 shape 距離）、progress_weight **0.004**、gamma 1.0 |
| 意義 | progress 位能 = 0.004 ×（已出貨 − 未出貨工單剩餘工序）；每完成一道工序或出貨 +0.004（一件出貨的 task 價值 ≈ 0.0039） |
| 驗證 | 單 episode 探針：shaping 加總 = −Φ(s0)（誤差 7e-15）；報告的 shaping_return 與之相符；回歸測試已加 |

## 結果

| Seed | 交付（5 cases） | 平均交付 | 平均 task return | 平均 passing |
|---|---|---|---|---|
| 101 | 52, 55, 45, 26, 52 | 46.0 | 0.438 | 0.95 |
| 102 | 46, 34, 33, 43, 38 | 38.8 | 0.393 | 0.94 |
| 103 | 45, 52, 49, 50, 47 | 48.6 | 0.435 | 0.92 |
| 104 | 25, 19, 18, 29, 24 | 23.0 | 0.329 | 0.92 |
| 105 | 58, 63, 30, 54, 59 | 52.8 | 0.462 | 0.94 |
| **5 seeds** | | **41.8 ± 11.7** | **0.411 ± 0.052** | |
| Step 1（無 shaping） | | 29.5 ± 17.3 | 0.356 ± 0.083 | |
- 執行目錄：`runs/central-01/20261011T084731Z-step3b_split_task_shape-4029505a5b`（啟動 2026-10-11 01:47）
