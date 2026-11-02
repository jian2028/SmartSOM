# Step 3a — Central PPO（拆 mover）＋ task reward ＋ progress shaping（progress_weight 0.001）

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（含 Justin 分支 squash 合併、ADR 0040 本地實作）
- 設定檔：`configs/runs/central01/split/step3a_task_shape.yaml`（與 Step 1 只差 algorithm 檔的 reward.team）
- 指令：`uv run --no-sync smartsom run configs/runs/central01/split/step3a_task_shape.yaml --background`

| 項目 | 設定 |
|---|---|
| 與 Step 1 相同 | 工廠、64 件工作量、data_seed 909、grid、seeds 101–105、65,536 ticks、num_envs 2、validation 關閉、評估 last ckpt 5 cases sampling、keep_continuations 2、mover 規則 shortest_path_yield |
| Reward | task（0.5／0.3／0.2，ref 128／4096），gamma 1.0 ＋ **builtin.production_potential**（ADR 0040，PBRS） |
| Shaping 參數 | weight 0、empty_weight 0（移動交給規則，不 shape 距離）、progress_weight **0.001**、gamma 1.0 |
| 意義 | progress 位能 = 0.001 ×（已出貨 − 未出貨工單剩餘工序）；每完成一道工序或出貨 +0.001（一件出貨的 task 價值 ≈ 0.0039） |
| 驗證 | 單 episode 探針：shaping 加總 = −Φ(s0)（誤差 7e-15）；報告的 shaping_return 與之相符；回歸測試已加 |

## 結果

| Seed | 交付（5 cases） | 平均交付 | 平均 task return | 平均 passing |
|---|---|---|---|---|
| 101 | 36, 37, 42, 43, 42 | 40.0 | 0.404 | 0.95 |
| 102 | 51, 49, 52, 49, 54 | 51.0 | 0.458 | 0.95 |
| 103 | 45, 48, 48, 44, 46 | 46.2 | 0.435 | 0.96 |
| 104 | 33, 33, 28, 29, 28 | 30.2 | 0.368 | 0.97 |
| 105 | 36, 54, 51, 58, 51 | 50.0 | 0.447 | 0.94 |
| **5 seeds** | | **43.5 ± 8.6** | **0.422 ± 0.037** | |
| Step 1（無 shaping） | | 29.5 ± 17.3 | 0.356 ± 0.083 | |
- 執行目錄：`runs/central-01/20261011T063045Z-step3a_split_task_shape-96aff2a8be`（啟動 2026-10-10 23:30）
- W&B：101 [hldcjovj](https://wandb.ai/creeps-usc/SmartSOM2026/runs/hldcjovj)、102 [k6eqt4kq](https://wandb.ai/creeps-usc/SmartSOM2026/runs/k6eqt4kq)、103 [79kujnrj](https://wandb.ai/creeps-usc/SmartSOM2026/runs/79kujnrj)、104 [xpg9prne](https://wandb.ai/creeps-usc/SmartSOM2026/runs/xpg9prne)、105 [6jpfo7sd](https://wandb.ai/creeps-usc/SmartSOM2026/runs/6jpfo7sd)
