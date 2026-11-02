# Step 1 — Central PPO（拆 mover）＋ task reward（無 shaping）

- 啟動時間：2026-10-10 19:20 PDT
- 程式版本：main @ e4a6ab2（含未 commit 的本地修改：central+規則 mover、central DQN、keep_continuations、traffic rules、production shaping）
- 設定檔：`configs/runs/central01/split/step1_task.yaml`、`configs/algorithms/central01_split_task.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/split/step1_task.yaml --background`

| 項目 | 設定 |
|---|---|
| 工廠 | batch02_small_h1（19×9、8 機器、8 AGV），grid 運輸，tick_limit 4096 |
| 工作量 | batch02_small_low（64 件），data_seed 909 |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor/critic 64×64 tanh，max_jobs 64 |
| 角色 | central：machine／buffer／dispatcher；mover：規則 `shortest_path_yield` |
| Reward | task（shipment 0.5／passing 0.3／tardiness 0.2，ref 128／4096），gamma 1.0，learner_scale 1.0，無 shaping |
| 訓練 seeds | 101, 102, 103, 104, 105 |
| 訓練長度 | 65,536 ticks（256 updates × 256 ticks），num_envs 2，sampling_processes 2，max_concurrent 5 |
| Validation | 關閉 |
| 評估 | last checkpoint，5 cases（data_seed 909 推導，所有 seed 相同），sampling，錄影，無 full_replay |
| 存檔 | 每 64 updates，keep_continuations 2 |
| 參考值 | grid rule baseline：64/64，平均 makespan ~2,640 ticks |

## 結果（2026-10-10 22:12 完成）

評估：last checkpoint，5 cases／seed，sampling，tick_limit 4,096（沒有 episode 在時間窗內完成 64 件，所以沒有 makespan）。

| Seed | 交付（5 cases） | 平均交付 | 平均 task return | 平均 passing rate |
|---|---|---|---|---|
| 101 | 11, 6, 5, 9, 9 | 8.0 | 0.254 | 0.92 |
| 102 | 40, 41, 40, 49, 37 | 41.4 | 0.414 | 0.96 |
| 103 | 22, 14, 13, 7, 11 | 13.4 | 0.279 | 0.92 |
| 104 | 43, 40, 49, 42, 48 | 44.4 | 0.426 | 0.95 |
| 105 | 39, 40, 41, 42, 39 | 40.2 | 0.408 | 0.95 |
| **5 seeds** | | **29.5 ± 17.3** | **0.356 ± 0.083** | |
| grid rule baseline | | 64 | ~0.53–0.55 | ~0.97 |

（± 為 5 個 seed 平均值的標準差。）

**解讀：**
- 拆 mover 後 central PPO 首次學會送貨：之前所有 central 設定每 episode 只有 0–3 件。
- 5 個 seed 分成兩群：102／104／105 約 40–44 件（約 rule 的 2/3），101／103 卡在 8–13 件。
- Passing rate 高（≥0.92），瓶頸在出貨量。
- 這是 reward shaping 對照的基線（Jian 要求）：只用 task reward、沒有 shaping。
- 執行目錄：`runs/central-01/20261011T022103Z-step1_split_task-7f5d681e14`
- W&B：seed 101 [mnkkkmhy](https://wandb.ai/creeps-usc/SmartSOM2026/runs/mnkkkmhy)、102 [pkl0welr](https://wandb.ai/creeps-usc/SmartSOM2026/runs/pkl0welr)、103 [yxt05t45](https://wandb.ai/creeps-usc/SmartSOM2026/runs/yxt05t45)、104 [6msbfeg1](https://wandb.ai/creeps-usc/SmartSOM2026/runs/6msbfeg1)、105 [2y9d6hmf](https://wandb.ai/creeps-usc/SmartSOM2026/runs/2y9d6hmf)
