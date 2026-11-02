# JobShopLab 式控制器 — f2_sequniform_base

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（Justin 分支 squash、ADR 0040、central 委派 buffer／mover、action_mode 含 sequential_binary_uniform）
- 設定檔：`configs/runs/central01/jsl/f2_sequniform_base.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/jsl/f2_sequniform_base.yaml --background`

| 項目 | 設定 |
|---|---|
| 網路控制 | machine（選 job 開工）＋ dispatcher（AGV 目標） |
| 規則（Justin） | buffer edd、mover clearance_shortest_path、pickup priority_greedy |
| 動作格式 | sequential_binary_uniform（yes／no，初始均勻） |
| Reward | task only，gamma 1.0，無 shaping |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor／critic 64×64 tanh |
| 場景／訓練 | grid、batch02_small_h1、64 件、data_seed 909；seeds 101–105、65,536 ticks、num_envs 2、validation 關閉 |
| 評估 | last checkpoint、5 cases（與先前相同）、sampling、錄影 |
| 對照 | F1（只差動作格式） |

## 結果（2026-10-12 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 7, 27, 31, 33, 34 | 26.4 |
| 102 | 12, 47, 25, 17, 40 | 28.2 |
| 103 | 50, 16, 55, 47, 13 | 36.2 |
| 104 | 27, 44, 26, 12, 17 | 25.2 |
| 105 | 20, 19, 14, 13, 16 | 16.4 |
| **5 seeds** | | **26.5 ± 7.1**（task return 0.346 ± 0.023；完成 64 件的案例 0/25） |
| 對照 | | F1（softmax，task only）：11.4 ± 8.7 |
- 執行目錄：`runs/central-01/20261012T101502Z-jsl_f2_sequniform_base-67b42bd9b5`（啟動 2026-10-12 03:15）
