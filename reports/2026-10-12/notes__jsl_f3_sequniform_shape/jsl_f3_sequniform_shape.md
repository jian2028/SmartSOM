# JobShopLab 式控制器 — f3_sequniform_shape

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（Justin 分支 squash、ADR 0040、central 委派 buffer／mover、action_mode 含 sequential_binary_uniform）
- 設定檔：`configs/runs/central01/jsl/f3_sequniform_shape.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/jsl/f3_sequniform_shape.yaml --background`

| 項目 | 設定 |
|---|---|
| 網路控制 | machine（選 job 開工）＋ dispatcher（AGV 目標） |
| 規則（Justin） | buffer edd、mover clearance_shortest_path、pickup priority_greedy |
| 動作格式 | sequential_binary_uniform（yes／no，初始均勻） |
| Reward | task ＋ production_potential progress 0.001（ADR 0040），gamma 1.0 |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor／critic 64×64 tanh |
| 場景／訓練 | grid、batch02_small_h1、64 件、data_seed 909；seeds 101–105、65,536 ticks、num_envs 2、validation 關閉 |
| 評估 | last checkpoint、5 cases（與先前相同）、sampling、錄影 |
| 對照 | F2（只差 shaping）、E1（只差動作格式） |

## 結果（2026-10-12 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 38, 48, 46, 36, 50 | 43.6 |
| 102 | 48, 31, 32, 36, 44 | 38.2 |
| 103 | 29, 23, 14, 20, 21 | 21.4 |
| 104 | 35, 34, 34, 32, 40 | 35.0 |
| 105 | 7, 64, 2, 11, 33 | 23.4 |
| **5 seeds** | | **32.3 ± 9.6**（task return 0.374 ± 0.050；完成 64 件的案例 1/25） |
| 對照 | | F2（無 shaping）：26.5 ± 7.1；E1（softmax＋shaping）：25.7 ± 11.0 |
- 執行目錄：`runs/central-01/20261012T125551Z-jsl_f3_sequniform_shape-b49ab642e7`（啟動 2026-10-12 05:55）
