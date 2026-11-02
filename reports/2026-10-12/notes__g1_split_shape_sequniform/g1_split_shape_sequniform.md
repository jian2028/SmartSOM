# G1 — Step 3a 設定 ＋ yes／no（sequential_binary_uniform）

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改
- 設定檔：`configs/runs/central01/split/g1_task_shape_sequniform.yaml`、`configs/algorithms/central01_split_task_shape_sequniform.yaml`
- 與 Step 3a 唯一差別：`action_mode: sequential_binary_uniform`（diff 已確認）

| 項目 | 設定 |
|---|---|
| 網路控制 | machine＋buffer＋dispatcher；mover 規則 shortest_path_yield；pickup global_optimal |
| 動作格式 | sequential_binary_uniform（逐條 yes／no，初始均勻） |
| Reward | task（0.5／0.3／0.2）＋ production_potential progress 0.001（ADR 0040），gamma 1.0 |
| 場景／訓練 | grid、batch02_small_h1、64 件、data_seed 909；seeds 101–105、65,536 ticks、num_envs 2、validation 關閉 |
| 評估 | last checkpoint、5 cases（與先前相同）、sampling、錄影 |
| Smoke | 路由正確（buffer 由網路、mover 由規則）、action_mode 與 task reward 正確；2,048 ticks 後交付 20 |
| 對照 | Step 3a（softmax）43.5 ± 8.6 |

## 結果（2026-10-12 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 45, 34, 40, 43, 39 | 40.2 |
| 102 | 53, 56, 53, 53, 53 | 53.6 |
| 103 | 16, 11, 9, 5, 10 | 10.2 |
| 104 | 21, 14, 9, 26, 33 | 20.6 |
| 105 | 6, 3, 6, 3, 5 | 4.6 |
| **5 seeds** | | **25.8 ± 20.6**（task return 0.341 ± 0.106；完成 64 件的案例 0/25） |
| Step 3a（只差 softmax） | | 43.5 ± 8.6（0.422 ± 0.037） |
- 執行目錄：`runs/central-01/20261012T173228Z-g1_split_shape_sequniform-9dad9725cd`（啟動 2026-10-12 10:32）
