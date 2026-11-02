# JobShopLab 式控制器 — seqbin

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（Justin 分支 squash、ADR 0040 實作、central 委派 buffer／mover、action_mode）
- 設定檔：`configs/runs/central01/jsl/e_seqbin.yaml`、`configs/algorithms/central01_jsl_seqbin.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/jsl/e_seqbin.yaml --background`

| 項目 | 設定 |
|---|---|
| 網路控制 | machine（選 job 開工）＋ dispatcher（AGV 目標） |
| 規則（Justin） | buffer edd、mover clearance_shortest_path、pickup priority_greedy |
| 動作格式 | sequential_binary（依序 yes／no，第一個 yes 執行，全部 no 選最後一個合法候選） |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor／critic 64×64 tanh |
| Reward | task（0.5／0.3／0.2，ref 128／4096）＋ production_potential progress 0.001（ADR 0040），gamma 1.0 |
| 場景 | grid、batch02_small_h1、64 件、data_seed 909、tick_limit 4096 |
| Seeds／訓練 | 101–105，65,536 ticks，num_envs 2，max_concurrent 5，validation 關閉 |
| 評估 | last checkpoint，5 cases（與 Step 1／3 相同），sampling |
| Smoke 驗證 | 路由正確（buffer／mover 為規則）、Justin mover 0 碰撞、端到端完成 |

## 結果（2026-10-12 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 5, 3, 1, 0, 4 | 2.6 |
| 102 | 3, 8, 0, 15, 17 | 8.6 |
| 103 | 3, 4, 3, 4, 3 | 3.4 |
| 104 | 2, 4, 5, 2, 5 | 3.6 |
| 105 | 2, 1, 1, 1, 2 | 1.4 |
| **5 seeds** | | **3.9 ± 2.8**（task return 0.221 ± 0.025） |
| E1 softmax（只差動作格式） | | 25.7 ± 11.0（0.347 ± 0.050） |

**原因（已確認）：**初始分佈嚴重偏向排在前面的候選。以真實的 22 選項派車決策、未訓練網路量測：
第一個候選被選機率 53.8%、最後的候選約 0%、entropy 1.28（均勻為 3.09）。後面的候選幾乎探索不到。
**修正：**新增 `sequential_binary_uniform`（固定的 hazard 先驗，初始為均勻分佈；同一決策實測 entropy 3.08）。
原 `sequential_binary` 保持不變，這個 run 的 checkpoint 仍可重現。
- 執行目錄：`runs/central-01/20261012T042827Z-jsl_seqbin-5f154cb32d`（啟動 2026-10-11 21:28）
- W&B：101 [9cd1sijj](https://wandb.ai/creeps-usc/SmartSOM2026/runs/9cd1sijj)、102 [7e66t6zw](https://wandb.ai/creeps-usc/SmartSOM2026/runs/7e66t6zw)、103 [8lwiz0v5](https://wandb.ai/creeps-usc/SmartSOM2026/runs/8lwiz0v5)、104 [q2dgqe4a](https://wandb.ai/creeps-usc/SmartSOM2026/runs/q2dgqe4a)、105 [ihlygzlz](https://wandb.ai/creeps-usc/SmartSOM2026/runs/ihlygzlz)
