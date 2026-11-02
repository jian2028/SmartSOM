# JobShopLab 式控制器 — f1_softmax_base

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（Justin 分支 squash、ADR 0040、central 委派 buffer／mover、action_mode 含 sequential_binary_uniform）
- 設定檔：`configs/runs/central01/jsl/f1_softmax_base.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/jsl/f1_softmax_base.yaml --background`

| 項目 | 設定 |
|---|---|
| 網路控制 | machine（選 job 開工）＋ dispatcher（AGV 目標） |
| 規則（Justin） | buffer edd、mover clearance_shortest_path、pickup priority_greedy |
| 動作格式 | softmax |
| Reward | task only（0.5／0.3／0.2），gamma 1.0，無 shaping |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor／critic 64×64 tanh |
| 場景／訓練 | grid、batch02_small_h1、64 件、data_seed 909；seeds 101–105、65,536 ticks、num_envs 2、validation 關閉 |
| 評估 | last checkpoint、5 cases（與先前相同）、sampling、錄影 |
| 對照 | E1（同設定＋shaping） |

## 結果（2026-10-12 03:14 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 52, 9, 0, 55, 2 | 23.6 |
| 102 | 2, 3, 5, 1, 8 | 3.8 |
| 103 | 0, 23, 36, 5, 17 | 16.2 |
| 104 | 11, 14, 8, 8, 12 | 10.6 |
| 105 | 1, 4, 7, 1, 2 | 3.0 |
| **5 seeds** | | **11.4 ± 8.7**（task return 0.259 ± 0.011） |
| E1（同設定＋shaping） | | 25.7 ± 11.0（0.347 ± 0.050） |

Shaping 讓 JobShopLab 範圍的 softmax 版本從 11.4 提升到 25.7，與 Step 1 → Step 3a 的趨勢一致。
- 執行目錄：`runs/central-01/20261012T074413Z-jsl_f1_softmax_base-0d349e6d12`（啟動 2026-10-12 00:44）
