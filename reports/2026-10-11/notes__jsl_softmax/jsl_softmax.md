# JobShopLab 式控制器 — softmax

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（Justin 分支 squash、ADR 0040 實作、central 委派 buffer／mover、action_mode）
- 設定檔：`configs/runs/central01/jsl/e_softmax.yaml`、`configs/algorithms/central01_jsl_softmax.yaml`
- 指令：`uv run --no-sync smartsom run configs/runs/central01/jsl/e_softmax.yaml --background`

| 項目 | 設定 |
|---|---|
| 網路控制 | machine（選 job 開工）＋ dispatcher（AGV 目標） |
| 規則（Justin） | buffer edd、mover clearance_shortest_path、pickup priority_greedy |
| 動作格式 | softmax（從全部候選挑一個） |
| 演算法 | Central PPO（RLlib），lr 3e-4、GAE 0.95、clip 0.2、entropy 0.01、batch 64、4 epochs，actor／critic 64×64 tanh |
| Reward | task（0.5／0.3／0.2，ref 128／4096）＋ production_potential progress 0.001（ADR 0040），gamma 1.0 |
| 場景 | grid、batch02_small_h1、64 件、data_seed 909、tick_limit 4096 |
| Seeds／訓練 | 101–105，65,536 ticks，num_envs 2，max_concurrent 5，validation 關閉 |
| 評估 | last checkpoint，5 cases（與 Step 1／3 相同），sampling |
| Smoke 驗證 | 路由正確（buffer／mover 為規則）、Justin mover 0 碰撞、端到端完成 |

## 結果（2026-10-11 21:27 完成）

| Seed | 交付（cases 0–4） | 平均 |
|---|---|---|
| 101 | 12, 22, 15, 7, 13 | 13.8 |
| 102 | 26, 19, 7, 26, 19 | 19.4 |
| 103 | 49, 14, 12, 24, 52 | 30.2 |
| 104 | 5, 34, 47, 27, 2 | 23.0 |
| 105 | 51, 45, 42, 43, 30 | 42.2 |
| **5 seeds** | | **25.7 ± 11.0**（task return 0.347 ± 0.050） |
| Step 3a（網路管 machine／buffer／dispatcher，我的 mover） | | 43.5 ± 8.6（0.422 ± 0.037） |

比 Step 3a 低約 18 件；變異來自 seed 與案例兩者，沒有單一壞案例。
E1 相對 Step 3a 同時改了 4 件事（buffer 交給 edd、mover 換成 Justin 的、pickup 改 priority_greedy、網路不管 buffer），無法單獨歸因。
- 執行目錄：`runs/central-01/20261012T020139Z-jsl_softmax-d55ad22d47`（啟動 2026-10-11 19:01）
- W&B：101 [o4mxmz7a](https://wandb.ai/creeps-usc/SmartSOM2026/runs/o4mxmz7a)、102 [lxlfomep](https://wandb.ai/creeps-usc/SmartSOM2026/runs/lxlfomep)、103 [4nte0a8a](https://wandb.ai/creeps-usc/SmartSOM2026/runs/4nte0a8a)、104 [3w392sy0](https://wandb.ai/creeps-usc/SmartSOM2026/runs/3w392sy0)、105 [475buxlo](https://wandb.ai/creeps-usc/SmartSOM2026/runs/475buxlo)
