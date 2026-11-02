# Grid rule baseline — Justin 的規則（two-sided AGV controller，本地 squash 合併）

- 設定檔：`configs/runs/central01/rule_grid_justin.yaml`、`configs/algorithms/central01_rules_justin.yaml`
- 規則：machine normal_first、buffer edd、dispatcher nearest（fleet_admission: traffic、work_in_progress_first: true）、mover clearance_shortest_path、pickup priority_greedy
- 場景：batch02_small_h1、64 件、data_seed 909（與 Step 1／3 相同的評估案例）、grid、tick_limit 16384
- 執行目錄：`runs/central-01/20261011T060451Z-rule_grid_justin-f5ddf17742`

| Case | 結果 | 交付 | Makespan | Passing |
|---|---|---|---|---|
| 0 | completed | 64 | 2,487 | 1.0 |
| 1 | **truncated（卡在 63/64 直到 16,384）** | 63 | — | 1.0 |
| 2 | completed | 64 | 2,346 | 1.0 |
| 3 | completed | 64 | 2,566 | 1.0 |
| 4 | completed | 64 | 2,500 | 1.0 |

完成的 4 個案例平均 makespan 2,475（我的 `nearest_sticky`＋`shortest_path_yield` 為 2,641），但 case 1 有一件始終沒出貨，需回報 Justin 並用 replay 檢查。
