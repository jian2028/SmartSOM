# Step 3c — Step 3a ＋ mover 停車修正（park_while_target_pending）

- 程式版本：main @ e4a6ab2 ＋ 本地未 commit 修改（同 Step 3a，另加 mover 停車參數）
- 設定檔：`configs/runs/central01/split/step3c_task_shape_park.yaml`（與 Step 3a 只差 algorithm 檔）
- Algorithm：`configs/algorithms/central01_split_task_shape_park.yaml`（與 `central01_split_task_shape.yaml` 只差 mover `parameters: {park_while_target_pending: true}`）
- 指令：`uv run --no-sync smartsom run configs/runs/central01/split/step3c_task_shape_park.yaml --background`
- 狀態：完成（2026-10-12 23:28 → 2026-10-13 02:11）

## 為什麼改

Step 3a replay（seed 102 case 0）中 51% 的移動是 A→B→A 來回，其中 64% 發生在「目標 post buffer 還沒貨、但機器正在加工」時：
空車開上 port → 規定必須離開 port → 又開回來。

## 改了什麼（`shortest_path_yield`，參數預設關閉）

- 空車的目標**還沒有可取的 job，但 supply > 0**（機器加工中，貨正在來）時，停在 port 旁的非 port 格子等，貨好了才上 port。
- 目標**完全沒貨**（supply = 0）時照常上 port。原因：模擬器只在空車**抵達**空目標時才重新派車（`production_protocol.py` 的 `empty_notified`）。
  第一版連這種情況也停車，結果車永遠不會被重派，整場卡死（1/64）。
- 停車時只讓路給「載貨中或目標不同」的鄰車（機率 0.25），等同一個 port 的車不互相換位。
- 被迫離開 port 時，優先離到非 port 的格子。

## 驗證（開跑前）

- 單元測試：`tests/unit/test_grid_traffic_rules.py` 新增 8 個（停車、有貨就進、無貨仍上 port、遠處正常接近、被迫離開、鄰車讓路、參數檢查）；相關測試 118 個全過。
- `smartsom check` 通過。
- **用已訓練的 Step 3a seed 102 模型（last ckpt），只換 mover，各評估 3 個 episode（sampling，同評估 seed）：**

| Mover | 交付 | passing | task return | 總移動 | 來回移動 |
|---|---|---|---|---|---|
| 原本 | 51, 49, 52（50.7） | 0.94–0.98 | 0.459 | ~22,000 | ~11,100（51%） |
| 停車 | 55, 47, 48（50.0） | 0.91–0.98 | 0.449 | ~16,500（−25%） | ~6,550（40%，−41%） |

- 原本 mover 的 case 0 與先前 replay 完全一致（51 件、0.941），確認探針可重現。
- 來回移動少了約 4 成，但這個模型是配原本 mover 訓練的，直接換上去交付量沒有變好；要重新訓練才知道效果。

## 結果

| Seed | 交付（5 cases） | 平均交付 | 平均 task return | 平均 passing |
|---|---|---|---|---|
| 101 | 36, 28, 40, 33, 41 | 35.6 | 0.385 | 0.95 |
| 102 | 36, 34, 36, 28, 35 | 33.8 | 0.384 | 0.97 |
| 103 | 7, 7, 7, 12, 19 | 10.4 | 0.275 | 0.96 |
| 104 | 20, 3, 26, 45, 11 | 21.0 | 0.327 | 0.97 |
| 105 | 36, 32, 35, 35, 31 | 33.8 | 0.371 | 0.92 |
| **5 seeds** | | **26.9 ± 10.9** | **0.348 ± 0.047** | |
| Step 3a（原 mover） | | 43.5 ± 8.6 | 0.422 ± 0.037 | |

**結論：比 Step 3a 差（−16.6 件），5 個 seed 全部較差；沒有任何 case 完成 64 件。**

### Replay 診斷（seed 102 case 0，Step 3a 51 件 vs Step 3c 36 件；AGV-tick 佔比）

| 狀態 | Step 3a | Step 3c |
|---|---|---|
| 來回移動佔總移動 | 51% | 40% |
| 空車、目標加工中、移動（原本的來回） | 33.3% | 10.5% |
| 空車、目標加工中、停住（停車等待） | 19.3% | 23.2% |
| 空車、目標完全沒貨（移動＋停住） | 9.7% | 18.5% |
| 空車、目標有貨卻停住 | 2.8% | 5.1% |
| 載貨移動 | 19.7% | 27.6% |

- mover 本身照設計運作：目標加工中時的來回大幅減少。
- 但重新訓練出的派車變差：派到完全沒貨的 buffer 的時間約翻倍；目標有貨卻停住的時間也翻倍，可能是停在 port 旁的車擋住窄走道（推測，未驗證）。
- 來回移動看起來不是交付量的瓶頸。後續仍以 Step 3a（原 mover）為最佳設定。
- 執行目錄：`runs/central-01/20261013T062857Z-step3c_split_task_shape_park-020b5d891c`（啟動 2026-10-12 23:28）
