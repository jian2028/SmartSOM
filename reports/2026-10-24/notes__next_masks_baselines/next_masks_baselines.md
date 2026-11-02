# 派車 mask／機器特徵／混合格式：實作、對照組與未訓練基準

日期：2026-10-16 ｜ 程式：main @ e4a6ab2 ＋ 本地未 commit 修改 ｜ 依 Jian 指示：先在本地改、和 Justin 規則與 random 比較

## 改動（全部是 algorithm 檔 `controller.extensions.network` 的選項，預設關閉；不動 simulation core）

| 代號 | 選項 | 作用 |
|---|---|---|
| A | `dispatch_supply_mask: true` | 空車只能選「有貨或貨正在來，且還沒被其他空車預約完」的目標；全部不符合時不限制 |
| B | `dispatch_commit: true` | 載貨車有目的地後只能維持原目的地，直到抵達；目的地滿了才解除 |
| C | `target_machine_features: true` | 派車候選加上目的地機器的狀態：是否有機器、閒置、加工中、待取／阻塞、故障、預估剩餘時間、前 buffer 排隊數（候選寬度 +7） |
| D | `role_action_modes: {machine: sequential_binary_uniform, dispatcher: sequential_binary_uniform}` | machine＋派車用 yes/no，buffer 維持 softmax |

- 程式：`src/smartsom/config/extensions.py`、`src/smartsom/learning/production_models.py`、`production_inference.py`、`config/experiment_v3.py`（encoder 收到 network 選項）
- 測試：`tests/unit/test_dispatch_policy_options.py`（9 個）
- Smoke：H1（A+B）、H2（C）、H4（全部）各跑一次訓練＋評估都完成。修掉一個 bug：value-only 請求（owner 不是車）不套 mask。

## 設定檔

- Algorithm：`configs/algorithms/central01_h1_mask_commit.yaml`、`central01_h2_machine_features.yaml`、`central01_h3_all.yaml`、`central01_h4_all_mixed.yaml`（都是 Step 3a 的 algorithm 加上述選項）
- 對照規則：`central01_rules_justin_yield.yaml`（Justin 的生產規則＋我們的 mover `shortest_path_yield`）、`central01_rules_random_yield.yaml`（machine／buffer／dispatcher 全 random＋同一個 mover）
- 實驗：`configs/runs/central01/next/`（`h1`～`h4` 正式訓練、`init_*` 只訓練 256 ticks、`base_*` 規則評估）
- 所有組別：同樣的 5 個評估案例（data_seed 909）、tick_limit 4096、grid；網路組 seeds 101–105；`record: false`（不存錄影）

## 結果：對照組與「幾乎沒訓練」的網路（init = 只訓練 256 ticks，1 次更新）

| 組別 | episodes | 平均交付 | 完成 64 件 | 平均 makespan（完成者） | 平均 tardiness／件 | passing |
|---|---|---|---|---|---|---|
| Justin 規則 | 5 | 64.0 | 5/5 | 2,596 | 46 | 1.000 |
| Random 規則 | 5 | 15.0 | 0/5 | — | 1,796 | 0.992 |
| Step 3a（訓練 65k ticks） | 25 | 43.5 | 0/25 | — | 1,047 | 0.954 |
| init＋A+B | 25 | 63.9 | 24/25 | 3,054 | 271 | 0.974 |
| init＋A+B+C | 25 | 63.6 | 20/25 | 3,021 | 246 | 0.974 |
| init＋A+B+C+D | 25 | 63.8 | 22/25 | 3,076 | 276 | 0.972 |
| init＋C | 25 | 16.2 | 0/25 | — | 1,448 | 0.970 |

## 解讀

- **A+B 兩個 mask 本身就把「幾乎隨機」的網路從約 15 件拉到約 64 件**，比訓練過的 Step 3a（43.5）好很多。表現主要來自 mask，不是學習，正是 Jian 擔心的情況。
- 之前的網路是在一個充滿無效動作的空間裡學，大部分時間花在學「不要做沒用的事」。
- 只加機器特徵（C）、不加 mask 時，未訓練的網路和 random 差不多（16 vs 15），符合預期：特徵要靠訓練才有用。
- **之後比較的重點從「交付件數」改成 makespan 和 tardiness**：
  - 訓練後的網路要贏過「未訓練＋同樣 mask」（約 3,050／270）。
  - 目標是接近 Justin 規則（2,596／46）。

## 正式訓練（65,536 ticks，seeds 101–105，依序 H3 → H4 → H1 → H2）

- 啟動：2026-10-16 01:59，自動接續（`launch_chain.sh`），每組約 2.5 小時
- H3：`runs/central-01/20261016T085954Z-h3_all-ea73b99600`（W&B 名稱 `h3_all-seed10x`）
- H4、H1、H2：依序自動啟動，W&B 名稱 `h4_all_mixed-*`、`h1_mask_commit-*`、`h2_machine_features-*`
- R3-long：依使用者指示**取消，沒有跑**（2026-10-19 09:05 停掉接續啟動）。
- R2-long 完成：2026-10-19 09:10（約 10 小時）。

| 組別 | 平均交付 | 完成 64 件 | 平均 makespan | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 64.0 | 5/5 | 2,596 | 46 | 1.000 |
| 未訓練＋A+B | 63.9 | 24/25 | 3,054 | 271 | 0.974 |
| R2（65k ticks） | 63.9 | 24/25 | 2,964 | 217 | 0.949 |
| **R2-long（262k ticks）** | 63.9 | 24/25 | **2,783** | **128** | 0.942 |

各 seed（交付、makespan、延遲／件）：(64, 2,794, 133)、(63.6, 2,782, 127)、(64, 2,822, 117)、(64, 2,759, 148)、(64, 2,758, 114)

- 訓練拉長 4 倍後明顯進步：比未訓練 makespan −9%、延遲 −53%；比 65k 的 R2 也好（2,964 → 2,783）。
- 5 個 seed 都很集中（2,758～2,822），沒有崩掉。
- 離 Justin 還差約 190 ticks（7%），延遲約 2.8 倍。
- 訓練中的 episode（前 1/3 → 最後 1/3）各 seed 都在進步，曲線還沒變平（例如 seed 105：3,037 → 2,687），再拉長可能還會進步。
- entropy：四個 seed 降到 0.2～1.3，seed 104 維持在 4.2。

## 訓練結果（H1～H4）

| 組別 | 平均交付 | 完成 64 件 | 平均 makespan | 延遲／件 | passing | 對照（未訓練＋同樣 mask） |
|---|---|---|---|---|---|---|
| H1（A+B） | 63.9 | 24/25 | 3,386 | 427 | 0.947 | 3,054／271／0.974 |
| H3（A+B+C） | 58.0 | 21/25 | 3,268 | 468 | 0.938 | 3,021／246／0.974 |
| H4（A+B+C+D） | 46.8 | 14/25 | 3,200 | 533 | 0.981 | 3,076／276／0.972 |
| H2（只有 C，無 mask） | 28.6 ± 13.2 | 0/25 | — | 739 | 0.949 | 未訓練＋C：16.2／1,448／0.970；Step 3a（同設定無 C）：43.5 |

- 有 mask 的三組（H1、H3、H4）訓練後都比未訓練差；H4 兩個 seed、H3 一個 seed 崩掉。
- H2（無 mask）：各 seed 平均交付 29.0、14.6、17.6、46.8、35.2，比未訓練（16.2）好，代表有學到東西；但比 Step 3a（43.5，同設定只差沒有 C）差，seed 間差異也大。
  - 無 mask 時出貨件數會變，所以 return 有訊號；訓練中 entropy 維持在 7～14，沒有崩掉。
  - 只看 5 個 seed，加 C 是否真的比較差還不能下定論。

## 診斷：訓練過程（`reports/training-episodes.json`、`learner-intervals.json`）

用三組共 236 個完成 64 件的訓練 episode：

| 項目 | 數值 |
|---|---|
| episode return 範圍 | 0.474～0.544（標準差 0.011） |
| makespan 範圍 | 2,437～4,073 |
| corr(return, passing) | **0.92** |
| corr(return, makespan) | **−0.20** |
| corr(return, 延遲) | −0.29 |
| return 中 passing 項的標準差 | 0.0106 |
| return 中延遲項的標準差 | 0.0043 |

- **Return 幾乎只反映 passing，不反映速度。** 64 件都出貨時，出貨項固定為 0.25；progress shaping 加總固定為 0.005（= −Φ(s0)）。剩下 passing 項（0.3 權重，且大部分是品檢的隨機結果）的變動是延遲項的 2.5 倍。makespan 從 2,437 到 4,073，return 幾乎沒差。
- 例：H3 seed 103，makespan 3,118 → 2,757（變好），return 卻 0.532 → 0.521（因為 passing 0.971 → 0.921）。
- **Advantage 幾乎是雜訊，卻被標準化放大**：raw |advantage| 約 0.008；`production_collection.normalize_advantages` 每個 batch 把 advantage 標準化成單位變異數，雜訊被放大成正常大小的梯度。
- **Entropy 崩掉**：所有 seed 從約 5～6 降到 0.1～1.5，策略變成幾乎固定、但沒有變好的選擇（critic 的 explained variance 約 0.95，因為 return 幾乎是常數，很好預測）。
- 結論：**reward 沒有給「更快完成」足夠的訊號**，訓練是在追雜訊，所以越訓練越偏離未訓練時的表現。

## 可能的修正（要先和 Jian 討論，會動到 ADR 0039／0040 的 reward 設計）

1. 讓 reward 對時間敏感：每個 tick 對在製品加小成本（flow time）、加大延遲權重，或 gamma < 1。
2. 降低 passing 的雜訊：例如用預期品質（風險）而不是實際的品檢結果，或降低 passing 權重。
3. 防止 entropy 崩掉：提高 entropy coefficient、降低 learning rate。

## 2026-10-18：整合最新 main ＋ reward 權重實驗

### 整合（只在本地，未 commit）
- 主目錄 `/Users/deng/code/SmartSOM` **沒動**（丟棄本地修改再 pull 的操作被安全機制擋下）。完整備份在 `/Users/deng/code/SmartSOM-backup-20261018`。
- 整合版在 worktree `/Users/deng/code/SmartSOM-integrate`：origin/main（107592b，9 個新 commit）＋我們全部的本地改動。
- 衝突 5 個檔案，已解：`authoring_v4.py`（保留 Jian 新增的 experiment 層級 `task_reward` 與我們的 `Learner.reward: RewardSpecV3`）、`experiment.py`（`retention` 與 `keep_continuations` 並存）、`experiment_v3.py`、`composable.py`、`production_inference.py`（改走 Jian 新的 `make_encoder`，在 `policy_factory.encoder` 傳入 network 選項）。
- 新測試檔登記到 `.github/ci/test-ownership.json`；`compositions.py` 錯誤訊息保留上游的 "independent role" 字樣。
- 測試：單元 3,352 通過；整合（learning／playback／grid）37 通過。剩下：
  - `test_registered_observation::test_legacy_digest_and_default_remain_readable` 失敗：Jian 用 implementation digest 釘住 `production_models.py`、`policy_factory.py` 等檔案，我們的改動會改變 digest。**要上傳前需和 Jian 討論**（例如把我們的選項移到獨立模組）。
  - `test_studies::test_ctrl_c_*` 在完整測試中失敗、單獨跑通過（負載造成的不穩定）。
- 驗證：未訓練＋A+B 在整合版重跑，結果與舊版**完全相同**（63.9／3,054／271／0.974）。

### Reward 權重預檢（離線，用 H1/H3/H4 的 282 個訓練 episode 重算 return）
| 權重 shipped/passing/tardiness | corr(return, makespan) | corr(return, passing) | corr(return, 交付) |
|---|---|---|---|
| 0.5/0.3/0.2（原本） | −0.20 | +0.92 | +0.96 |
| 0.5/0.1/1.0 | −0.79 | +0.07 | +0.95 |
| **0.5/0.1/2.0（採用）** | **−0.80** | −0.02 | +0.92 |
| 0.5/0.1/4.0 | −0.81 | −0.06 | +0.90 |

（延遲項用 `accumulated_overdue_time`，重算誤差 0。）

### 實驗（在整合版跑，輸出到主目錄 `runs/central-01/`）
| 組別 | 基底 | 改動 |
|---|---|---|
| R1 `r1_h1_time` | H1（A+B） | 權重 0.5/0.1/2.0 |
| R2 `r2_h1_time_ent` | R1 | ＋ entropy 0.03、learning rate 1e-4 |
| R3 `r3_h3_time_ent` | H3（A+B+C） | 同 R2 |

- R1 啟動：2026-10-18 13:19（`runs/central-01/20261018T201910Z-r1_h1_time-f5b552ff95`），R2、R3 依序自動接續。
- 對照：未訓練＋同樣 mask（A+B：3,054／271；A+B+C：3,021／246）、Justin 規則（2,596／46）。

### 附帶：passing 下降是否來自高風險加工模式？（H1 seed 101 vs 未訓練，各 1 個 case，有錄影）
| | 選 fast（快、錯誤率 4.7%） | normal（2.7%） | slow（0.7%） | passing | makespan |
|---|---|---|---|---|---|
| 未訓練 | 112 | 78 | 81 | 0.938 | 3,029 |
| H1 訓練後 | 56 | 145 | 67 | 0.922 | 3,681 |

訓練後反而少選 fast，所以 passing 下降不是因為選高風險模式；只有一個 case，passing 的差異可能只是品檢隨機。錄影在 `runs/replays/mode_init`、`runs/replays/mode_h1`（各約 2.5 GB，不需要可刪）。

### 結果（2026-10-18 20:05 全部完成；5 seeds × 5 cases）

| 組別 | 平均交付 | 完成 64 件 | 平均 makespan | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 64.0 | 5/5 | 2,596 | 46 | 1.000 |
| 未訓練＋A+B | 63.9 | 24/25 | 3,054 | 271 | 0.974 |
| H1（舊 reward） | 63.9 | 24/25 | 3,386 | 427 | 0.947 |
| R1（新權重） | 63.9 | 24/25 | 3,108 | 251 | 0.941 |
| **R2（新權重＋entropy 0.03、lr 1e-4）** | 63.9 | 24/25 | **2,964** | **217** | 0.949 |
| 未訓練＋A+B+C | 63.6 | 20/25 | 3,021 | 246 | 0.974 |
| H3（舊 reward） | 58.0 | 21/25 | 3,268 | 468 | 0.938 |
| **R3（A+B+C，同 R2 設定）** | 63.8 | 23/25 | **2,922** | 234 | 0.959 |

各 seed（交付、makespan、延遲／件）：
- R1：(63.6, 2,975, 168)、(64, 2,862, 154)、(64, 3,482, 456)、(64, 2,977, 187)、(64, 3,218, 290)
- R2：(64, 3,026, 231)、(64, 3,040, 236)、(64, 2,743, 165)、(63.6, 3,037, 246)、(64, 2,987, 206)
- R3：(63.6, 2,911, 199)、(64, 2,743, 114)、(64, 2,855, 180)、(64, 2,953, 259)、(63.6, 3,201, 418)

解讀：
- **新權重修好了「越訓練越差」**：R1 比 H1 好很多（3,386 → 3,108），但還沒贏過未訓練。
- **加上防崩設定後，第一次贏過「未訓練＋同樣 mask」**：R2 makespan −3%（3,054 → 2,964）、延遲 −20%（271 → 217）；R3 makespan 3,021 → 2,922。R2 五個 seed 都在 2,743～3,040，沒有崩掉。
- 進步幅度不大，離 Justin（2,596／46）還有差距，尤其延遲差很多。
- entropy 仍降到約 1（R1 約 0.4～1.5），比舊 reward（0.1～0.5）好但還是偏低；R3 seed 105 訓練中變差（3,130 → 3,329）。

### 長訓練（2026-10-18 23:02 起）
- R2-long、R3-long：設定和 R2、R3 完全相同，只把 total_ticks 從 65,536 改成 **262,144**（4 倍，每個 seed 約 80 場 episode）。
- 設定檔：`SmartSOM-integrate/configs/runs/central01/next/r2_h1_time_ent_long.yaml`、`r3_h3_time_ent_long.yaml`；依序跑。
- R2-long：`runs/central-01/20261019T060223Z-r2_h1_time_ent_long-149e35ccff`
- R3-long：依使用者指示**取消，沒有跑**（2026-10-19 09:05 停掉接續啟動）。
- R2-long 完成：2026-10-19 09:10（約 10 小時）。

| 組別 | 平均交付 | 完成 64 件 | 平均 makespan | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 64.0 | 5/5 | 2,596 | 46 | 1.000 |
| 未訓練＋A+B | 63.9 | 24/25 | 3,054 | 271 | 0.974 |
| R2（65k ticks） | 63.9 | 24/25 | 2,964 | 217 | 0.949 |
| **R2-long（262k ticks）** | 63.9 | 24/25 | **2,783** | **128** | 0.942 |

各 seed（交付、makespan、延遲／件）：(64, 2,794, 133)、(63.6, 2,782, 127)、(64, 2,822, 117)、(64, 2,759, 148)、(64, 2,758, 114)

- 訓練拉長 4 倍後明顯進步：比未訓練 makespan −9%、延遲 −53%；比 65k 的 R2 也好（2,964 → 2,783）。
- 5 個 seed 都很集中（2,758～2,822），沒有崩掉。
- 離 Justin 還差約 190 ticks（7%），延遲約 2.8 倍。
- 訓練中的 episode（前 1/3 → 最後 1/3）各 seed 都在進步，曲線還沒變平（例如 seed 105：3,037 → 2,687），再拉長可能還會進步。
- entropy：四個 seed 降到 0.2～1.3，seed 104 維持在 4.2。

### R4：passing 權重 0（2026-10-21，回應 Jian「把 passing 去掉」）
- 設定：和 R2 完全相同（masks A+B、entropy 0.03、lr 1e-4、65,536 ticks、5 seeds × 5 cases），只把 task reward 權重改成 shipment/passing/tardiness = **0.5/0/2.0**。
- 離線預檢（用 H1/H3/H4 的訓練 episode 重算）：corr(return, makespan) = −0.81（R2 的 0.5/0.1/2.0 為 −0.80）。
- 設定檔：`SmartSOM-integrate/configs/algorithms/central01_r4_h1_nopass.yaml`、`configs/runs/central01/next/r4_h1_nopass.yaml`
- 執行目錄：`runs/central-01/20261021T231849Z-r4_h1_nopass-a588bf1794`（啟動 2026-10-21 16:18）
- 比較對象：R2（2,964／217）、未訓練＋A+B（3,054／271）、Justin（2,596／46）。
- 完成：2026-10-21 18:27。

| 組別（65k ticks 除非註明） | 平均交付 | 完成 64 件 | makespan（seed 間標準差） | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 64.0 | 5/5 | 2,596 | 46 | 1.000 |
| 未訓練＋A+B | 63.9 | 24/25 | 3,054（53） | 271 | 0.974 |
| R2（0.5/0.1/2.0） | 63.9 | 24/25 | 2,964（127） | 217 | 0.949 |
| **R4（0.5/0/2.0）** | 63.9 | 24/25 | **2,880（73）** | **193** | 0.951 |
| R2-long（0.5/0.1/2.0，262k） | 63.9 | 24/25 | 2,783（27） | 128 | 0.942 |

各 seed（交付、makespan、延遲／件）：(64, 2,897, 214)、(64, 2,982, 261)、(64, 2,795, 195)、(63.6, 2,904, 168)、(64, 2,829, 124)

- 同樣 65k ticks，拿掉 passing 後比 R2 好：makespan 2,964 → 2,880（−3%）、延遲 217 → 193，seed 間差異也變小。
- passing 沒有變差（0.949 → 0.951）：passing 項本來就幾乎沒在引導策略。
- 最好的單場 episode 是 2,673～2,675，接近 Justin（2,596）。
- 還沒做長訓練；R2 拉長 4 倍從 2,964 進步到 2,783，R4 拉長預期也會進步。

### R4-long（2026-10-21 21:24 起）
- 和 R4 完全相同，只把 total_ticks 改成 262,144（4 倍）。
- 設定檔：`SmartSOM-integrate/configs/runs/central01/next/r4_h1_nopass_long.yaml`
- 執行目錄：`runs/central-01/20261022T042438Z-r4_h1_nopass_long-3cdf7febcd`
- 輔助腳本移到不會被清掉的 `/Users/deng/code/SmartSOM-tools/`（W&B bridge、launch chain、disk watchdog）。
- 比較對象：R2-long（2,783／128）、R4（2,880／193）、Justin（2,596／46）。
- 完成：2026-10-22 07:07。

| 組別 | 完成 64 件 | makespan（seed 間標準差） | 各 seed makespan | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 5/5 | 2,596 | — | 46 | 1.000 |
| R4（65k） | 24/25 | 2,880（73） | 2,897, 2,795, 2,904, 2,829, 2,982 | 193 | 0.951 |
| R2-long（passing 0.1，262k） | 24/25 | **2,783（27）** | 2,794, 2,822, 2,759, 2,758, 2,782 | **128** | 0.942 |
| **R4-long（passing 0，262k）** | 23/25 | 2,878（117） | 2,780, 2,823, 2,808, 3,060, 2,946 | 153 | 0.939 |

（seed 依 101～105 排序；R4-long seed 105 有 2 個 case 卡在 62 件沒做完。）

訓練中 episode 的 makespan（每 1/4 平均）與 entropy（開始／中間／結束）：

| Seed | 1/4 | 2/4 | 3/4 | 4/4 | entropy |
|---|---|---|---|---|---|
| 101 | 2,865 | 2,828 | 2,946 | 2,911 | 5.6 / 1.1 / 1.9 |
| 102 | 3,082 | 2,822 | 2,967 | 3,040 | 6.2 / 3.0 / 3.7 |
| 103 | 3,052 | 2,735 | 2,726 | 2,923 | 4.8 / 0.8 / 0.6 |
| 104 | 3,041 | 2,891 | 2,709 | 2,822 | 5.2 / 1.2 / 0.9 |
| 105 | 3,090 | 3,099 | 3,275 | 3,139 | 6.7 / 0.5 / 0.8 |

解讀：
- **R4-long 沒有比 R4 好**（2,878 vs 2,880），也比 R2-long 差（2,783）；延遲有改善（193 → 153），但 seed 104、105 變差，還有 2 個沒做完的 case。
- 訓練曲線不是單調進步：多數 seed 在中段最好（例如 seed 103：2,735／2,726），最後 1/4 又變差；seed 105 一直沒進步。R2-long 則是一路進步。
- 原因還不確定。可能是後期不穩定（learning rate 固定、沒有衰減），而評估用的是最後的 checkpoint，不是中段最好的。
- 所有 1,025 個 checkpoint 都還在，可以評估中段的 checkpoint 來確認。
- 只有 5 個 seed，R2-long 和 R4-long 的差距（約 95 ticks）大致可信（seed 標準差 27 vs 117），但「拿掉 passing 在長訓練下比較差」還不能下定論。

### R4-long 中段 checkpoint 評估（2026-10-22）
同樣 5 個評估案例、sampling；輸出 `runs/ckpt_eval/r4long/`。第 1,024 次更新（最後）5 個 seed 的結果與原評估**完全相同**，確認可重現。

| 更新次數（訓練 ticks） | seed 101 | 102 | 103 | 104 | 105 | 平均 makespan | 延遲／件 | 完成 64 件 |
|---|---|---|---|---|---|---|---|---|
| 256（65k） | 2,897 | 2,982 | 2,795 | 2,904* | 2,829 | 2,880 | 193 | 24/25 |
| 512（131k） | 2,935 | 2,857* | 2,811 | 2,912 | 3,243* | 2,943 | 202 | 23/25 |
| 640（164k） | 3,050* | 2,985 | 2,932* | 2,814* | 3,084** | 2,968 | 197 | 20/25 |
| 768（197k） | 3,016 | 3,029 | 2,952* | 2,810 | 3,118*** | 2,968 | 200 | 21/25 |
| 896（229k） | 2,835 | 3,074** | 2,808* | 2,877* | 3,148 | 2,946 | 175 | 21/25 |
| 1,024（262k，最後） | 2,780 | 2,823 | 2,808 | 3,060 | 2,946** | 2,878 | 153 | 23/25 |

（* 的數量 = 該 seed 沒完成 64 件的 case 數；makespan 只算完成的。）

- 第 256 次更新的結果和 R4（65k）完全一樣，符合預期（同 seed、同設定）。
- **中段沒有比較好**：512～896 次更新的 makespan（2,943～2,968）反而比開始和最後差，沒做完的 case 也比較多（最多 5/25）。訓練中 episode 看起來中段較好，是因為那些是探索中的 episode，不能代表評估結果。
- makespan 從 256 到 1,024 次更新基本沒有進步（2,880 → 2,878）；只有延遲慢慢改善（193 → 153）。
- 結論：**拿掉 passing 之後，65k 之後的訓練對 makespan 沒有幫助**；提早停止也不會更好。和 R2-long 一路進步（2,964 → 2,783）不同，原因還不清楚。

### 同一個 tick 內的互斥 mask（2026-10-22，回應 Jian「中央控制器要一次給出不衝突的組合」）
- 新選項 `dispatch_tick_exclusive`（algorithm 檔 `controller.extensions.network`，預設關閉；只在整合版 `SmartSOM-integrate`，不動 core）。
- 做法：同一個派車回合（同 tick、同 stage、同一個狀態快照）內，控制器記住自己前面幾台車的選擇：
  - 空車：某個貨源的貨（現有＋正在來）扣掉「之前 tick 已預約、這回合還沒重選的車」和「這回合已選的車」後，沒剩就不能選。
  - 載貨車：有限容量的目的地，空位扣掉「正前往的車」和「這回合已選的車」後，沒剩就不能選。
  - 全部都不符合時不限制（模擬器要求每台車都要有目標）。
- 程式：`learning/production_models.py`（`dispatch_round`、`note_choice`、`exclusive_mask`）、`production_inference.py`（`ModelPolicy.choose` 選完後呼叫 `note_choice`）、`config/extensions.py`。
- 測試：`tests/unit/test_dispatch_policy_options.py` 新增 5 個（同回合擋住已拿走的貨、換回合重置、目的地空位、未開啟時無作用、ModelPolicy 實際選擇後下一台看得到），共 14 個通過；相關單元與整合測試 263 個通過。
- Smoke：R5（= R2 設定＋`dispatch_tick_exclusive: true`）訓練 1,024 ticks＋評估 1 個 case，正常完成。
- 設定檔：`SmartSOM-integrate/configs/algorithms/central01_r5_h1_exclusive.yaml`、`configs/runs/central01/next/r5_h1_exclusive.yaml`（尚未正式跑）。

#### 驗證（2026-10-22）
衝突計算程式：`/Users/deng/code/SmartSOM-tools/tick_conflicts.py`（讀錄影，逐 tick 檢查每個派車選擇）。
同一個案例（seed 101、case 0），都是幾乎沒訓練的網路：

| | 舊 mask（A+B） | ＋同一 tick 互斥 |
|---|---|---|
| 空車：選到已沒貨的貨源 | 585 | 1,081 |
| 　其中「其實還有別的貨源可選」（可避免） | **6** | **0** |
| 　其中「所有貨源都沒貨，模擬器強制要選一個」 | 579 | 1,081 |
| 載貨車：目的地空位被重複預約 | **149**（同一 tick 78） | **0** |

- 可避免的衝突全部消失：空車 6 → 0，載貨車 149 → 0。
- 剩下的空車「衝突」都是強制的：車比可拿的貨多時，模擬器規定空車一定要有目標（沒有「原地待命」選項），只能指派到沒貨的地方。這不是同一個 tick 內的搶貨。

未訓練基準（5 seeds × 5 cases）：

| | makespan | 各 seed | 延遲／件 | passing |
|---|---|---|---|---|
| 未訓練＋A+B | 3,054 | 3,044, 3,031, 3,118, 3,100, 2,986 | 271 | 0.974 |
| 未訓練＋A+B＋互斥 | **3,018** | 2,905, 2,968, 3,169, 3,033, 3,004 | **237** | 0.968 |

- 加互斥後，未訓練的表現小幅變好（makespan −1%、延遲 −13%）。之後 R5 訓練要比的基準是 3,018／237。

### R5：R2 設定＋同一 tick 互斥（65k，2026-10-22 14:09 起）
- 設定：masks A+B＋`dispatch_tick_exclusive`、權重 0.5/0.1/2.0、entropy 0.03、lr 1e-4、65,536 ticks、5 seeds × 5 cases。
- 執行目錄：`runs/central-01/20261022T210950Z-r5_h1_exclusive-a80221803f`
- 比較對象：未訓練＋互斥（3,018／237）、R2（2,964／217）、Justin（2,596／46）。
- 完成：2026-10-22 16:16。

| 組別（65k ticks 除非註明） | 完成 64 件 | makespan（seed 標準差） | 各 seed | 延遲／件 | passing | 最好單場 |
|---|---|---|---|---|---|---|
| Justin 規則 | 5/5 | 2,596 | — | 46 | 1.000 | 2,521 |
| 未訓練＋互斥 | 24/25 | 3,018（98） | 2,905, 2,968, 3,169, 3,033, 3,004 | 237 | 0.968 | 2,699 |
| R2（無互斥） | 24/25 | 2,964（127） | 3,026, 3,040, 2,743, 3,037, 2,987 | 217 | 0.949 | 2,519 |
| R4（無互斥，passing 0） | 24/25 | 2,880（73） | 2,897, 2,982, 2,795, 2,904, 2,829 | 193 | 0.951 | 2,673 |
| **R5（R2＋互斥）** | 23/25 | **2,855**（125） | 2,808, 2,711, 2,818, 2,890, 3,047 | 202 | 0.955 | 2,560 |
| R2-long（無互斥，262k） | 24/25 | 2,783（27） | 2,794, 2,782, 2,822, 2,759, 2,758 | 128 | 0.942 | 2,577 |

- 同樣 65k，R5 是目前最好的：比 R2 makespan −3.7%（2,964 → 2,855）、延遲 217 → 202；比未訓練＋互斥 −5%。
- 4 個 seed 在 2,711～2,890，seed 105 較差（3,047）；有 2 個 case 沒做完 64 件。
- 還沒有長訓練；R2 拉長 4 倍從 2,964 進步到 2,783。

## 2026-10-22：再次整合最新 main（#25、#26）
- 新 worktree：`/Users/deng/code/SmartSOM-integrate2`（origin/main 04195bb ＋ 我們全部本地改動）；舊的 `SmartSOM-integrate` 保留不動；備份 `/Users/deng/code/SmartSOM-backup-20261022`。未 commit、未 push。
- 衝突只有 CI 測試清單（已解，並去掉一個重複項）；我們的 `tests/unit/test_production_shaping.py` 和 Jian 新增的同名檔衝突，改名為 `test_production_potential_shaping.py`。
- 測試：單元 3,637 通過，只剩已知的 digest 測試失敗；整合（learning／playback／grid）37 通過。
- Jian 新增 ADR 0050（production progress＋blocking potential shaping），和我們本地的 progress shaping 功能重疊，之後要決定是否改用他的。他的 ADR 0040 編號也和我草稿的 0040 撞號。

### 重要：評估結果和舊版不同
- 用新程式重評估**同一個**舊模型（未訓練＋A+B，seed 101，同樣 5 個案例、同樣的世界 `world_sha256`）：5 場結果都不一樣（例如 3,029 → 未完成、3,046 → 2,920）。
- 原因：上游 #25 改了評估時的 policy 抽樣種子（`composable.py`：`policy_seed = case.get("policy_root", case["seed"])`，以前沒有 policy_root 時用預設種子）。**模擬器沒有改**，世界完全相同，只是 sampling 抽到的動作不同。
- 未訓練＋A+B 的 25 場平均：舊 3,054／271 → 新 2,971／256。**光是換抽樣種子，平均 makespan 就差約 80 ticks**。
- 影響：之前各組之間小於約 100 ticks 的差異（例如 R5 2,855 vs R4 2,880）可能只是評估雜訊；需要用新程式重新評估，並增加評估案例數。

## 2026-10-23：用新程式重新評估（每個 seed 20 個案例，共 100 場／模型）
- 程式：`SmartSOM-integrate2`（最新 main＋本地改動）；評估設定 `configs/runs/central01/reeval/*.yaml`，輸出 `runs/reeval/`。
- 不確定性有兩種：「場次 SE」只反映評估雜訊；「seed 間 SE」（= seed 平均的標準差 ÷ √5）還包含訓練的隨機性，比較兩個訓練方法時應看這個。

| 組別 | 完成 64 件 | makespan（場次 SE） | seed 平均（seed 間 SE） | 延遲／件 | passing |
|---|---|---|---|---|---|
| Justin 規則 | 19/20 | **2,591**（±24） | — | **65** | 1.000 |
| **R2-long**（passing 0.1，262k） | 94/100 | **2,815**（±22） | 2,756, 2,937, 2,910, 2,812, 2,658（±51） | **140** | 0.921 |
| R5（R2＋同一 tick 互斥，65k） | 93/100 | 2,887（±26） | 2,779, 2,733, 2,956, 2,977, 3,004（±55） | 218 | 0.943 |
| R4（passing 0，65k） | 92/100 | 2,890（±25） | 2,960, 2,893, 2,757, 2,801, 3,047（±52） | 210 | 0.945 |
| R4-long（passing 0，262k） | 93/100 | 2,922（±22） | 2,862, 3,008, 2,906, 2,965, 2,856（±29） | 172 | 0.929 |
| R2（passing 0.1，65k） | 95/100 | 2,998（±26） | 3,186, 3,011, 2,756, 3,083, 2,970（±72） | 240 | 0.941 |
| 未訓練＋A+B | 94/100 | 3,044（±22） | （±13） | 300 | 0.970 |
| 未訓練＋A+B＋互斥 | 93/100 | 3,053（±21） | （±16） | 281 | 0.972 |
| Random 規則 | 0/20 | 平均只交付 16.2 件 | — | 1,460 | 0.968 |

結論（用 seed 間 SE 判斷）：
- **R2-long 仍是最好的**，比 Justin 慢約 220 ticks（約 9%），差距明確（約 4 個 SE）；延遲約 2 倍。
- **長訓練對 R2 有用**：R2 → R2-long 進步 183 ticks（約 2 個 SE）。對 R4 沒用（2,890 → 2,922）。
- **R5、R4 都比 R2 好約 110 ticks，但只有約 1.2 個 SE**，還不能確定；三者（R4、R4-long、R5）差不多。
- **同一 tick 互斥 mask 單獨沒有幫助**：未訓練時 3,044 vs 3,053（衝突確實消失了，但交付速度沒變）。
- 之前 5 個案例時看到的「R5 比 R4 好」「中段 checkpoint 比較差」等小差異都在雜訊範圍內。
- 訓練的隨機性（seed 間差異約 ±100～160 ticks）比評估雜訊大；要分辨 100 ticks 以內的差異，需要更多 seed。

## R6：改用 Jian 的 ADR 0050 shaping（2026-10-23 12:15 起）
- 設定：R2（masks A+B、權重 0.5/0.1/2.0、entropy 0.03、lr 1e-4），把我們的 progress shaping（`team: builtin.production_potential`）換成 Jian 的 `production_potential_shaping`（progress 0.1、blocking 0.1、handoff 16、blocking age 32、buffer pressure 0.75、cargo grace 64；ADR 0050 的初始值）。
- 65,536 ticks、5 seeds、評估 20 cases。程式：`SmartSOM-integrate2`。
- 修了 Jian 程式的一個小 bug（只在本地）：`experiments/production_shaping.py` 的 `_nominal` 用 `float()` 讀 `reference_ticks`，遇到分數字串（例如 "22236987/500000"）會出錯；改成 `float(Fraction(...))`，並在 `tests/unit/test_production_shaping.py` 加了測試。**需要告訴 Jian。**
- Smoke（8,192 ticks）：shaping 帳本正常，progress／blocking 每場加總約 0（closure error ~1e-16），training return = objective return。
- 執行目錄：`runs/central-01/20261023T191539Z-r6_jian_shaping-9dea57800b`
- 比較對象（新評估）：R2（2,998／240）、R2-long（2,815／140）、Justin（2,591／65）。
- 完成：2026-10-23 15:20。

| 組別（65k 除非註明） | makespan | seed 平均（seed 間 SE） | 延遲／件 | passing |
|---|---|---|---|---|
| Justin 規則 | 2,591 | — | 65 | 1.000 |
| R2-long（我們的 shaping，262k） | 2,815 | （±51） | 140 | 0.921 |
| R5（我們的 shaping＋互斥） | 2,887 | （±56） | 218 | 0.943 |
| R4（我們的 shaping，passing 0） | 2,890 | （±52） | 210 | 0.945 |
| R2（我們的 shaping） | 2,998 | 3,186, 3,011, 2,756, 3,083, 2,970（±71） | 240 | 0.941 |
| **R6（Jian 的 shaping）** | **3,061** | 3,018, 3,011, 3,215, 2,814, 3,238（±78） | **269** | 0.956 |
| 未訓練＋A+B | 3,044 | （±13） | 300 | 0.970 |

- **R6 沒有比 R2 好**：makespan 3,061 vs 2,998、延遲 269 vs 240，差異在 seed 間雜訊範圍內（約 0.6 個 SE），所以只能說「沒有看到幫助」。
- R6 和未訓練（3,044）幾乎一樣：用 ADR 0050 的初始權重、訓練 65k ticks 時，策略沒有明顯學到東西。
- 注意：R2 是用舊版程式訓練的，R6 是新版；兩者還混著程式版本差異（R2c 對照組沒有跑）。
- blocking／progress 的權重是 Jian 標明「未校準」的初始值，可能需要調整（相對於 tardiness 權重 2.0，0.1 可能太小）。

## R2c：R2 在新版程式重新訓練（對照組，2026-10-23 16:22 起）
- 設定和 R2 完全相同（含我們的 progress shaping），只是在 `SmartSOM-integrate2`（新版）訓練；評估 20 cases。除了 shaping 之外和 R6 完全相同。
- 目的：分辨 R6 沒進步是因為 Jian 的 shaping，還是因為新版程式。
- 執行目錄：`runs/central-01/20261023T232200Z-r2c_control_newcode-e5bc32727f`
- 完成：2026-10-23 19:22。

| 組別 | makespan | 各 seed | 延遲／件 |
|---|---|---|---|
| R2（舊版訓練，新版評估） | 2,998 | 3,186, 3,011, 2,756, 3,083, 2,970 | 240 |
| **R2c（新版訓練）** | **2,998** | 3,186, 3,011, 2,756, 3,083, 2,970 | 240 |
| R6（新版訓練，Jian 的 shaping） | 3,061 | 3,018, 3,011, 3,215, 2,814, 3,238 | 269 |

- **R2c 和 R2 的 100 場評估逐場完全相同**：新版程式不影響訓練結果（同樣 seed 訓練出一模一樣的模型）。之前的訓練結果都可以直接和新版比較。
- 所以 R6 沒有進步，是 **Jian 的 shaping（初始權重）本身**的效果，不是程式版本。
- R6 − R2c 逐 seed：−168、0、+459、−269、+268（平均 +58，SE ±135）：有好有壞，沒有一致的方向，平均來說沒有幫助。（seed 102 兩者平均都是 3,011，但 20 場逐場都不同，只是巧合。）

## 2026-10-24 夜間：R7 → R5-long（依序）
- **R7**：R6 的 Jian shaping，權重 progress／blocking 從 0.1 提高到 **1.0**；65k、5 seeds、評估 20 cases。執行目錄 `runs/central-01/20261024T074549Z-r7_jian_shaping_w1-8c6094d17e`（00:45 啟動，已確認讀到權重 1.0）。
- **R5-long**：R5（R2＋同 tick 互斥）訓練 262,144 ticks；評估 20 cases。R7 結束後自動啟動（04:01，`runs/central-01/20261024T110101Z-r5_h1_exclusive_long-21032d4813`）。

### R7 結果（03:48 完成）
| 組別（65k） | makespan | 各 seed | 延遲／件 | passing |
|---|---|---|---|---|
| R2c（我們的 shaping） | 2,998 | 3,186, 3,011, 2,756, 3,083, 2,970 | 240 | 0.941 |
| R6（Jian shaping 0.1／0.1） | 3,061 | 3,018, 3,011, 3,215, 2,814, 3,238 | 269 | 0.956 |
| **R7（Jian shaping 1.0／1.0）** | **2,968** | 3,073, 2,828, 2,978, 3,035, 2,931（seed SE ±43） | 270 | 0.966 |

- 權重調高後比 R6 好（3,061 → 2,968），和我們的 shaping（R2c 2,998）差不多：R7 − R2c 逐 seed −113、−183、+222、−48、−39，平均 −32（SE ±69），不顯著。
- R7 的 seed 間差異比較小（SE ±43 vs ±71），延遲沒有改善（270 vs 240）。
- 結論：Jian 的 shaping 權重要調高才有作用；調到 1.0 後效果和我們的 shaping 相當，但沒有更好。

### R5-long 結果（2026-10-24 14:53 完成）
| 組別（262k） | 完成 64 件 | makespan | 各 seed | 延遲／件 | passing |
|---|---|---|---|---|---|
| R2-long（無互斥） | 94/100 | **2,815** | 2,756, 2,937, 2,910, 2,812, 2,658（±51） | **140** | 0.921 |
| R5-long（＋同 tick 互斥） | 89/100 | 2,830 | 3,007, 2,795, 2,873, 2,826, 2,671（±55） | 184 | 0.927 |

- R5-long 和 R2-long 差不多（逐 seed 差 +251、−142、−37、+14、+13，平均 +20），但延遲較高、沒做完的場次較多（11 vs 6）。
- 同 tick 互斥 mask 在長訓練下也沒有幫助。R2-long 仍是最好的。

### R2-long 確定性評估（每次選機率最高的動作；2026-10-24 18:14 完成）
| Seed | 完成 64 件 | 平均交付 | makespan（完成者） |
|---|---|---|---|
| 101 | 4/20 | 34.9 | 3,616 |
| 102 | 12/20 | 63.1 | 3,918 |
| 103 | 20/20 | 64.0 | 3,364 |
| 104 | 9/20 | 45.1 | 2,735 |
| 105 | 0/20 | 13.3 | — |
| 全部 | **45/100** | 44.1 | 3,408 |

- **確定性評估差很多**：一半以上的場次做不完（平均只交付 44 件，sampling 時 63.9 件）。
- 推測：每次都選同一個最高分的動作，容易卡在重複的循環裡（例如一直派同一台車去同一個地方）；sampling 的隨機性反而幫助脫困。
- 結論：**評估繼續用 sampling**；確定性的策略目前不可用。這也表示網路學到的策略本身還不夠穩定。
