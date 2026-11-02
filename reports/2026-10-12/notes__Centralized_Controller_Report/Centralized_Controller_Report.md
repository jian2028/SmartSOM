# SmartSOM — Centralized Controller 工作總整理

作者：Morris（Shih-Chun Deng）｜ 期間：2026-10-06 ～ 2026-10-11
程式基準：Jian 的 `main`（最新已整合 `e4a6ab2`），加上 Justin 分支 `codex/two-sided-agv-controller-review` 的本地 squash 合併。
**所有程式修改目前都在本地、尚未 commit／push。**

---

## 1. 一頁摘要

| 階段 | 設定 | 結果 |
|---|---|---|
| Matrix 運輸、central PPO | AGV 瞬移、central 控制全部角色 | greedy 評估只用 4～5 台機器；修 bug 後 greedy 崩潰（0～0.4 件）；**發現隨機策略就能 64/64**，場景太簡單 |
| Grid 運輸、central PPO（控制全部角色含 mover） | 3 種 reward（legacy／task／shaping）、learner_scale、DQN | 0～3 件，學成「到處亂走」；shaping 有劑量反應但交付仍近 0 |
| **Grid、central PPO 拆 mover（mover 交給規則）** | task reward、seed 101～105 | **29.5 ± 17.3 件**（首次學會送貨） |
| **＋ progress shaping（ADR 0040）** | task＋PBRS，progress 0.001 | **43.5 ± 8.6 件**，seed 間更穩定 |
| 參考：規則 | 我的規則／Justin 的規則 | 64/64（makespan 2,641／2,475，Justin 規則 1 個案例卡在 63） |

**三個關鍵轉折：**
1. **改用 grid**：matrix 小圖連隨機策略都能完成，學習的價值看不出來。
2. **拆 mover**：central 同時控制約 16 個決策（8 台車的每一步移動加上派車），只有一個共同 reward，credit assignment 太難。把移動交給規則後，central PPO 才學會送貨。
3. **Progress shaping**：task reward 太稀疏，加上 PBRS 形式的工單進度 shaping 後，平均交付提升約 14 件。

---

## 2. 研究設定

- **分工**：Morris 做 Centralized controller，Jian 做 multi-agent self-organizing policy，最後在相同場景下比較。
- **場景**：`batch02_small_h1`（19×9 格、8 台機器、8 台 AGV）。工作量主要用 `batch02_small_low`（64 件），也試過 128 件。
- **時間窗**：4,096 ticks。
- **入口格式**：v4 experiment（Jian 的四檔案設定），執行時編譯成 v3 core。
- **監看**：W&B 專案 `creeps-usc/SmartSOM2026`。v4 會忽略 `logging.wandb`，所以用外掛 bridge 讀 `progress.json` 上傳。

---

## 3. 時間軸與實驗

### 3.1 Matrix 運輸（10-06～10-08）
| 實驗 | 設定 | 結果 |
|---|---|---|
| Stage A／B | central PPO、seeds 101／1101、16,384 ticks | 交付 62.6（2/5 完成）／54.4（0/5）；**只用 4～5 台機器**，偏好慢的雙胞胎機器 |
| Long | 65,536 ticks | 訓練 4 倍長，沒有改善 |
| 拉 Jian 修 bug 的 main（`359501c`）重跑 Stage B | 相同 | **greedy 評估崩潰**：0～0.4 件。seed 1101 死鎖：所有 AGV 都送到 machine_008，收成品的合法動作被提供 11,382 次從未選中 |
| Sampling 診斷 | 改用 sampling 評估 | 訓練後的策略 64/64、8 台機器全用；**未訓練的隨機策略也 64/64**（makespan ~3,075 vs rule 2,851） |

**結論**：「只用部分機器」是 greedy 評估的假象；matrix 小圖太簡單，無法看出學習效果 → 改用 grid（Jian 確認最終目標就是 grid）。

### 3.2 Grid 交通規則（10-08）
- Jian 原本的規則（`nearest`＋`shortest_path`）在 grid 上 **8／4／2 台 AGV 全部卡死**（gridlock 或兩台互擋來回）。
- 新增規則變體（**原規則一行沒改**）：
  - `nearest_sticky`：未到達前不換目標；空車優先選別人沒選的目標。
  - `shortest_path_yield`：走靜態最短路，被擋就等或以 seeded 機率 0.25 讓路；空車在沒貨可拿時停車。
- 結果：**25/25（5 seeds × 5 cases）64/64 完成**，正式 baseline 平均 makespan **2,641**。

### 3.3 Grid、central 控制全部角色（10-08～10-09）
| 實驗 | 設定 | 結果 |
|---|---|---|
| 3 種 reward × 2 seeds（128 件） | legacy／task／distance shaping | 交付 0～1（rule 106） |
| 診斷 | 訓練數值 | legacy／shape：value loss ~7,500、KL ~0.0002，策略幾乎不更新；task：有更新但訊號太稀疏 |
| learner_scale 0.001 | 縮放 reward | value loss → 0.003、KL 正常、entropy 下降，**但學成亂走**（移動 1.2～1.7 萬格、碰撞上萬、幾乎不取貨） |
| Shaping 權重掃描 | 10／100／1000 | **劑量反應明確**：取貨 0～6 → 8～36、碰撞 9～13k → 2.3～3k；交付仍 0～3 |
| Central DQN（照 Jian 建議開放）vs PPO（64 件） | 新 production shaping | DQN：0 取貨、碰撞最多；PPO 新 shaping 碰撞減半；**都沒學會送貨** |

**結論**：瓶頸是 central 模式的 credit assignment（約 16 個決策共用一個 reward）。

### 3.4 拆 mover ＋ reward shaping（10-10～10-11，依 Jian 的指示）
所有實驗：grid、64 件、data_seed 909、seeds 101～105、65,536 ticks、validation 關閉、評估 last checkpoint 5 cases（sampling）、mover 規則 `shortest_path_yield`。

| 實驗 | Reward | 平均交付 | Task return | 紀錄 |
|---|---|---|---|---|
| **Step 1** | task（50/30/20） | 29.5 ± 17.3 | 0.356 ± 0.083 | `notes/runs/step1_split_task.md` |
| **Step 3a** | task＋progress shaping 0.001 | **43.5 ± 8.6** | **0.422 ± 0.037** | `notes/runs/step3a_split_task_shape.md` |
| **Step 3b** | task＋progress shaping 0.004 | 41.8 ± 11.7 | 0.411 ± 0.052 | `notes/runs/step3b_split_task_shape.md` |

各 seed 平均交付：

| Seed | Step 1 | Step 3a | Step 3b |
|---|---|---|---|
| 101 | 8.0 | 40.0 | 46.0 |
| 102 | 41.4 | 51.0 | 38.8 |
| 103 | 13.4 | 46.2 | 48.6 |
| 104 | 44.4 | 30.2 | 23.0 |
| 105 | 40.2 | 50.0 | 52.8 |

**結論**：拆 mover 後 central PPO 學會送貨；progress shaping 讓平均提升約 14 件，原本卡住的 seed 101／103 也學會；但 seed 104 在兩種 shaping 強度下都變差。最好的設定約達 rule 的 2/3。

### 3.5 規則 baseline（同一組評估案例）
| 規則 | 結果 | 紀錄 |
|---|---|---|
| 我的 `nearest_sticky`＋`shortest_path_yield` | 5/5 完成，平均 makespan 2,641 | — |
| Justin 的 two-sided controller | 4/5 完成，平均 makespan 2,475；**case 1 卡在 63/64** | `notes/runs/rule_grid_justin.md` |

---

## 4. 程式修改（全部本地、未 commit）

### 4.1 新增檔案
| 檔案 | 內容 |
|---|---|
| `src/smartsom/learning/reward_shaping.py` | `builtin.agv_distance_potential`：AGV 到目標距離的 PBRS shaping |
| `src/smartsom/learning/production_shaping.py` | `builtin.production_potential`：載貨／空車分開的距離權重＋工單進度位能 |
| `docs/decisions/0040-v3-potential-shaping-on-task-reward.md` | ADR 0040 草稿（Proposed）：task reward 可疊加 PBRS shaping |
| 測試：`test_reward_shaping.py`、`test_production_shaping.py`、`test_grid_traffic_rules.py`、`test_grid_traffic_baseline.py` | 新功能的單元與整合測試 |

### 4.2 修改 Jian 的程式（皆非 simulation core）
| 修改 | 檔案 | 原因 |
|---|---|---|
| v4 algorithm 可設定 task reward；check 階段檢查 gamma 與 validation | `config/authoring_v4.py`、`config/experiment_v4.py` | v4 原本無法使用 ADR 0039 的 task reward |
| **修 bug**：評估忽略 `deterministic` 設定（永遠 greedy） | `experiments/composable.py` | 之前的「只用部分機器」等結論都受影響 |
| 開放 central DQN（provider、截斷 bootstrap 的 key） | `authoring_v4.py`、`experiment_v3.py`、`production_inference.py`、`composable.py` | Jian 建議試 DQN |
| 選用 `checkpointing.keep_continuations`（預設不變） | `config/experiment.py`、`composable.py` | 每個 update 都存一次、`keep_last` 不刪檔，DQN 一次 1.5 GB 把硬碟塞滿 |
| **Central controller 可把 mover 交給規則** | `authoring_v4.py`、`compositions.py`、`experiment_v3.py`、`experiment_v4.py`、`composable.py` | 拆 mover（Jian 同意） |
| ADR 0040 實作：task＋PBRS shaping、shaping 紀錄 | `experiment_v3.py`、`experiment_v4.py`、`composable.py` | Jian 同意先在本地小規模試 |
| Replay 支援 matrix 運輸的瞬移 | `studio/playback.py` | 原本會報 Non-adjacent movement |
| 新規則變體 | `algorithms/production_rules.py`、`rule_registry.py` | grid 上原規則會卡死 |

### 4.3 Justin 分支（本地 squash 合併）
- 加入他的 two-sided AGV controller（`agv_dispatcher.py`、`agv_planner.py`、`clearance_shortest_path` 等）。
- 衝突在 `production_rules.py`、`rule_registry.py`：`nearest` 使用 Justin 新的 `dispatch_target()`，`nearest_sticky` 保留舊邏輯以便重現先前實驗。

### 4.4 測試狀態
- 相關測試全部通過（規則 168、設定 70＋、學習 90＋、shaping 15 等）。
- 全量測試：與未修改的 main 相比**沒有新的失敗**。有 13 個既有失敗（Ray tuning／背景程序／Ctrl-C 類），在未修改的 main 上也一樣失敗；其中部分單獨跑會通過，屬於負載敏感。

---

## 5. 遇到的問題與處理

| 問題 | 原因 | 處理 |
|---|---|---|
| 評估永遠 greedy | `ModelPolicy.deterministic` 寫死為 True | 修正並加回歸測試 |
| W&B 沒有資料 | v4 忽略 `logging.wandb` | 外掛 bridge（每個 seed 一個程序） |
| Replay 報 Non-adjacent movement | matrix 運輸是瞬移 | 只在記錄的 travel 剛好抵達時允許跳格 |
| Grid 上規則卡死 | 規則沒有讓路／打破對稱 | 新規則變體 |
| 訓練幾乎不更新 | legacy reward 尺度太大 | `learner_scale: 0.001` |
| 學成亂走 | central 約 16 個決策共用一個 reward | 拆 mover |
| Task reward 學不動 | 訊號太稀疏 | progress shaping（ADR 0040） |
| **硬碟塞滿，實驗中斷** | 每個 update 都存 checkpoint，`keep_last` 不刪檔；DQN 續跑檔 1.5 GB；grid 評估錄影每案例 3～4 GB | `keep_continuations`、清理舊檔、硬碟監看程式（低於 50 GB 自動停止） |
| 訓練中斷（電腦睡眠） | 闔上螢幕 | caffeinate＋提醒保持開蓋 |
| 搬移 repo 後測試出現 24 個 error | 舊的 `__pycache__` 指向舊路徑 | 清除 pycache |
| Shaping 紀錄少記最後一步 | episode 在 reward 計算前就寫入 | 修正並加回歸測試（以真實 episode 驗證） |
| 「port 死鎖無法用規則解決」 | **我的推論錯誤**：實際案例都還有其他可走的格子 | 已向 Jian 更正；45 個 episode 沒有再出現死鎖 |

**曾經說錯、已更正的地方：**
1. 以為訓練變慢 50～100 倍，其實是電腦睡眠。
2. 低估 DQN 的速度約 2 倍。
3. Observation 維度一開始算成 3,681，實際是 973。
4. Port 死鎖的說法（見上表最後一列）。

---

## 6. 驗證方法

- **Task reward**：逐 tick 加總等於最終公式（誤差約 1e-14）；出貨數、延遲時間和 trace 獨立重算完全一致。matrix 與 grid 都驗證過。
- **Shaping（PBRS）**：一整個 episode 的 shaping 加總等於 −Φ(初始狀態)（誤差 7e-15），報告值與之一致。
- **拆 mover**：錄影中 mover 的 32,768 個決策全部來自規則，central 只做機器／buffer／派車決策。
- **每次長跑前**：先 `check`，再跑 smoke，確認路由、reward、評估、硬碟用量都正確。

---

## 7. 尚待處理

| 項目 | 狀態 |
|---|---|
| JobShopLab 式控制器（網路管機器＋派車、逐條 yes／no，buffer／mover 用 Justin 規則） | 下一步，先出設計 |
| Justin 規則 case 1 卡在 63/64 | 待用 replay 檢查 |
| ADR 0040 正式通過 | 待 Jian 確認 |
| 評估 seed 404001～404005 | v3／v4 都無法直接指定；目前用 data_seed 909 |
| Observation 未歸一化（`tick/100` 到 41、兩個候選特徵到 ±16） | 待 Jian 決定是否升級格式 |
| Task reward 的 tardiness 權重實際影響很小 | 待 Jian 決定 |
| 所有本地修改的 commit／push | 待和 Jian 確認 |

---

## 8. 更新（2026-10-11～10-12）：JobShopLab 式控制器

依 Jian 建議：網路只管**調度**（machine 選 job、dispatcher 派車），buffer／mover／pickup 用 **Justin 的規則**（edd、clearance_shortest_path、priority_greedy）；比較動作格式 softmax vs **逐條 yes／no**。
新增程式（本地、未 commit）：central 可委派 buffer＋mover 給規則；`action_mode`（`sequential_binary`、`sequential_binary_uniform`）；測試含手算機率、真實訓練／存檔／續跑一致。

| 實驗 | 網路控制 | 動作格式 | Shaping | 平均交付（64） |
|---|---|---|---|---|
| Step 1 | machine＋buffer＋dispatcher | softmax | 無 | 29.5 ± 17.3 |
| Step 3a | machine＋buffer＋dispatcher | softmax | progress 0.001 | **43.5 ± 8.6** |
| F1 | machine＋dispatcher | softmax | 無 | 11.4 ± 8.7 |
| E1 | machine＋dispatcher | softmax | progress 0.001 | 25.7 ± 11.0 |
| E2 | machine＋dispatcher | yes／no（初始偏差） | progress 0.001 | 3.9 ± 2.8 |
| F2 | machine＋dispatcher | **yes／no（初始均勻）** | 無 | 26.5 ± 7.1 |
| F3 | machine＋dispatcher | **yes／no（初始均勻）** | progress 0.001 | 32.3 ± 9.6 |

**發現：**
1. **初始偏差是 E2 失敗的原因**：原本的 yes／no 在 22 個選項的派車決策中，第一個選項初始被選機率 53.8%、entropy 1.28（均勻為 3.09）。加入固定 hazard 先驗（`sequential_binary_uniform`）後初始均勻（entropy 3.08），F2／F3 大幅改善。
2. **在 JobShopLab 範圍內，yes／no 明顯優於 softmax**：無 shaping 26.5 vs 11.4；有 shaping 32.3 vs 25.7。
3. **Shaping 在所有設定都有幫助**：+14.0（Step 1→3a）、+14.3（F1→E1）、+5.8（F2→F3）。
4. **整體最好仍是 Step 3a**（網路也管 buffer、用我的 mover）：43.5。JobShopLab 範圍最好的 F3 為 32.3。範圍與規則的差異同時包含 4 個改動（buffer 交給 edd、Justin mover、priority_greedy、網路不管 buffer），尚無法單獨歸因。
