# 2026-10-18 工作摘要（Centralized controller）

**一句話：改 reward 權重＋防止 entropy 崩掉後，訓練第一次贏過「未訓練＋同樣 mask」（makespan −3%、延遲 −20%），但離 Justin 規則還有差距。**

## 1. 整合最新 main（只在本地，未 commit）
- 主目錄沒動；整合版在 `/Users/deng/code/SmartSOM-integrate`（origin/main 9 個新 commit ＋ 我們的改動），備份在 `/Users/deng/code/SmartSOM-backup-20261018`。
- 5 個檔案衝突已解；單元測試 3,352 通過、整合測試 37 通過。
- 一個需要和 Jian 討論的失敗：Jian 用 digest 釘住 `production_models.py`／`policy_factory.py`，我們的改動會改變 digest。
- 未訓練＋A+B 在整合版重跑，數字完全相同。

## 2. Reward 權重預檢
權重 shipped/passing/tardiness 從 0.5/0.3/0.2 改成 0.5/0.1/2.0：return 和 makespan 的相關從 −0.20 變成 −0.80，passing 雜訊的影響消失。

## 3. 實驗結果（5 seeds × 5 cases）

| 組別 | makespan | 延遲／件 | 對照（未訓練＋同樣 mask） |
|---|---|---|---|
| Justin 規則 | 2,596 | 46 | — |
| H1（舊 reward） | 3,386 | 427 | 3,054／271 |
| R1（新權重） | 3,108 | 251 | 3,054／271 |
| **R2（新權重＋entropy 0.03＋lr 1e-4）** | **2,964** | **217** | 3,054／271 |
| **R3（R2 設定＋機器特徵）** | **2,922** | 234 | 3,021／246 |

- 新權重修好了「越訓練越差」；加上防崩設定後才真正贏過未訓練。
- R2 五個 seed 都穩定（2,743～3,040）。
- 進步幅度小，離 Justin 還有約 330～370 ticks、延遲差約 4 倍。

## 4. 其他
- passing 下降不是因為選高風險加工模式（訓練後反而少選 fast），應該是品檢隨機。

## 建議下一步
1. 訓練更久（目前每個 seed 只有約 20 場 episode），看 R2／R3 會不會繼續進步。
2. entropy 還是降到約 1：可再提高 entropy 或降低 learning rate。
3. 和 Jian 討論：reward 權重的改法（動到 ADR 0039／0040）、digest 釘住檔案的問題、要不要把主目錄切到整合版。

詳細紀錄：`notes/runs/next_masks_baselines.md`。沒有 commit、push，也沒有傳訊息給任何人。
