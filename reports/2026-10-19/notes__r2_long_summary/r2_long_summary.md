# R2-long 結果（2026-10-19）

**訓練拉長 4 倍（65k → 262k ticks）後明顯進步，5 個 seed 都穩定，離 Justin 規則的差距縮小到約 7%。**

| 組別 | 平均 makespan | 延遲／件 | 完成 64 件 |
|---|---|---|---|
| Justin 規則 | 2,596 | 46 | 5/5 |
| 未訓練＋A+B | 3,054 | 271 | 24/25 |
| R2（65k ticks） | 2,964 | 217 | 24/25 |
| **R2-long（262k ticks）** | **2,783** | **128** | 24/25 |

- 比未訓練：makespan −9%，延遲 −53%。
- 5 個 seed 的 makespan 都在 2,758～2,822。
- 離 Justin：makespan 還差約 190 ticks（7%），延遲約 2.8 倍。
- 訓練曲線還沒變平，再拉長可能還會進步。
- R3-long 依指示沒有跑。

設定：masks A+B、reward 權重 0.5/0.1/2.0、entropy 0.03、lr 1e-4、64 件、5 seeds × 5 cases。
詳細紀錄：`notes/runs/next_masks_baselines.md`。沒有 commit、push，也沒有傳訊息給任何人。
