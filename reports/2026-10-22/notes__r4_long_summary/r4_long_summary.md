# R4-long 結果（2026-10-22）

**拿掉 passing 後拉長訓練沒有再進步：R4-long（2,878）和 R4（2,880）幾乎一樣，比 R2-long（2,783）差。目前最佳仍是 R2-long。**

| 組別 | makespan | 延遲／件 | 完成 64 件 |
|---|---|---|---|
| Justin 規則 | 2,596 | 46 | 5/5 |
| 未訓練＋masks | 3,054 | 271 | 24/25 |
| R4（passing 0，65k） | 2,880 | 193 | 24/25 |
| **R2-long（passing 0.1，262k）** | **2,783** | **128** | 24/25 |
| R4-long（passing 0，262k） | 2,878 | 153 | 23/25 |

- R4-long 各 seed：2,780、2,823、2,808、3,060、2,946；seed 105 有 2 個 case 卡在 62 件。
- 訓練曲線：多數 seed 在中段最好（例如 2,726），最後 1/4 又變差；seed 105 一直沒進步。
- 可能原因（未驗證）：後期不穩定（learning rate 固定），而評估用的是最後的 checkpoint。

## 建議下一步
1. 評估 R4-long 中段的 checkpoint（例如第 512、640、768 次更新），確認中段是否真的比較好。只要評估、不用訓練，約 1 小時。
2. 如果中段比較好：加 learning rate 衰減，或改用 validation 挑最好的 checkpoint。
3. 回覆 Jian：拿掉 passing 在 65k 有幫助（2,964 → 2,880），但長訓練沒有再進步。

詳細紀錄：`notes/runs/next_masks_baselines.md`。
