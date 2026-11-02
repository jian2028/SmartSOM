# 分析：雙胞胎機器失衡（Step 3a seed 102 case 0）

日期：2026-10-15 ｜ 資料：`runs/replays/step3a_seed102/.../evidence/case-0000/trace.jsonl`（51/64 交付）

## 現象

每道工序有兩台機器可做，加工時間完全相同（抽查 120 個 job×工序，兩台的名義時間都一樣）。網路卻幾乎只用其中一台：

| 工序 | 機器 | 實際送達 job 數 | 使用率 | 載貨車第一次選的目的地 |
|---|---|---|---|---|
| operation 1 | machine 1 ／ machine 8 | 0 ／ 62 | 0% ／ 49% | 6 ／ 48 |
| operation 2 | machine 7 ／ machine 2 | 8 ／ 50 | 9% ／ 40% | 25 ／ 40 |
| operation 3 | machine 6 ／ machine 3 | 1 ／ 58 | 1% ／ 62% | 8 ／ 55 |
| operation 4 | machine 4 ／ machine 5 | 7 ／ 48 | 6% ／ 52% | 22 ／ 37 |

（送達數含重工，所以總和大於 64。）

- **不是距離造成的**：operation 1 選中的 machine 8 平均反而比較遠（10.7 vs 8.9 格）。
- **選中的機器常常正在忙、雙胞胎卻閒置**：載貨車第一次決定目的地時，約 1/3 的情況是「選中的那台在加工、另一台空著」（operation 1：20/54；2：16/65；3：29/63；4：21/59）。
- **第一次選擇比較平均，最後送達卻幾乎全集中在一台**：載貨車一路上一直改目的地（見 Demo 第 5 點），最後收斂到「偏好」的那台。

## 原因：派車候選看不到目的地機器的狀態

`src/smartsom/learning/production_models.py` 的 `encode()` 中，派車候選（目的地）的特徵只有：

- 距離、port 座標、目的地 buffer 已有的 job 數、容量、是否已滿、是否為目前目標
- 目的地 buffer 和 port 的 one-hot（身分）
- 所載 job 的工序資訊（對兩個雙胞胎候選完全相同）

**沒有「這個 buffer 後面那台機器是否在忙、還要多久、排了幾件」。** 機器狀態只出現在共用的 context 裡，以機器 ID 排列。
網路必須自己學會「buffer_003 ↔ machine_001」的對應，再去 context 找 machine_001 的狀態。用 64×64 的網路、每個 seed 約 16 場 episode，這很難學到。

## 建議

1. **在派車候選加上目的地機器的特徵**：machine 狀態（閒置／加工中）、剩餘加工時間、前 buffer 排隊數。
   - 改的是 `learning/production_models.py` 的觀測編碼（不是 simulation core），但會改變模型輸入，舊 checkpoint 不相容。
   - 這是 Jian 的程式，**要先和 Jian 確認**。
2. 改完重跑 Step 3a 設定（5 seeds），比較雙胞胎使用率和交付量。

## 限制

只分析了一個 case（seed 102 case 0）；其他 seed 的評估錄影已刪除，需要時可重新評估產生。
