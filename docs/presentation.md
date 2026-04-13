---
marp: true
theme: default
paginate: true
size: 16:9
math: mathjax
style: |
  section {
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    font-size: 24px;
  }
  h1 { font-size: 36px; color: #1a3a8c; }
  h2 { font-size: 30px; color: #3a62b8; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
  .highlight { color: #8B2500; font-weight: bold; }
  .good { color: #2E8B4A; }
  .bad { color: #8B0000; }
---

# 繁體中文 Tokenizer 碎片化對 LLM 語境腐化的影響

**Needle-in-a-Haystack 實驗與架構世代分析**

<br>

研究者：Dave Chen
日期：2026-04-13

---

## 研究動機

繁體中文因字形複雜度高，在主流 LLM tokenizer 中產生**更多 token**

**實測 Token Overhead（prompt 級，非字元級估算）：**

| Tokenizer 家族 | 代表模型 | 繁體 Overhead |
|---------------|---------|-------------|
| Gemma 3/4 | gemma3:4b 等 | **+5.9%** |
| LLaMA 3 | llama3.1:8b | **+7.2%** |
| Qwen 3.5 | qwen3.5:2b–35b | **+10.3%** |

> 65K 字元文件：繁體比簡體多消耗 2,714–3,849 tokens

**核心問題**：這些額外 token 是否使語境腐化（context rot）更早發生、更快惡化？

---

## 實驗設計

**方法**：Needle-in-a-Haystack（NIAH）檢索任務

**三種 Variant**（控制變因：僅改變文字腳本）

| Variant | Haystack | Question | 簡稱 |
|---------|----------|----------|------|
| 繁問繁答 | **繁體** | 繁體 | trad |
| 繁問簡答 | **簡體** | 繁體 | simp |
| 簡問簡答 | **簡體** | 簡體 | simp_q |

**實驗矩陣**：12 長度 × 11 位置 × 5 needle × 10 重複 = **1,320 trials / variant / model**

**模型**：16 個（Dense 1B–70B + MoE 2B/4B active），Q4_K_M 量化

---

## RQ1：Token 膨脹是否使腐化時機點提前？

**65K 字元下三 Variant 對比**（reeval，評估標準一致）

| 模型 | 繁問繁答 | 繁問簡答 | 簡問簡答 | 差距 |
|------|---------|---------|---------|------|
| gemma3:4b | 88.2% | 93.6% | 92.7% | <span class="good">**+4.5pp**</span> |
| llama3.1:8b | 92.7% | 94.5% | 96.4% | <span class="good">**+3.6pp**</span> |

- 三 variant 排序符合假說：**簡問簡答 ≈ 繁問簡答 > 繁問繁答**
- 問題語言影響小（0.9–1.9pp），**context 語言是主要因素**
- gemma3:4b 繁體在 12K 字元即開始滑落，簡體在 16K 仍穩定

---

## RQ2：Token 膨脹是否影響腐化速率？

**gemma3:4b — 退化最清晰的 benchmark**

| 字元數 | 繁問繁答 | 簡問簡答 | 差距 | 觀察 |
|--------|---------|---------|------|------|
| 12K | 99.1% | 100.0% | −0.9pp | 繁體略早滑落 |
| **65K** | **88.2%** | **92.7%** | **−4.5pp** | 差距最大 |
| 100K | 68.2% | 68.2% | **0pp** | 差距歸零 |
| 130K | 52.7% | 52.7% | **0pp** | 差距歸零 |

> 結果：**(a) 僅時機點提前（曲線左移）**，速率未加快
> 到 100K+，繁/簡 token 差（~2,700）相對總量（~75K+）已不顯著

---

## RQ2（續）：llama3.1:8b 的反轉現象

| 字元數 | 繁問繁答 | 簡問簡答 | 差距 |
|--------|---------|---------|------|
| 65K | 92.7% | 96.4% | <span class="good">−3.6pp</span>（符合假說）|
| 100K | 90.9% | 91.8% | −0.9pp |
| **130K** | **91.8%** | **82.7%** | <span class="bad">**+9.1pp（反轉）**</span> |

- 130K 繁體反勝簡體 9.1pp，**方向違反假說**
- 可能原因：訓練分布偏差、統計噪音（僅 110 筆 @130K）
- **無法歸類為 (a)(b)(c) 任一種退化模式**

---

## 位置敏感度：質性佐證

**gemma3:4b — 繁/簡的注意力分布差異**

| 位置 | 繁問繁答 | 繁問簡答（簡體 context）|
|------|---------|----------------------|
| 0.0（開頭）| 100% | 100% |
| 0.3 | 86.7% | 99.0% |
| **0.5（中段）** | **80.8%** | **98.0%** |
| 0.7 | 92.5% | 100% |
| 1.0（結尾）| 100% | 100% |

> 繁體：明顯 **Lost in the Middle**（中段 0.81）
> 簡體：幾乎平坦（0.98–1.0）
> 相同問題語言，不同 context 腳本 → **tokenizer overhead 加劇中段注意力稀釋**

---

## 探索性發現：架構世代壓倒 Tokenizer 效應

| 模型 | 架構 | overhead | 繁問繁答 @130K | 簡問簡答 @130K |
|------|------|----------|--------------|--------------|
| gemma3:4b | Dense 4B, Gemma 3 | +5.9% | <span class="bad">52.7%</span> | <span class="bad">52.7%</span> |
| gemma4:e2b | MoE **2B** active, Gemma 4 | +5.9% | <span class="good">**100.0%**</span> | <span class="good">**100.0%**</span> |
| qwen3.5:2b | Dense **2B**, Qwen 3.5 | +10.3% | <span class="good">**99.1%**</span> | <span class="good">**100.0%**</span> |
| llama3.1:8b | Dense 8B, LLaMA 3 | +7.2% | 91.8% | 82.7% |

- **Qwen3.5**：overhead 最高（+10.3%）卻完全免疫
- **Gemma4 e2b**：僅 2B 活躍參數，@130K = 100%
- 架構升級效果（+47pp）遠大於繁/簡差異（1.8–5.4pp）

---

## 為什麼新架構免疫？三個關鍵改進

### 1. 注意力機制：從 O(n²) 到混合架構

| 世代 | 機制 | 長 context 表現 |
|------|------|---------------|
| LLaMA 3.x | 純 softmax（O(n²)） | 128K 後品質衰減 |
| Gemma 4 | Sliding window + global（5:1 交錯） | 256K stable |
| Qwen 3.5 | Gated DeltaNet + full softmax（3:1 交錯） | **262K 原生，1M 可擴展** |

### 2. 位置編碼：自適應縮放

- **Gemma 4**：Proportional RoPE（p-RoPE）— 隨序列長度等比縮放
- **Qwen 3.5**：YaRN + NTK-aware — 高頻少縮、低頻多縮

### 3. KV Cache 共享（Gemma 4）
- 後 N 層複用前面層 K/V → 降低記憶體壓力 + 量化穩定性

---

## 新架構為何特別幫助繁體中文？

### 安全餘量吸收 overhead

```
繁體 130K 字元 ≈ 98K tokens

Gemma 3（有效 ctx ~77K tokens）→ 127% 利用率 → 已超限
Qwen 3.5（有效 ctx 262K tokens）→  37% 利用率 → 極安全
```

### 混合注意力消除 Lost in the Middle

Sliding window 層**只看局部窗口**，不受全局長度影響
→ 繁體多 5.9% tokens 不再稀釋中段注意力

### 線性注意力對 token 密度不敏感

GDN 壓縮為**固定大小 state**
→ 90K tokens（簡體）和 98K tokens（繁體）壓縮後相同

---

## 模型行為四類模式

| 模式 | 代表模型 | 特徵 | 適合研究？ |
|------|---------|------|----------|
| A 硬性上限 | gemma3:1b, qwen3:8b | 突然崩潰（ctx 截斷） | 否 |
| B 漸進退化 | gemma3:4b, llama3.1:8b | 準確率單調下降 | **最佳** |
| C 突發崩潰 | gemma3:27b | pad token 輸出退化 | 否 |
| D 近乎完美 | Gemma4, Qwen3.5 全系列 | 天花板效應 | 否（需 200K+）|

> 繁體 context rot 的**可觀測窗口**僅存在於模式 B 的特定 context 長度區間

---

## 宣稱 Context Window vs 實際有效範圍

| 模型 | 宣稱 ctx | 腐化起始點 | 有效利用率 |
|------|---------|----------|----------|
| gemma3:1b | 128K | 4,661 tokens | <span class="bad">**3.6%**</span> |
| gemma3:4b | 128K | 49,323 tokens | 38.5% |
| llama3.1:8b | 131K | 57,583 tokens | 43.9% |
| gemma3:12b | 128K | 75,847 tokens | 59.3% |
| Qwen3.5 全系列 | 262K | 無（>130K 測試範圍） | — |
| Gemma4 全系列 | 256K | 無（>130K 測試範圍） | — |

> **宣稱 128K ≠ 可靠使用 128K**
> 繁體輸入的額外 overhead 進一步壓縮實際可用範圍

---

## 理論限制（討論章節）

### Token 碎片化的雙效應（不可分離）

1. **Token 數量增加** → 佔用更多 context window
2. **語意密度降低** → 每 token 資訊量下降

兩者在現有設計下無法分離 → 作為機制詮釋，非獨立可測變數

### 英語樞紐機制（Latent Pivot）

- 模型中間層可能以英語語意空間推論
- 關閉 thinking mode 不影響此機制
- 可能部分抵消 token 膨脹效應
- 作為**無法排除的干擾因素**陳述

---

## 結論

### 已建立

1. **架構世代是壓倒性主因**，tokenizer overhead 最多是次要加重因子
2. **脆弱模型中繁體確實加速退化**（65K: +1.8–5.4pp），三 variant 排序符合假說
3. 效應模式為**曲線左移（onset 提前）**，非速率加快
4. **效應在極長 context 下非單調**（歸零或反轉）

### 未建立

- overhead-performance 定量關係（Qwen3.5 反例）
- llama3.1:8b @130K 反轉機制
- 前沿模型的真實臨界點（天花板效應）

### 實務建議

繁體中文長文件處理 → **優先選擇 Qwen 3.5 或 Gemma 4 系列**

---

## 下一步

**近期（進行中）：**
1. 補完 llama3.3:70b 簡問簡答（目前 502/1320 筆）
2. 調查 llama3.1:8b @130K 反轉現象

**中期：**
3. 擴充至 200K+ 字元，克服天花板效應
4. 英文 baseline（模型能力理論上限參考）

---

## 參考文獻

<div class="small">

| 主題 | 來源 |
|------|------|
| Gemma 3 架構 | Gemma Team. *Gemma 3 Technical Report.* arXiv:2503.19786 |
| Gemma 4 架構 | Google DeepMind. *Gemma 4 Model Card.* ai.google.dev |
| Qwen 3/3.5 架構 | Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388 |
| LLaMA 3 架構 | Grattafiori et al. *The Llama 3 Herd of Models.* arXiv:2407.21783 |
| YaRN 位置編碼 | Peng et al. *YaRN: Efficient Context Window Extension.* arXiv:2309.00071 |
| 架構效率比較 | arXiv:2604.07035 |

</div>

<br>

**實驗數據與程式碼**：https://github.com/dave0629g/context-rot-zh
