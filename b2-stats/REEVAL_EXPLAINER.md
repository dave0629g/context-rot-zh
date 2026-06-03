# 資料判定修正說明（給 claude.ai 評估論文修改用）

## TL;DR

本研究 `*_results.jsonl` 內每筆「模型回答是否正確」(`evaluation.is_correct`) 原本由 `03_run_experiment.py` 的 `evaluate_answer()` 規則式判定。後來在 `04_analyze.py` 中加入修正版 `reevaluate()`，新增三項規則來消除 false negative；但**先前未把修正回寫到 JSONL**，導致 raw JSONL 與 `results/analysis/` 統整檔之間口徑不一致。

現已將全部 34 個 JSONL（17 模型 × {main, h2}）以 `reevaluate()` 結果重寫，共改動 **4,656 筆 / 65,905 筆（7.1%）**，**全部都是 False → True 的 false negative 修正**。原始檔保留在 `results.backup_pre_reeval/`，並於 `evaluation.reevaluated: true` 留下旗標。

---

## 一、什麼改變了

| 欄位 | 改動方式 |
|---|---|
| `evaluation.is_correct` | **改寫**為 `reevaluate()` 的結果 |
| `evaluation.reevaluated` | **新增** `true` 旗標（冪等保護） |
| `evaluation.exact_match` | 保留原值（歷史記錄） |
| `evaluation.number_match` | 保留原值（歷史記錄） |
| `evaluation.char_overlap` | 保留原值（歷史記錄） |
| 其他欄位（model_response、token_count_* 等） | **不變** |

模型回答本身沒有被重新生成 — 只是用新規則重判一次。

---

## 二、為什麼要改：舊規則的三個破口

### Bug 1 ─ 中文數字無法對阿拉伯數字

舊規則只用 `re.findall(r"[\d.]+", text)` 抽出阿拉伯數字做集合比對。中文數字字串（「三點七」「四百七十三億」「百分之九十二點六」）抽出來是空集合，跟模型輸出的 `3.7`、`47,300,000,000` 無法配對。

```
expected = "三點七平方公里"     → 抽出 ∅
response = "3.7平方公里"        → 抽出 {3.7}
→ number_match = False（誤判）
```

修正：`extract_all_numbers()` 先把中文數字（含「萬／億」大單位、小數點、「百分之」前綴）轉成阿拉伯數字再比對。

```
expected → {3.7}
response → {3.7}
→ number_match = True ✓
```

### Bug 2 ─ 同義詞與異體字 miss

needle 答案使用台灣慣用語（「新台幣」「隻」），模型答案常用替代詞（「元」「只」），舊規則純字串包含比對會 miss：

```
"新台幣四百七十三億元" vs "473 億元"  → miss
"八百五十隻" vs "約八百五十只"        → miss（隻/只 異體字）
```

修正：`normalize_for_match()` 把同義詞統一替換成 token：
- `元 / 台幣 / 新台幣` → `＄`
- `隻 / 只 / 頭` → `＃`

### Bug 3 ─ simplified_q variant 沒做 OpenCC t2s 轉換（最嚴重）

舊規則只對 `variant == "simplified"` 做 t2s（繁體 expected → 簡體再比對），但**沒處理 `simplified_q`**（簡問繁答 / 簡問簡答的問題變項）。當 needle 是繁體、回答是簡體時：

```
expected = "陳美玲博士"
response = "陈美玲博士"
→ exact_match = False（漢字不同）
```

修正：`reevaluate()` 對所有非繁體 variant 都把 expected 過 t2s 加入候選比對清單。

> 這是 commit `0c86a2f`（2026-04-13）"修復 simplified_q reeval" 的主要內容。

---

## 三、影響規模（per-model）

| model | n_total | raw acc | reeval acc | Δ acc | % records changed |
|---|---:|---:|---:|---:|---:|
| **llama3.1:70b** | 3,960 | 0.5568 | 0.9773 | **+0.4205** | **42.05%** |
| **llama3.3:70b** | 3,960 | 0.8000 | 0.9846 | +0.1846 | 18.46% |
| **gemma3:27b**   | 3,960 | 0.7646 | 0.8861 | +0.1215 | 12.15% |
| gemma3:1b        | 2,981 | 0.6923 | 0.7926 | +0.1003 | 10.03% |
| qwen3.5:2b       | 3,960 | 0.9232 | 0.9995 | +0.0763 | 7.63% |
| llama3.1:8b      | 3,960 | 0.9056 | 0.9712 | +0.0657 | 6.57% |
| gemma3:4b        | 3,960 | 0.8576 | 0.9227 | +0.0652 | 6.52% |
| qwen3.5:9b       | 3,960 | 0.9462 | 1.0000 | +0.0538 | 5.38% |
| gemma3:12b       | 3,960 | 0.9290 | 0.9768 | +0.0477 | 4.77% |
| qwen3.5:27b      | 3,960 | 0.9684 | 0.9995 | +0.0311 | 3.11% |
| qwen3.5:35b      | 3,960 | 0.9742 | 1.0000 | +0.0258 | 2.58% |
| qwen3:8b         | 3,300 | 0.9348 | 0.9433 | +0.0085 | 0.85% |
| gemma4:26b       | 3,960 | 0.9934 | 0.9947 | +0.0013 | 0.13% |
| gemma4:31b       | 3,960 | 0.9997 | 1.0000 | +0.0003 | 0.03% |
| gemma4:e2b / e4b / qwen3.5:4b | 3,960 ×3 | ≈1.000 | ≈1.000 | 0 | 0% |

**llama3.1:70b 為何特別嚴重（42%）**：該模型偏好用阿拉伯數字 + 英文單位回答（如「3.7」「47.3 billion 元」），但 needle 答案大量是中文數字（「三點七」「四百七十三億」）。舊規則三個破口剛好全踩。其他模型（Gemma 系列、Qwen 系列）較常照抄繁體中文寫法，所以漏判較少。

**所有改動都是 False → True**（共 4,656 筆，0 筆 True → False）。即修正後 accuracy **只升不降**。

---

## 四、對既有分析的影響

### 不受影響（已用 reeval 口徑）
- `results/analysis/*_analysis.json`：04_analyze.py 在生成這些 JSON 時已內部跑過 reevaluate，所以 accuracy_by_length、breakpoints、rot_coefficient 等數字本就是 reeval 後的，與本次修正一致。
- 本研究 `b2-stats/` 內 RQ1/RQ2/RQ3 的所有統計：`compute_model_metrics.py` 強制呼叫 reevaluate，所以結果與 analysis JSON 對齊。
- 任何引用 `analysis JSON` 或 `b2-stats CSV` 的圖表、表格、敘述。

### 可能受影響（若論文寫作時讀的是 raw JSONL）
- 早期版本若有任何「直接讀 raw is_correct 計算 accuracy」的描述，數字會偏低（尤其是 llama 系列）。
- 文獻表格中如果出現過 llama3.1:70b 約 55%、llama3.3:70b 約 80%、gemma3:27b 約 76% 這類數字 — 那是 raw 口徑，**應替換為 reeval 後的 97.7% / 98.5% / 88.6%**。

---

## 五、論文章節可能要修改的清單

### 方法章（Methodology）

1. **評估流程**：需明確說明採用「兩階段規則式判定」：
   - 階段一：實驗執行時 `evaluate_answer()` 即時記錄 exact_match / number_match。
   - 階段二：分析時 `reevaluate()` 加入 (a) 中文數字 ↔ 阿拉伯數字轉換、(b) 同義詞 / 異體字統一、(c) OpenCC t2s 對非繁體 variant 的雙向比對；以階段二結果作為最終判定。
2. **評估規則清單**：建議列出 Bug 1/2/3 對應的正規化規則（中文數字、同義詞、t2s），讓口試 / 評審者可以追溯每個比對的依據。
3. **Reproducibility 段落**：點出 `results.backup_pre_reeval/` 保有 raw 判定可供獨立重判。

### 結果章（Results）

1. **整體 accuracy 表**：以 reeval 後數字為主（17 模型如上表第三欄）。
2. **per-length / per-position 圖**：用 `results/analysis/*.json` 出圖即可（這份本來就是 reeval 後）。
3. **B2 推論統計**（RQ1/2/3）：不需重跑，因為 `compute_model_metrics.py` 一開始就走 reeval。

### 討論章（Discussion）

1. **「LLaMA 系列衰減較輕」之類的描述**：reeval 後 llama3.1:70b 與 llama3.3:70b 都在 ≥ 97% baseline accuracy，遠高於 raw 結果。若舊版討論基於 raw 數字推論「LLaMA 系列 baseline 顯著低於 Gemma」，需修正。
2. **fertility / token-overhead 與 accuracy 關係**：本研究結論（RQ2 LRT p < .001 支持 token_overhead 為獨立因素）成立的前提是 reeval 口徑。如果方法章先說明這點，討論章不用大改。
3. **Context Rot 量級**：reeval 後多數模型 baseline 接近 100%，所以衰減量值會比 raw 更明顯（從近滿分掉下來）。

### 限制章（Limitations）

1. 即使 reeval 後仍是規則式判定，未涵蓋「語義對但用詞完全不同」的少數情況（例如把「四百七十三億」回答成「47.3 billion」缺單位的變形）。
2. 若論文要強調此一限制，可同時報告 `char_overlap`（部分匹配）作為輔助指標。

### 附錄 / Reproducibility

1. 在附錄補一個小段「Evaluation Rule Patches」，列 Bug 1/2/3 的具體規則與修正前後 accuracy 對照（即本文件第三節的表）。
2. 提供 commit SHA：
   - `ae404c1` 加入中文數字正規化與同義詞比對
   - `b2c26d8` 加入混合格式數字解析（億 → 1e9）
   - `0c86a2f` 修復 simplified_q 的 t2s reeval
3. 說明本次回寫 JSONL 的批次操作（一次性 idempotent 重寫，並備份原檔）。

---

## 六、給 claude.ai 的提問模板

> 我已將實驗 JSONL 的 `is_correct` 欄位用修正後的判定規則重寫（細節見上）。請你：
>
> 1. 評估我論文「方法章」是否需要新增一節描述「兩階段評估」流程；給我建議的小節結構與大約字數。
> 2. 在「結果章」中，哪些表格 / 段落可能因為直接報 raw accuracy 而需要替換？我會把目前的論文檔貼上來。
> 3. 「討論章」中若提到 LLaMA 系列 baseline accuracy 偏低，要怎麼改寫才能與 reeval 後（≥ 97%）一致？
> 4. 「限制章」中要怎麼陳述「規則式判定仍可能 miss 語義同義但詞彙不同的案例」？
> 5. 是否建議我在附錄補一段「Evaluation Rule Patches」並附上修正前後對照表？

---

## 附：原始 vs 修正的具體案例

| variant      | expected            | response               | raw is_correct | reeval is_correct | 修正規則            |
|--------------|---------------------|------------------------|:---------------:|:------------------:|---------------------|
| simplified_q | 三點七平方公里      | 3.7平方公里            | ✗ False        | ✓ True             | Bug 1（中文數字）   |
| simplified_q | 陳美玲博士          | 陈美玲博士             | ✗ False        | ✓ True             | Bug 3（t2s）        |
| traditional  | 八百五十隻          | 約八百五十只。         | ✗ False        | ✓ True             | Bug 2（隻 ↔ 只）    |
| simplified_q | 百分之九十二點六    | 百分之九十二点六       | ✗ False        | ✓ True             | Bug 3（t2s）        |
| traditional  | 新台幣四百七十三億元 | 47.3 billion 元        | ✗ False        | ✓ True             | Bug 1 + Bug 2 結合  |
