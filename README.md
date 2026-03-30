# 繁體中文 Context Rot 實驗

## 研究問題

繁體中文因 BPE tokenizer 碎片化（同一段文字比簡體多用 5-7% 的 tokens），是否影響 LLM 在長 context 下的資訊檢索能力？

## 實驗方法

使用 Needle-in-a-Haystack (NIAH) 測試：將一段虛構事實（needle）插入真實的維基百科文章（haystack）中，讓模型回答問題，檢驗是否能準確找到 needle。

### 三種實驗 Variant

| Variant | Haystack | Question | 模型回答語言 | 腳本 |
|---------|----------|----------|-------------|------|
| 繁問繁答 (`traditional`) | 繁體 | 繁體 | 繁體 | `03_run_experiment.py --variant traditional` |
| 簡問簡答 (`simplified_q`) | 簡體 | 簡體 | 簡體 | `06_hypothesis2_simp_question.py` |
| 繁問簡答 (`simplified`) | 簡體 | 繁體 | 簡體 | `03_run_experiment.py --variant simplified` |

Haystack 的繁/簡版本由 `02_build_haystacks.py` 同時產生（OpenCC t2s 轉換），每個 experiment 包含 `traditional.text` 和 `simplified.text` 兩個版本。

期望答案（`expected_answer`）定義在 `configs/wiki_articles_v2.json`，**永遠是繁體中文**。評估時會自動用 OpenCC 轉換為簡體作為額外比對候選。

### 實驗矩陣

- Context 長度：500, 2K, 4K, 6K, 8K, 12K, 16K, 24K, 32K, 65K, 100K, 130K 字元（12 級）
- Needle 位置：0%, 10%, 20%, ..., 100%（11 個位置）
- 每組合重複：10 次（不同 haystack）
- 每個 variant 每個模型：12 x 11 x 10 = **1,320 筆**

### Needle 清單（5 題）

| ID | 類型 | 問題 | 期望答案 |
|----|------|------|---------|
| N01 | 金額 | 鳳梨酥的全球年銷售額是多少？ | 四百七十三億元 |
| N02 | 人名 | 二零二四年特別成就獎頒給了誰？ | 陳美玲博士 |
| N03 | 面積 | 玉山山脈東側的礦物沉積區面積是多少？ | 三點七平方公里 |
| N04 | 數量 | 台灣黑熊的野外族群數量是多少？ | 八百五十隻 |
| N05 | 百分比 | 新型疫苗對高齡族群的保護效力是多少？ | 百分之九十二點六 |

### 測試模型

| 模型 | 大小 | Context Window |
|------|------|---------------|
| gemma3:4b | 4B | 131,072 |
| llama3.1:8b | 8B | 131,072 |
| qwen3:8b | 8B | 40,960 |
| qwen3.5:35b | 35B | 131,072 |
| gemma3:27b | 27B | 131,072 |
| llama3.3:70b | 70B | 131,072 |

## 環境需求

### 硬體
- GPU：需要能跑 Ollama 模型的 GPU（建議 VRAM 或統一記憶體 >= 16GB）

### 軟體
```bash
# Python 套件
pip install opencc-python-reimplemented matplotlib numpy

# Ollama（需先安裝並啟動）
ollama serve
ollama pull gemma3:4b    # 下載要測試的模型
```

## 執行步驟

### Step 1: 下載維基百科語料
```bash
python scripts/01_fetch_wiki_v2.py
```
從維基百科 API 下載 `configs/wiki_articles_v2.json` 指定的條目，自動清理數學標記。
輸出：`data/wiki_raw_v2/zh/*.txt`

### Step 2: 建構 Haystack
```bash
python scripts/02_build_haystacks.py
```
將維基百科文章拼接成指定長度的 haystack，在指定位置插入 needle，同時產生繁體和簡體版本。
輸出：`data/haystacks/experiments.jsonl`（~102MB）

### Step 3: 執行實驗
```bash
# 繁問繁答（繁體 context + 繁體 question）
python scripts/03_run_experiment.py --model gemma3:4b --variant traditional

# 繁問簡答（簡體 context + 繁體 question）
python scripts/03_run_experiment.py --model gemma3:4b --variant simplified

# 簡問簡答（簡體 context + 簡體 question）
python scripts/06_hypothesis2_simp_question.py --model gemma3:4b

# 三種 variant 一起跑（不建議，建議分開跑以利追蹤）
python scripts/03_run_experiment.py --model gemma3:4b --variant both
```

常用選項：
- `--resume`：從中斷處繼續（實驗隨時可暫停）
- `--max-experiments N`：只跑 N 筆（用於測試）

輸出：
- `results/{model}_results.jsonl`（traditional + simplified）
- `results/h2_{model}_results.jsonl`（simplified_q）

### Step 4: 分析結果
```bash
# 單一模型
python scripts/04_analyze.py --model gemma3:4b --reeval

# 所有模型
python scripts/04_analyze.py --all --reeval
```

`--reeval` 使用修正後的評估邏輯（中文數字正規化 + 同義詞比對）。
輸出：`results/analysis/{model}_analysis.json`

### Step 5: 產生圖表
```bash
python scripts/05_plot_results.py --models gemma3:4b llama3.1:8b
```

每模型產出 4 張圖（準確率 vs 長度/位置、熱力圖、各 Needle 準確率），
跨模型產出 3 張比較圖（多模型曲線、Token 比率、65k 準確率）。
輸出：`results/plots/*.png`

### 監控進度
```bash
watch -n 5 bash scripts/watch_progress.sh
```

## 評估方法

### 期望答案與正規化

期望答案定義在 `configs/wiki_articles_v2.json`，**原始格式為繁體中文**（例如「四百七十三億元」）。

評估時根據 variant 自動處理：

| Variant | 期望答案（原始） | 正規化比對候選 |
|---------|----------------|--------------|
| 繁問繁答 | 繁體 | 繁體原文 |
| 簡問簡答 | 繁體 | 繁體原文 + OpenCC 轉簡體 |
| 繁問簡答 | 繁體 | 繁體原文 + OpenCC 轉簡體 |

### 三層規則比對

模型回答以三層規則比對判定正確（任一層通過即為正確）：

1. **字串包含**：期望答案（含同義詞正規化後）是否出現在模型回答中
   - 同義詞替換：台幣/新台幣/元 → 統一 token、隻/只/頭 → 統一 token
2. **阿拉伯數字比對**：期望答案的數字集合是否為回答數字集合的子集
3. **中文數字正規化**：支援三種格式統一轉換後比對
   - 純中文：三點七 → 3.7、四百七十三億 → 47300000000
   - 純阿拉伯：直接提取 473、3.7 等
   - 混合格式：473億 → 47300000000（阿拉伯數字 + 中文單位）

## 初步結果（繁問繁答）

以下為 gemma3:4b 和 llama3.1:8b 在繁問繁答 variant 的完整結果（各 1,320 筆，含 100K、130K 壓力測試）。

### 整體準確率

| 模型 | 參數量 | 整體準確率 | Context Window | 最大測試 tokens |
|------|--------|-----------|---------------|----------------|
| gemma3:4b | 4B | **91.6%** | 131,072 | ~98K (75%) |
| llama3.1:8b | 8B | **97.1%** | 131,072 | ~115K (87%) |

### 準確率 vs Context 長度

| 長度 | gemma3:4b tokens | gemma3:4b 準確率 | llama3.1:8b tokens | llama3.1:8b 準確率 |
|------|----------------|----------------|-----------------|----------------|
| 500 | 466 | 100.0% | 544 | 94.5% |
| 2K | 1,595 | 100.0% | 1,866 | 98.2% |
| 4K | 3,105 | 100.0% | 3,634 | 99.1% |
| 6K | 4,648 | 100.0% | 5,425 | 100.0% |
| 8K | 6,139 | 100.0% | 7,174 | 100.0% |
| 12K | 9,148 | 99.1% | 10,663 | 100.0% |
| 16K | 12,169 | 97.3% | 14,205 | 100.0% |
| 24K | 18,171 | 95.5% | 21,172 | 100.0% |
| 32K | 24,260 | 98.2% | 28,324 | 98.2% |
| 65K | 49,062 | 88.2% | 57,248 | 92.7% |
| **100K** | **75,529** | **68.2%** | **88,040** | **90.9%** |
| **130K** | **98,288** | **52.7%** | **114,661** | **91.8%** |

### 關鍵發現

1. **Context rot 的壓力閾值不同**
   - gemma3:4b：65K 字元（~49K tokens, 37%）開始明顯下降，130K（~98K tokens, 75%）降至 52.7%
   - llama3.1:8b：即使 130K 字元（~115K tokens, 87%）仍維持 91.8%，衰退幅度遠小於 gemma3:4b

2. **模型大小的影響顯著**
   - 同為 100K 字元：gemma3:4b 68.2% vs llama3.1:8b 90.9%（差距 22.7pp）
   - 8B 模型的長 context 穩健性明顯優於 4B

3. **各 Needle 題型難度不同**

   | Needle | gemma3:4b | llama3.1:8b | 說明 |
   |--------|---------|---------|------|
   | N01 金額 | 92.4% | 93.6% | 數字干擾（haystack 中有其他金額） |
   | N02 人名 | **85.2%** | 98.5% | gemma3:4b 最弱，被其他人名誤導 |
   | N03 面積 | 95.8% | 96.6% | 穩定 |
   | N04 數量 | 90.2% | 97.0% | |
   | N05 百分比 | 94.3% | **100.0%** | llama3.1:8b 完美 |

4. **llama3.1:8b 在短 context（500 字元）反而較低（94.5%）**
   - 疑似短文本下模型傾向自由發揮而非嚴格檢索

### 圖表

分析圖表位於 `results/plots/`：

- `gemma3_4b_accuracy_vs_length.png` — gemma3:4b 準確率 vs 長度曲線
- `llama3.1_8b_accuracy_vs_length.png` — llama3.1:8b 準確率 vs 長度曲線
- `compare_accuracy_vs_length.png` — 兩模型對比
- `compare_65k_accuracy.png` — 65K 長度下各 variant 比較
- `*_heatmap.png` — 熱力圖（長度 × 位置）
- `*_needle_accuracy.png` — 各 Needle 準確率

## 目錄結構

```
context-rot-zh/
├── configs/
│   └── wiki_articles_v2.json      # 語料設定與 needle 定義
├── data/
│   ├── wiki_raw_v2/zh/            # 下載的維基百科文章（可重建，不入 git）
│   └── haystacks/experiments.jsonl # 建構的實驗資料（可重建，不入 git）
├── results/
│   ├── {model}_results.jsonl      # 繁問繁答 + 繁問簡答原始結果
│   ├── h2_{model}_results.jsonl   # 簡問簡答原始結果
│   ├── analysis/                  # 分析輸出 JSON
│   └── plots/                     # 圖表 PNG
├── scripts/
│   ├── 01_fetch_wiki_v2.py        # 下載維基百科語料
│   ├── 02_build_haystacks.py      # 建構 haystack + 插入 needle
│   ├── 03_run_experiment.py       # 執行主實驗（繁問繁答/繁問簡答）
│   ├── 04_analyze.py              # 分析結果 + 評估修正
│   ├── 05_plot_results.py         # 產生視覺化圖表
│   ├── 06_hypothesis2_simp_question.py  # 簡問簡答實驗
│   └── watch_progress.sh          # 即時進度監控
└── README.md
```

## Thinking 模型處理

qwen3 系列和 deepseek-r1 系列為 thinking 模型，實驗腳本會自動：
- 透過 Ollama API 參數 `"think": false` 關閉推理模式
- 確保 response 只包含答案，不含推理過程，不污染 prompt

## 可復現性

- 所有 random seed 固定（`configs/wiki_articles_v2.json` 中的 `random_seed`）
- 維基百科條目 ID 固定於設定檔
- 所有模型回答原樣保存於 JSONL，可用不同評估方法重新分析
- 評估修正只影響 `is_correct` 判定，不修改 `model_response`
