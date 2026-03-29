# 繁體中文 Context Rot 實驗

## 研究問題

繁體中文因 BPE tokenizer 碎片化（同一段文字比簡體多用 5-7% 的 tokens），是否影響 LLM 在長 context 下的資訊檢索能力？

## 實驗方法

使用 Needle-in-a-Haystack (NIAH) 測試：將一段虛構事實（needle）插入真實的維基百科文章（haystack）中，讓模型回答問題，檢驗是否能準確找到 needle。

### 三種實驗 Variant

| Variant | Context 語言 | Question 語言 | 腳本 |
|---------|-------------|-------------|------|
| 繁問繁答 (`traditional`) | 繁體 | 繁體 | `03_run_experiment.py --variant traditional` |
| 簡問簡答 (`simplified_q`) | 簡體 | 簡體 | `06_hypothesis2_simp_question.py` |
| 繁問簡答 (`simplified`) | 簡體 | 繁體 | `03_run_experiment.py --variant simplified` |

### 實驗矩陣

- Context 長度：500, 2K, 4K, 6K, 8K, 12K, 16K, 24K, 32K, 65K 字元（10 級）
- Needle 位置：0%, 10%, 20%, ..., 100%（11 個位置）
- 每組合重複：10 次（不同 haystack）
- 每個 variant 每個模型：10 x 11 x 10 = **1,100 筆**

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

模型回答以三層規則比對判定正確：

1. **字串包含**：期望答案（含同義詞正規化）是否出現在回答中
   - 同義詞：台幣/新台幣/元、隻/只/頭
2. **阿拉伯數字比對**：期望數字集合是否為回答數字集合的子集
3. **中文數字正規化**：將中文數字/混合格式統一轉換後比對
   - 三點七 → 3.7、四百七十三億 → 47300000000、473億 → 47300000000

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
