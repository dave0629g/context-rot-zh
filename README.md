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

Needle 是插入 haystack 中的虛構事實句，模型需從長文中找到該事實並回答問題。

| ID | 類型 | Needle（插入 haystack 的事實句） | 問題 | 期望答案 |
|----|------|--------------------------------|------|---------|
| N01 | 金額 | 根據內部研究報告，鳳梨酥的全球年銷售額在二零二三年達到了新台幣四百七十三億元。 | 鳳梨酥的全球年銷售額是多少？ | 四百七十三億元 |
| N02 | 人名 | 瑞典皇家科學院宣布，二零二四年的特別成就獎頒發給了在深海珊瑚復育領域有卓越貢獻的陳美玲博士。 | 二零二四年特別成就獎頒給了誰？ | 陳美玲博士 |
| N03 | 面積 | 根據最新的地質調查，玉山山脈的東側發現了一處面積約為三點七平方公里的罕見礦物沉積區。 | 玉山山脈東側的礦物沉積區面積是多少？ | 三點七平方公里 |
| N04 | 數量 | 在最近一次的國際會議上，與會學者一致認為台灣黑熊的野外族群數量已恢復至約八百五十隻。 | 台灣黑熊的野外族群數量是多少？ | 八百五十隻 |
| N05 | 百分比 | 經過為期三年的大規模臨床試驗，研究團隊證實該新型疫苗對高齡族群的保護效力達到百分之九十二點六。 | 新型疫苗對高齡族群的保護效力是多少？ | 百分之九十二點六 |

### 已知限制：Haystack 中的干擾資訊（Distractor Information）

部分 Wikipedia 語料中包含與 needle 主題相似的真實內容，這些內容可能成為干擾項（distractors），導致模型找到真實段落而非我們插入的 needle：

| Needle | 干擾源 | 嚴重度 |
|--------|--------|--------|
| N01「473」| 14,473（二戰俘虜數）、1473（航海年份） | 低：上下文差異大 |
| N02 陳美玲 | 無干擾 | — |
| N03「3.7」| 3.7°C（氣溫）、3.7%（產量比例） | 中：數字同但單位不同 |
| N04「黑熊」| `24_台灣地理.txt` 提到真實的「臺灣黑熊」 | **高：直接相關** |
| N05 百分比 | 無干擾 | — |

**對研究結論的影響：**

- 干擾對三種 variant 的影響是**對等的**（同一個 haystack，同一個干擾源），因此 variant 之間的相對比較仍然有效
- N04 的絕對準確率偏低是干擾造成的，不代表模型能力較差
- 錯誤分析中 N04 出現的「4350萬」「150隻」「60-80隻」等幻覺數字，可能源自模型找到了 haystack 中真實的黑熊相關段落

**未來改進：** 建構 haystack 前應增加干擾檢測步驟（distractor detection），確認語料中不含與 needle 相似的資訊。

### 測試模型

| 模型 | 大小 | Context Window | Thinking |
|------|------|---------------|---------|
| gemma3:1b | 1B | 131,072 | — |
| gemma3:4b | 4B | 131,072 | — |
| gemma3:12b | 12B | 131,072 | — |
| gemma3:27b | 27B | 131,072 | — |
| gemma4:e2b | 2B（edge）| 131,072 | ✓（關閉）|
| gemma4:e4b | 4B（edge）| 131,072 | ✓（關閉）|
| gemma4:26b | 26B（MoE, 4B active）| 262,144 | ✓（關閉）|
| gemma4:31b | 31B（dense）| 262,144 | ✓（關閉）|
| llama3.1:8b | 8B | 131,072 | — |
| llama3.3:70b | 70B | 131,072 | — |
| qwen3:8b | 8B | 40,960 | ✓（關閉）|
| qwen3.5:2b | 2B | 262,144 | ✓（關閉）|
| qwen3.5:4b | 4B | 262,144 | ✓（關閉）|
| qwen3.5:9b | 9B | 262,144 | ✓（關閉）|
| qwen3.5:27b | 27B | 262,144 | ✓（關閉）|
| qwen3.5:35b | 35B | 262,144 | ✓（關閉）|

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

### 實驗執行參數（本次實驗固定設定）

| 參數 | 值 | 說明 |
|------|-----|------|
| Ollama 版本 | 0.18.1 | |
| 模型量化格式 | Q4_K_M | 所有模型統一使用 Ollama 預設 Q4_K_M |
| `OLLAMA_FLASH_ATTENTION` | 未設定（預設關閉）| |
| `OLLAMA_KV_CACHE_TYPE` | 未設定（預設 f16）| |
| API timeout | 1200 秒 | 70B 模型 130K context 每筆約 650s+ |
| `num_ctx`（context window）| 模型預設值 | 見下表 |
| `temperature` | 0.0 | 固定為 greedy decoding，確保可復現 |
| `think` | false | qwen3 / deepseek-r1 系列關閉推理模式 |

> 以上參數在實驗期間不做調整。若未來調整（如啟用 Flash Attention），需重新執行並另行標注。

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
- `--lengths 100000,130000`：只跑指定長度
- `--max-experiments N`：只跑 N 筆（用於測試）

輸出：
- `results/{model}_results.jsonl`（traditional + simplified）
- `results/h2_{model}_results.jsonl`（simplified_q）

#### 自動排程（一次跑完所有實驗）

```bash
# 背景執行，關掉終端也不會停
nohup bash scripts/run_all.sh > /tmp/run_all.log 2>&1 &

# 查看 log
tail -f /tmp/run_all.log
```

`run_all.sh` 會依序執行所有模型的所有 variant，每個實驗完成後自動 `git commit`。單個實驗失敗不影響後續。可隨時 `kill` 暫停，重跑時自動 `--resume` 跳過已完成的部分。

#### 操作技巧

```bash
# 查看目前正在跑的實驗
ps aux | grep 03_run_experiment | grep -v grep

# 暫停目前的實驗（找到 PID 後 kill）
kill <PID>

# 從手動切換到自動排程：
# 1. 先 kill 目前的實驗（或等它自然跑完）
# 2. 啟動 run_all.sh，它會自動 --resume 跳過已完成的
nohup bash scripts/run_all.sh > /tmp/run_all.log 2>&1 &

# 監控進度（三種方式）
watch --color -n 5 bash scripts/watch_progress.sh   # 即時進度表（含顏色）
python3 scripts/estimate_time.py                      # 時間估算（含顏色）
tail -f /tmp/run_all.log                              # 原始 log
```

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

## 實驗進度

| 模型 | 大小 | 繁問繁答 | 繁問簡答 | 簡問簡答 |
|------|------|---------|---------|---------|
| gemma4:e2b | E2B | 1,320 筆 ✓ | 進行中 | 進行中 |
| gemma4:e4b | E4B | 1,320 筆 ✓ | 待執行 | 待執行 |
| gemma4:26b | 26B | 1,320 筆 ✓ | 待執行 | 待執行 |
| gemma4:31b | 31B | 進行中（100K 部分）| 待執行 | 待執行 |
| qwen3.5:2b | 2B | 待執行 | 待執行 | 待執行 |
| qwen3.5:4b | 4B | 待執行 | 待執行 | 待執行 |
| qwen3.5:9b | 9B | 待執行 | 待執行 | 待執行 |
| qwen3.5:27b | 27B | 待執行 | 待執行 | 待執行 |
| qwen3.5:35b | 35B | 220 筆（100K+130K）| 待執行 | 待執行 |
| gemma3:1b | 1B | 待執行 | 待執行 | 待執行 |
| gemma3:4b | 4B | 1,320 筆 ✓ | 1,100 筆 ✓ | 1,100 筆 ✓ |
| gemma3:12b | 12B | 待執行 | 待執行 | 待執行 |
| gemma3:27b | 27B | 220 筆（100K+130K，資料異常⚠️）| 待執行 | 待執行 |
| llama3.1:8b | 8B | 1,320 筆 ✓ | 1,100 筆 ✓ | 1,100 筆 ✓ |
| llama3.3:70b | 70B | 144 筆（100K + 130K 部分）| 待執行 | 待執行 |
| qwen3:8b | 8B | 1,100 筆（500–65K）✓ | 125 筆（部分）| 待執行 |

## 結果（已完成部分）

準確率以修正版評估邏輯計算（`--reeval`：中文數字正規化 + 同義詞比對）。

### 整體檢索成功率

| 模型 | 繁問繁答 | 繁問簡答 | 簡問簡答 | Tokenizer Overhead |
|------|---------|---------|---------|-------------------|
| gemma4:e2b | **99.8%** | — | — | — |
| gemma4:e4b | **99.9%** | — | — | — |
| gemma4:26b | **98.9%** | — | — | — |
| qwen3.5:35b | **100.0%**（100K+130K）| — | — | — |
| gemma3:4b | 91.6% | **99.1%** | 96.8% | +5.7% |
| llama3.1:8b | 97.1% | **98.8%** | 95.3% | +7.1% |
| qwen3:8b | 94.1%（≤32K 全滿，65K 崩潰）| 100.0%（部分）| — | +13.6% |

> gemma4 系列（thinking 模型，以 prompt 注入關閉推理）在繁體中文長 context 下表現極佳。

### 準確率 vs Context 長度

#### gemma4:e2b

| 長度 | 繁問繁答 |
|------|---------|
| 500–24K | 100.0% |
| 32K | 99.1% |
| 65K | 100.0% |
| 100K | 99.1% |
| 130K | 100.0% |

#### gemma4:e4b

| 長度 | 繁問繁答 |
|------|---------|
| 500、4K–130K | 100.0% |
| 2K | 99.1% |

#### gemma4:26b

| 長度 | 繁問繁答 |
|------|---------|
| 500–12K | 100.0% |
| 16K | 99.1% |
| 24K–32K | 100.0% |
| 65K | 98.2% |
| 100K | 97.3% |
| 130K | 92.7% |

> gemma4:26b（MoE 架構，4B active params）在 130K 字元下仍維持 92.7%，對比 gemma3:4b（dense 4B）同長度只有 52.7%，架構差異顯著。

#### gemma3:4b

| 長度 | 繁問繁答 | 繁問簡答 | 簡問簡答 |
|------|---------|---------|---------|
| 500 | 100.0% | 100.0% | 100.0% |
| 2K | 100.0% | 100.0% | 100.0% |
| 4K | 100.0% | 100.0% | 100.0% |
| 6K | 100.0% | 100.0% | 100.0% |
| 8K | 100.0% | 100.0% | 100.0% |
| 12K | 99.1% | 100.0% | 100.0% |
| 16K | 97.3% | 97.3% | 98.2% |
| 24K | 95.5% | 100.0% | 90.9% |
| 32K | 98.2% | 100.0% | 96.4% |
| 65K | 88.2% | **93.6%** | 82.7% |
| 100K | 68.2% | — | — |
| 130K | 52.7% | — | — |

#### llama3.1:8b

| 長度 | 繁問繁答 | 繁問簡答 | 簡問簡答 |
|------|---------|---------|---------|
| 500 | 94.5% | 97.3% | 92.7% |
| 2K | 98.2% | 99.1% | 95.5% |
| 4K | 99.1% | 100.0% | 98.2% |
| 6K | 100.0% | 100.0% | 98.2% |
| 8K | 100.0% | 100.0% | 98.2% |
| 12K | 100.0% | 100.0% | 99.1% |
| 16K | 100.0% | 100.0% | 96.4% |
| 24K | 100.0% | 100.0% | 94.5% |
| 32K | 98.2% | 97.3% | 95.5% |
| 65K | 92.7% | **94.5%** | 84.5% |
| 100K | 90.9% | — | — |
| 130K | 91.8% | — | — |

#### qwen3:8b（繁問繁答，最大 context window 40,960 tokens）

| 長度 | 繁問繁答 | 備註 |
|------|---------|------|
| 500–32K | 100.0% | 穩定 |
| **65K** | **40.9%** | ~36K tokens ≈ 88% context window，嚴重崩潰 |

> qwen3:8b 的 context window 只有 40,960 tokens，65K 字元繁體（~36K tokens）已逼近上限，是 context window 截斷效應而非 context rot。

#### qwen3.5:35b（繁問繁答，部分資料：僅 100K+130K）

| 長度 | 繁問繁答 | 備註 |
|------|---------|------|
| 100K | **100.0%** | ~70K tokens，26% context window |
| 130K | **100.0%** | ~91K tokens，35% context window |

#### llama3.3:70b（繁問繁答，部分資料：100K 完整 + 130K 部分）

| 長度 | 繁問繁答 | 備註 |
|------|---------|------|
| 100K | 97.3% | 完整 110 筆 |
| 130K | 67.7% | 34 筆（進行中）|

### 關鍵發現

1. **gemma4 系列在長 context 下表現優異**
   - gemma4:e2b（2B edge）全長度維持 99.8%，包含 130K
   - gemma4:e4b（4B edge）幾乎全部 100%
   - gemma4:26b（26B MoE）在 130K 仍有 92.7%，遠優於同 active params 的 gemma3:4b（52.7%）
   - 推測 MoE 架構與 gemma4 訓練改善對長 context 有顯著幫助

2. **繁體 context rot 假說在 gemma3:4b 得到支持**
   - 65K 字元下：繁問繁答 88.2% vs 繁問簡答 93.6%（差距 **+5.4pp**）
   - llama3.1:8b 差距較小（+1.8pp），整體更穩健
   - gemma4 系列差距目前尚無資料（繁問簡答/簡問簡答尚在執行中）

3. **模型大小是長 context 穩健性的關鍵因素（gemma3 系列內）**
   - 100K：gemma3:4b 68.2% vs llama3.1:8b 90.9%（差距 **22.7pp**）
   - 130K：gemma3:4b 崩潰至 52.7%；llama3.1:8b 仍維持 91.8%

4. **qwen3:8b 的 context window 限制導致 65K 嚴重崩潰**
   - 0–32K 全部 100%，65K 驟降至 40.9%（非漸進式衰退）
   - tokenizer overhead 最高（+13.6%），65K 字元已佔 88% context window

5. **gemma3:27b 資料異常（⚠️ 待重跑）**
   - 100K+130K 結果幾乎全部為 `<pad>` token，推論過程異常，資料不可信

6. **llama3.3:70b 初步結果**
   - 100K 達 97.3%，表現良好；130K 降至 67.7%（資料仍不完整）

### 圖表

#### 跨模型比較：準確率 vs Context 長度

![跨模型準確率 vs 長度](results/plots/compare_accuracy_vs_length.png)

#### 字元數 → 實際 Token 數對照（各模型 Tokenizer 差異）

![Token 對照圖](results/plots/compare_token_map.png)

#### Tokenizer Overhead（繁體 vs 簡體 token 數比較）

![Token 比率](results/plots/compare_token_ratio.png)

#### 65K 字元下各 Variant 準確率

![65K 準確率](results/plots/compare_65k_accuracy.png)

#### gemma4:e2b 個別圖表

![gemma4:e2b 準確率 vs 長度](results/plots/gemma4_e2b_accuracy_vs_length.png)

![gemma4:e2b 準確率 vs 位置](results/plots/gemma4_e2b_accuracy_vs_position.png)

![gemma4:e2b 熱力圖](results/plots/gemma4_e2b_heatmap.png)

![gemma4:e2b Needle 準確率](results/plots/gemma4_e2b_needle_accuracy.png)

#### gemma4:e4b 個別圖表

![gemma4:e4b 準確率 vs 長度](results/plots/gemma4_e4b_accuracy_vs_length.png)

![gemma4:e4b 準確率 vs 位置](results/plots/gemma4_e4b_accuracy_vs_position.png)

![gemma4:e4b 熱力圖](results/plots/gemma4_e4b_heatmap.png)

![gemma4:e4b Needle 準確率](results/plots/gemma4_e4b_needle_accuracy.png)

#### gemma4:26b 個別圖表

![gemma4:26b 準確率 vs 長度](results/plots/gemma4_26b_accuracy_vs_length.png)

![gemma4:26b 準確率 vs 位置](results/plots/gemma4_26b_accuracy_vs_position.png)

![gemma4:26b 熱力圖](results/plots/gemma4_26b_heatmap.png)

![gemma4:26b Needle 準確率](results/plots/gemma4_26b_needle_accuracy.png)

#### gemma3:4b 個別圖表

![gemma3:4b 準確率 vs 長度](results/plots/gemma3_4b_accuracy_vs_length.png)

![gemma3:4b 準確率 vs 位置](results/plots/gemma3_4b_accuracy_vs_position.png)

![gemma3:4b 熱力圖](results/plots/gemma3_4b_heatmap.png)

![gemma3:4b Needle 準確率](results/plots/gemma3_4b_needle_accuracy.png)

#### llama3.1:8b 個別圖表

![llama3.1:8b 準確率 vs 長度](results/plots/llama3.1_8b_accuracy_vs_length.png)

![llama3.1:8b 準確率 vs 位置](results/plots/llama3.1_8b_accuracy_vs_position.png)

![llama3.1:8b 熱力圖](results/plots/llama3.1_8b_heatmap.png)

![llama3.1:8b Needle 準確率](results/plots/llama3.1_8b_needle_accuracy.png)

#### qwen3:8b 個別圖表

![qwen3:8b 準確率 vs 長度](results/plots/qwen3_8b_accuracy_vs_length.png)

![qwen3:8b 準確率 vs 位置](results/plots/qwen3_8b_accuracy_vs_position.png)

![qwen3:8b 熱力圖](results/plots/qwen3_8b_heatmap.png)

![qwen3:8b Needle 準確率](results/plots/qwen3_8b_needle_accuracy.png)

#### qwen3.5:35b 個別圖表（部分資料：僅 100K+130K）

![qwen3.5:35b 準確率 vs 長度](results/plots/qwen3.5_35b_accuracy_vs_length.png)

![qwen3.5:35b 準確率 vs 位置](results/plots/qwen3.5_35b_accuracy_vs_position.png)

![qwen3.5:35b 熱力圖](results/plots/qwen3.5_35b_heatmap.png)

![qwen3.5:35b Needle 準確率](results/plots/qwen3.5_35b_needle_accuracy.png)

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

下列模型為 thinking 模型，實驗腳本會自動關閉推理模式：

| 模型系列 | 關閉方式 |
|---------|---------|
| qwen3.*、deepseek-r1.* | Ollama API 參數 `"think": false`（原生支援）|
| gemma4.* | Ollama API 參數 `"think": false`（底層透過 prompt 注入控制 token 實作）|

目的是確保 response 只包含答案，不含推理過程，不污染評估。

> **注意**：部分模型（如 gemma4）的 thinking 控制是透過 prompt 注入實作，這類 token 會計入 context 消耗，導致跨模型之間的實際 token 使用量不完全可比。本實驗以**字元數**為控制單位，不強調跨模型的 token 數對等。

## 可復現性

- 所有 random seed 固定（`configs/wiki_articles_v2.json` 中的 `random_seed`）
- 維基百科條目 ID 固定於設定檔
- 所有模型回答原樣保存於 JSONL，可用不同評估方法重新分析
- 評估修正只影響 `is_correct` 判定，不修改 `model_response`
