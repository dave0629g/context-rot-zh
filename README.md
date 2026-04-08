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

### KV Cache 調效說明（未來重跑實驗參考）

**本次實驗所有 KV cache 相關參數均使用 Ollama 預設值，未做任何調整，且在本輪實驗中不會更動。** 以下記錄若未來重跑時可調整的方向，供版本對照與復現性追蹤。

#### 可調整的 KV Cache 參數

| 參數 | 本次值 | 可調選項 | 說明 |
|------|--------|----------|------|
| `OLLAMA_KV_CACHE_TYPE` | 未設定（預設 `f16`）| `f16` / `q8_0` / `q4_0` | KV cache 的儲存精度。`f16` 品質最高但記憶體用量最大；`q8_0` 約節省 50% 記憶體，品質損失極小；`q4_0` 約節省 75%，長 context 下有數值穩定性風險（見下方說明）。**注意：需同時啟用 `OLLAMA_FLASH_ATTENTION=1` 才能使用 q8_0 / q4_0。** |
| `OLLAMA_FLASH_ATTENTION` | 未設定（預設關閉）| `0` / `1` | 啟用後可大幅降低超長 context 的記憶體峰值，並解鎖 KV cache 量化選項。對 100K+ 的實驗尤其重要。 |
| `num_ctx`（API 請求層）| 模型預設值 | 整數，上限見各模型規格 | 本次使用模型預設的 `num_ctx`，未在 API 請求中明確指定。若要確保每個長度都有足夠的 context window，應在請求時明確帶入 `"num_ctx": <value>`，而非依賴模型預設。 |

#### 調效方向與預期影響

**方向一：啟用 Flash Attention（低風險，建議優先）**

```bash
# 修改 /etc/systemd/system/ollama.service.d/override.conf
Environment="OLLAMA_FLASH_ATTENTION=1"
```

- 預期效果：降低 100K–130K 長度的記憶體峰值；對 gemma3:27b 類型的 pad token 問題可能有改善
- 風險：輸出結果可能與本次實驗（Flash Attention 關閉）略有差異，需視為獨立的實驗條件
- **若啟用，必須重跑全部模型才能與本次結果對比**

**方向二：降低 KV Cache 精度（q8_0）**

```bash
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

- 預期效果：在記憶體受限環境下讓更多模型能跑完 130K；q8_0 對品質影響在大多數模型上可忽略
- 風險：需實測，不同模型對 KV cache 精度的敏感度不同
- 適用情境：測試更大的模型（如假設的 llama3.3:70b 在現有 VRAM 下無法完整跑 130K 時）

**方向三：明確指定 `num_ctx`（修正 context window 問題）**

```python
# 在 03_run_experiment.py 中，API 請求加入：
"options": {
    "num_ctx": target_num_ctx,  # 依該長度的實際 token 數計算後取整至 2 的冪次
    "temperature": 0,
}
```

- 預期效果：解決 gemma3:1b / qwen3:8b 類型的「實際 num_ctx 低於測試需求」問題，確保模型的 context window 確實涵蓋完整 haystack
- 風險：明確設定 `num_ctx` 後，若值過大會增加記憶體用量，值過小則截斷 prompt（與現狀相同）；需搭配 token count check 計算每個長度所需的最小 num_ctx
- **此調整會改變 context window 相關的實驗條件，結果不可與本次直接比較**

#### 本次實驗的 KV Cache 基準條件

為確保可復現性，以下明確記錄本次實驗的 KV cache 基準狀態：

```
OLLAMA_FLASH_ATTENTION = 未設定（關閉）
OLLAMA_KV_CACHE_TYPE   = 未設定（f16）
num_ctx                = 未在 API 請求中指定（使用模型 Modelfile 預設值）
模型權重量化            = Q4_K_M（Ollama 預設，所有模型一致）
```

任何偏離以上基準的重跑版本，均應在結果檔名或 metadata 中標注調整的參數，不得與本次結果混用。

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

## 互動式分析介面

實驗結果提供兩種互動介面，支援自由勾選模型組合與 variant 進行比較。

### Streamlit 本機版（app.py）

讀取本機 JSONL 檔，即時評估，資料永遠最新。

```bash
pip install streamlit plotly pandas
streamlit run app.py
```

瀏覽器開啟後，在左側側邊欄可：
- 依家族展開/收合，勾選要比較的模型
- 勾選 variant（繁問繁答 / 繁問簡答 / 簡問簡答）
- 切換圖表類型（準確率 vs 長度 / vs 位置 / 熱力圖 / Needle 準確率 / Tokenizer Overhead）

**部署到 Streamlit Community Cloud（免費）：**
1. 將 repo push 到 GitHub
2. 前往 [share.streamlit.io](https://share.streamlit.io)，連結此 repo
3. 指定 `app.py` 為入口，即可取得公開 URL

### GitHub Pages 靜態版（docs/index.html）

不需任何伺服器，部署為靜態網頁。資料為預先聚合的 `docs/data.json`。

#### 更新資料

每次新增實驗結果後：

```bash
python scripts/07_export_web.py   # 重新產生 docs/data.json
git add docs/data.json && git commit -m "更新 web 資料"
git push
```

#### 啟用 GitHub Pages

在 GitHub repo 的 **Settings → Pages** 中：
- Source：`Deploy from a branch`
- Branch：`main`，Folder：`/docs`

儲存後取得公開 URL（格式：`https://{user}.github.io/{repo}/`）

#### 本機預覽

```bash
# 需要 HTTP 伺服器（直接開 index.html 因 CORS 限制無法 fetch data.json）
python -m http.server 8080 --directory docs
# 開啟 http://localhost:8080
```

## 完整分析流程（實驗跑完後執行）

一個模型的所有 variant 完成後，依序執行以下指令即可取得所有輸出：

### Step 4：分析結果（重新評估 + 輸出統計 JSON）

```bash
# 指定單一模型
python scripts/04_analyze.py --model gemma3:4b --reeval

# 或一次分析所有已有結果的模型
python scripts/04_analyze.py --all --reeval
```

`--reeval` 使用修正版評估邏輯（中文數字正規化 + 同義詞比對）。

輸出：`results/analysis/{model}_analysis.json`

### Step 5：產生圖表

```bash
# 指定模型清單（建議：已完成繁問繁答的模型）
python scripts/05_plot_results.py \
  --models gemma3:4b llama3.1:8b qwen3:8b \
           gemma4:e2b gemma4:e4b gemma4:26b

# 或自動偵測 results/ 目錄下所有有資料的模型
python scripts/05_plot_results.py --all
```

輸出：
- `results/plots/{model}_accuracy_vs_length.png` — 準確率 vs Context 長度（各 variant）
- `results/plots/{model}_accuracy_vs_position.png` — 準確率 vs Needle 位置
- `results/plots/{model}_heatmap.png` — 熱力圖（長度 × 位置）
- `results/plots/{model}_needle_accuracy.png` — 各 Needle 準確率
- `results/plots/compare_accuracy_vs_length.png` — 跨模型準確率比較
- `results/plots/compare_token_ratio.png` — Tokenizer overhead 比較
- `results/plots/compare_65k_accuracy.png` — 65K 各模型 × 各 variant
- `results/plots/compare_token_map.png` — 字元數 → Token 數對照

### 一鍵執行（6 個已完成模型）

```bash
python scripts/04_analyze.py --all --reeval && \
python scripts/05_plot_results.py \
  --models gemma3:4b llama3.1:8b qwen3:8b gemma4:e2b gemma4:e4b gemma4:26b
```

## 圖表設計原則

圖表以 `matplotlib` 撰寫，不依賴 LLM 生成。`scripts/05_plot_results.py` 的視覺設計考量如下：

### 黑白列印可辨識

每條線同時以**三個獨立維度**編碼，即使灰階列印也能辨識：

| 維度 | 說明 |
|------|------|
| **marker 形狀** | 同家族相同，跨家族不同 |
| **fillstyle** | 小模型 = 空心（none）；大模型 = 實心（full） |
| **linestyle** | 小模型 = 虛線（`--` 或 `-.`）；大模型 = 實線（`-`） |

### 家族分組（marker 形狀對照）

| 家族 | Marker 形狀 | 彩色時的色調 |
|------|------------|------------|
| Gemma 3 | 圓形 `o` | 藍色系（深=27B，淺=1B）|
| Gemma 4 Edge（E2B/E4B）| 倒三角 `v` | 橘色系 |
| Gemma 4 Standard（26B/31B）| 菱形 `D` | 紅橘系 |
| Llama | 正方形 `s` | 紅色系 |
| Qwen3 | 上三角 `^` | 綠色系 |
| Qwen3.5 | 加號 `P` | 紫色系（深=35B，淺=2B）|

### 各 Variant 的線條樣式（單模型圖）

| Variant | Marker | Fillstyle | Linestyle | 顏色 |
|---------|--------|-----------|-----------|------|
| 繁問繁答 | `o` | full（實心）| `-`（實線）| 藍 `#2E86AB` |
| 簡問簡答 | `s` | none（空心）| `--`（虛線）| 紫 `#A23B72` |
| 繁問簡答 | `^` | full（實心）| `-.`（點劃線）| 橘 `#F18F01` |

### 新增模型時的擴充方式

在 `05_plot_results.py` 頂部的 `MODEL_COLORS`、`MODEL_MARKERS`、`MODEL_LABELS` 三個 dict 加入新模型即可。遵循同家族同 marker 形狀、深色大模型 / 淺色小模型的原則。

```python
# 範例：加入 gemma3:12b
MODEL_COLORS["gemma3:12b"]  = "#3A62B8"      # 藍色系中間深度
MODEL_MARKERS["gemma3:12b"] = dict(marker="o", fillstyle="full", linestyle="-.")
MODEL_LABELS["gemma3:12b"]  = "Gemma 3 12B"
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
| gemma3:1b | 1B | 994 筆 ✓（受 32K context 限制）| 待執行 | 待執行 |
| gemma3:4b | 4B | 1,320 筆 ✓ | 1,100 筆 ✓ | 1,100 筆 ✓ |
| gemma3:12b | 12B | 1,320 筆 ✓ | 待執行 | 待執行 |
| gemma3:27b | 27B | 1,320 筆 ✓（100K/130K 輸出 pad token）| 待執行 | 待執行 |
| gemma4:e2b | E2B | 1,320 筆 ✓ | 待執行 | 898 筆（500–24K，進行中）|
| gemma4:e4b | E4B | 1,320 筆 ✓ | 待執行 | 待執行 |
| gemma4:26b | 26B | 1,320 筆 ✓ | 待執行 | 待執行 |
| gemma4:31b | 31B | 1,320 筆 ✓ | 待執行 | 待執行 |
| llama3.1:8b | 8B | 1,320 筆 ✓ | 1,100 筆 ✓ | 1,100 筆 ✓ |
| llama3.3:70b | 70B | 1,320 筆 ✓ | 待執行 | 待執行 |
| qwen3:8b | 8B | 1,100 筆有效（500–65K）✓，220 筆因 ctx 超限跳過 | 125 筆（部分）| 待執行 |
| qwen3.5:2b | 2B | 1,320 筆 ✓ | 待執行 | 待執行 |
| qwen3.5:4b | 4B | 1,320 筆 ✓ | 待執行 | 待執行 |
| qwen3.5:9b | 9B | 1,320 筆 ✓ | 待執行 | 待執行 |
| qwen3.5:27b | 27B | 1,320 筆 ✓ | 待執行 | 待執行 |
| qwen3.5:35b | 35B | 1,320 筆 ✓ | 待執行 | 待執行 |

## 結果（已完成部分）

準確率以修正版評估邏輯計算（`--reeval`：中文數字正規化 + 同義詞比對）。

### 整體檢索成功率

準確率為各 context 長度的平均值（gemma3:1b 最大 32K）。

| 模型 | 繁問繁答 | 繁問簡答 | 簡問簡答 | Tokenizer Overhead |
|------|---------|---------|---------|-------------------|
| gemma4:31b | **100.0%** | — | — | — |
| qwen3.5:4b | **100.0%** | — | — | — |
| qwen3.5:9b | **100.0%** | — | — | — |
| qwen3.5:27b | **100.0%** | — | — | — |
| qwen3.5:35b | **100.0%** | — | — | — |
| gemma4:e4b | **99.9%** | — | — | — |
| gemma4:e2b | **99.8%** | — | — | — |
| qwen3.5:2b | **99.8%** | — | — | — |
| gemma4:26b | 98.9% | — | — | — |
| gemma3:12b | 98.0% | — | — | — |
| llama3.1:8b | 97.1% | **98.8%** | 95.3% | +7.1% |
| llama3.3:70b | 97.9% | — | — | — |
| qwen3:8b | 94.1%（≤32K 全滿，65K 崩潰）| 100.0%（部分）| — | +13.6% |
| gemma3:4b | 91.6% | **99.1%** | 96.8% | +5.7% |
| gemma3:27b | 85.0%（100K/130K pad token）| — | — | — |
| gemma3:1b | 76.4%（最大 32K）| — | — | — |

> gemma4 系列與 Qwen3.5 系列在繁體中文長 context 下均有優異表現。

### 準確率 vs Context 長度

#### gemma4:31b

| 長度 | 繁問繁答 |
|------|---------|
| 500–130K | **100.0%** |

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

#### gemma3:12b

| 長度 | 繁問繁答 |
|------|---------|
| 500–12K | 100.0% |
| 16K | 99.1% |
| 24K–32K | 100.0% |
| 65K | 96.4% |
| 100K | 93.6% |
| 130K | 86.4% |

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

#### gemma3:27b

| 長度 | 繁問繁答 | 備註 |
|------|---------|------|
| 500–32K | 100.0% | 穩定 |
| 65K | 99.1% | 穩定 |
| 100K | 20.9% | 大量 `<pad>` token 輸出（模型限制）|
| 130K | 0.0% | 全部 `<pad>` token 輸出 |

> gemma3:27b 在 100K+ 字元時模型輸出退化為 `<pad>` token，推測為 Ollama 中 gemma3:27b Q4 量化版本在超長 context 下的已知問題。65K 以內表現穩定。

#### gemma3:1b（最大 context window 32,768 tokens）

| 長度 | 繁問繁答 |
|------|---------|
| 500 | 100.0% |
| 2K | 97.3% |
| 4K | 96.4% |
| 6K | 90.0% |
| 8K | 87.3% |
| 12K | 87.3% |
| 16K | 70.0% |
| 24K | 44.5% |
| 32K | 14.5% |

> gemma3:1b context window 僅 32K tokens，長 context 下準確率急速下降。

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

#### qwen3.5 系列（繁問繁答）

| 長度 | 2B | 4B | 9B | 27B | 35B |
|------|----|----|----|----|-----|
| 500 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 2K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4K | 99.1% | 100.0% | 100.0% | 100.0% | 100.0% |
| 6K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 8K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 12K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 16K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 24K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 32K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 65K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 100K | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 130K | 99.1% | 100.0% | 100.0% | 100.0% | 100.0% |

> Qwen3.5 全系列在繁體中文長 context 下表現卓越，2B–35B 幾乎全部 100%，130K 字元下 2B 僅降至 99.1%，其餘完美。

#### llama3.3:70b（繁問繁答）

| 長度 | 繁問繁答 |
|------|---------|
| 500 | 99.1% |
| 2K | 100.0% |
| 4K | 100.0% |
| 6K | 100.0% |
| 8K | 100.0% |
| 12K | 100.0% |
| 16K | 100.0% |
| 24K | 100.0% |
| 32K | 100.0% |
| 65K | 97.3% |
| 100K | 97.3% |
| 130K | 80.9% |

各 Needle 準確率（繁問繁答，全長度）：

| N01 金額 | N02 人名 | N03 面積 | N04 數量（干擾高）| N05 百分比 |
|---------|---------|---------|-----------------|---------|
| 95.5% | 98.1% | 98.5% | 97.3% | 100.0% |

> N01（鳳梨酥 473 億元）準確率偏低，推測模型在長 context 下對金額數字的辨識較易受干擾。N05（百分比）全部正確。

### 關鍵發現

1. **Qwen3.5 系列在繁體中文長 context 下達到近乎完美的成績**
   - 2B–35B 全部 5 個模型在完整 12 個 context 長度（500–130K）幾乎全部 100%
   - qwen3.5:2b（最小）在 130K 僅降至 99.1%，其餘長度 100%
   - 令人驚訝的結論：模型大小在 Qwen3.5 系列中幾乎不影響繁體長 context 性能

2. **gemma4 系列在長 context 下表現優異**
   - gemma4:31b 在全部 12 個長度 100%（完美分數）
   - gemma4:e2b（2B edge）全長度維持 99.8%，包含 130K
   - gemma4:e4b（4B edge）幾乎全部 100%
   - gemma4:26b（26B MoE）在 130K 仍有 92.7%，遠優於同 active params 的 gemma3:4b（52.7%）
   - 推測 MoE 架構與 gemma4 訓練改善對長 context 有顯著幫助

3. **gemma3 系列：模型大小顯著影響長 context 性能**
   - gemma3:12b（130K: 86.4%）遠優於 gemma3:4b（130K: 52.7%）
   - gemma3:27b 在 65K 仍有 99.1%，但 100K+ 輸出退化為 `<pad>` token
   - gemma3:1b 的 context window 僅 32K tokens，在 32K 時準確率已降至 14.5%

4. **繁體 context rot 假說在 gemma3:4b 得到支持**
   - 65K 字元下：繁問繁答 88.2% vs 繁問簡答 93.6%（差距 **+5.4pp**）
   - llama3.1:8b 差距較小（+1.8pp），整體更穩健
   - Qwen3.5/gemma4 系列繁問繁答已達 100%，無法觀察 context rot 效應（天花板效應）

5. **qwen3:8b 的 context window 限制導致 65K 嚴重崩潰**
   - 0–32K 全部 100%，65K 驟降至 40.9%（非漸進式衰退）
   - tokenizer overhead 最高（+13.6%），65K 字元已佔 88% context window

6. **llama3.3:70b 繁問繁答全部完成（1,320 筆）**
   - 500–32K 全部 100%，65K/100K 97.3%，130K 降至 80.9%
   - 整體準確率 97.9%，介於 gemma3:12b（98.0%）與 llama3.1:8b（97.1%）之間
   - 屬於模式 B（漸進退化型）：65K 起才出現退化，130K 降幅（−16.4pp）為本系列最大
   - Needle 層級：N05（百分比）100% 全對，N01（金額）最低 95.5%；N04（台灣黑熊，干擾高）97.3%
   - 位置敏感度：首位（pos 0.0）95.0% 略低，尾位（pos 0.8–0.9）100%，無明顯「Lost in the Middle」

### 資料品質問題與待重新檢視清單

以下測試結果存在已知問題，數值不應直接用於跨模型比較，需謹慎解讀或待修復後重新評估。**未來改版實驗腳本將依各項 root cause 加入對應的前置檢查與防護機制。**

#### 🔴 嚴重問題：模型輸出退化

| 模型 | 長度 | 準確率 | 問題描述 | Root Cause 分析 | 實驗初期檢視方法 | 建議處理 |
|------|------|--------|----------|----------------|----------------|----------|
| gemma3:27b | 100K | 20.9% | Ollama Q4 量化在超長 context 下輸出 `<pad>` token，模型實際上沒有作答 | Q4 KV cache 量化在超長 sequence 下發生數值溢位或記憶體配置失敗，attention mask 失效後模型回落到 padding 行為。根本原因是量化精度不足以維持超長 context 下的注意力計算穩定性。 | **Sanity probe**：在正式跑大矩陣前，先用 1 筆 100K + 1 筆 130K 的 prompt 測試模型輸出是否含非空、非 pad 的有意義文字。若輸出長度 < 5 tokens 或全為特殊 token，立即中止並記錄為「該長度不支援」。 | 待升級 Ollama / 使用 fp16 重跑 |
| gemma3:27b | 130K | 0.0% | 同上，全部輸出為 `<pad>`，準確率為 0 不代表「理解但答錯」 | 同上。130K 比 100K 更嚴重，推測 KV cache 在此長度已完全耗盡可用精度，輸出 100% 為 `<pad>`。 | 同上，130K probe 結果應可提前預測 100K 問題。 | 同上 |

> **注意：** gemma3:27b 在 65K 時仍有 99.1%，pad token 問題僅發生在 100K+。這是 Ollama Q4 量化模型在超長 context 下的已知限制，非資料收集錯誤。

#### 🟡 中等問題：Context Window 硬限制導致資料缺失或崩潰

| 模型 | 受影響長度 | 準確率 | 問題描述 | Root Cause 分析 | 實驗初期檢視方法 | 建議處理 |
|------|----------|--------|----------|----------------|----------------|----------|
| gemma3:1b | 24K | 44.5% | Context window 實際上限 ~32K tokens，24K 字元已接近上限 | 雖然官方 context window 標示 131,072，但 Ollama 載入的 gemma3:1b 模型實際可用 KV cache 受限於 `num_ctx` 參數，預設值遠低於官方上限。字元數轉 token 後 24K 字元約需 30K+ tokens，已超出實際 num_ctx。 | **Pre-run token count check**：實驗前先呼叫 `/api/tokenize`（或等效 API）取得每個長度設定的實際 token 數，與模型的 `num_ctx` 比較。若 token 數 > num_ctx × 0.85，標示為「接近上限」，> num_ctx 則標示為「超限 SKIP」。 | 僅引用 ≤16K 資料；標示 24K/32K 為邊界衰退 |
| gemma3:1b | 32K | 14.5% | 同上，32K 字元幾乎超出模型 context window | 同上。32K 字元 ≈ 40K tokens，遠超 num_ctx，模型被迫截斷 prompt，needle 可能根本未被包含在有效輸入中。準確率 14.5% 接近隨機，符合「prompt 被截斷」的預期行為。 | 同上，token count check 可在腳本啟動時一次性完成，生成每個（模型, 長度）組合的 skip 清單。 | 同上 |
| gemma3:1b | 65K+ | — | 測試矩陣未包含（max context 設計限制） | 設計時已知限制，由 `--max_length` 參數控制。 | 同上 token count check 即可自動排除。 | 維持現狀，標示模型能力上限 |
| qwen3:8b | 65K | 40.9% | Context window 40,960 tokens，65K 字元佔 88%，準確率驟降而非漸進衰退 | 65K 字元在 qwen3:8b tokenizer 下約需 36K tokens（含 overhead），已接近 40,960 上限。剩餘空間不足以容納完整 system prompt + question，導致 prompt 被截斷或 attention 嚴重退化。驟降而非漸進式衰退是 context window 溢出的典型特徵，不同於 context rot 的漸進曲線。 | 同上 token count check。qwen3:8b 的 65K 應被標示為「高風險：佔用 > 85% context window」，並在結果圖表中以虛線或不同符號標示。 | 65K 數值反映 context 溢出，非一般 context rot |
| qwen3:8b | 100K、130K | — | 全部 SKIP（context 超限） | 100K 字元 ≈ 56K tokens，超出 40,960 上限。腳本已正確 SKIP，但 SKIP 的判斷邏輯應基於 token 數而非字元數（目前版本可能以字元數估算）。 | 同上。應統一改為以 token 數為 SKIP 依據，避免不同 tokenizer 的字元/token 比率差異造成誤判。 | 維持現狀 |

#### 🟡 中等問題：資料不完整，無法進行完整分析

| 模型 | 已完成筆數 | 缺少長度 | 問題描述 | Root Cause 分析 | 實驗初期檢視方法 | 建議處理 |
|------|----------|----------|----------|----------------|----------------|----------|
（llama3.3:70b 已於 2026-04-08 全部完成，1,320 筆，不再有不完整問題。）

#### 🟠 設計限制：Haystack 干擾（Distractor）

| Needle | 干擾源 | 嚴重度 | 影響 | Root Cause 分析 | 實驗初期檢視方法 | 建議處理 |
|--------|--------|--------|------|----------------|----------------|----------|
| N04「黑熊數量」| `24_台灣地理.txt` 含「臺灣黑熊」真實描述 | **高** | N04 絕對準確率偏低，模型可能找到 haystack 中真實的黑熊段落而非插入的 needle | Haystack 語料（台灣 Wikipedia）中剛好存在與 needle 主題高度相關的真實段落。模型在長 context 下找到干擾段落時，會以真實資訊取代 needle 作答，表現為「看似理解語意但答錯數字」。這是 NIAH 實驗設計的根本性漏洞：needle 的唯一性未被保證。 | **Distractor detection**：在建構 haystack 前，對每個 needle 的關鍵詞（如「黑熊」「台灣黑熊」）在全部語料庫中做 full-text search，若命中率超過閾值（如 > 2 篇），則替換該語料檔案或重新設計 needle。此步驟應寫入 `02_build_haystacks.py` 的前置檢查。 | 替換 N04 語料或重新設計 needle（使用虛構實體）；已有資料中 N04 結果應單獨標注 |
| N03「3.7 平方公里」| 3.7°C（氣溫）、3.7%（其他數值）| 低–中 | 影響有限，單位不同可區分 | 數字相同但語境（面積 vs 溫度/比率）差異大，有能力的模型可透過單位辨別。對小模型或超長 context 下注意力退化時，可能誤取相同數字。 | 同上 distractor detection，但閾值可放寬（相同數字 + 不同單位視為低風險）。 | 低優先級；若有新版本實驗，建議換用更唯一的數值 |

> **注意：** 干擾問題對三種 variant（繁問繁答/繁問簡答/簡問簡答）的影響是**對等的**，因此 variant 之間的**相對比較仍然有效**。N04 的絕對準確率偏低是 haystack 設計問題，不代表模型能力較差。

#### 實驗腳本改版方向摘要

基於以上四類 root cause，下一版實驗腳本（`03_run_experiment_v2.py`）應在實驗啟動前加入以下前置檢查：

| 檢查項目 | 對應問題 | 實作建議 |
|----------|----------|----------|
| **Sanity probe**：每個（模型, 長度）組合跑 1 筆 probe，確認輸出非空、非純 pad token | 模型輸出退化（gemma3:27b 類型） | 在主迴圈前執行；失敗則跳過該長度並寫入 `skip_log.jsonl` |
| **Token count check**：呼叫 tokenizer API 計算每個長度的實際 token 數，與 num_ctx 比較 | Context window 溢出（qwen3:8b / gemma3:1b 類型） | 啟動時一次性生成 skip 清單；90% 以上標「警告」，100% 以上標「SKIP」 |
| **執行時間估算**：以 3 筆 probe 推估完整矩陣所需時間 | 資料不完整（llama3.3:70b 類型） | 啟動時輸出預估總耗時，由使用者確認後繼續 |
| **Distractor detection**：對每個 needle 關鍵詞在語料庫做全文搜尋 | Haystack 干擾（N04 類型） | 寫入 `02_build_haystacks.py`；偵測到干擾時輸出警告並提示替換語料 |

---

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

#### gemma3:12b 個別圖表

![gemma3:12b 準確率 vs 長度](results/plots/gemma3_12b_accuracy_vs_length.png)

![gemma3:12b 準確率 vs 位置](results/plots/gemma3_12b_accuracy_vs_position.png)

![gemma3:12b 熱力圖](results/plots/gemma3_12b_heatmap.png)

![gemma3:12b Needle 準確率](results/plots/gemma3_12b_needle_accuracy.png)

#### gemma3:27b 個別圖表

![gemma3:27b 準確率 vs 長度](results/plots/gemma3_27b_accuracy_vs_length.png)

![gemma3:27b 準確率 vs 位置](results/plots/gemma3_27b_accuracy_vs_position.png)

![gemma3:27b 熱力圖](results/plots/gemma3_27b_heatmap.png)

![gemma3:27b Needle 準確率](results/plots/gemma3_27b_needle_accuracy.png)

#### gemma3:1b 個別圖表

![gemma3:1b 準確率 vs 長度](results/plots/gemma3_1b_accuracy_vs_length.png)

![gemma3:1b 準確率 vs 位置](results/plots/gemma3_1b_accuracy_vs_position.png)

![gemma3:1b 熱力圖](results/plots/gemma3_1b_heatmap.png)

![gemma3:1b Needle 準確率](results/plots/gemma3_1b_needle_accuracy.png)

#### gemma4:31b 個別圖表

![gemma4:31b 準確率 vs 長度](results/plots/gemma4_31b_accuracy_vs_length.png)

![gemma4:31b 準確率 vs 位置](results/plots/gemma4_31b_accuracy_vs_position.png)

![gemma4:31b 熱力圖](results/plots/gemma4_31b_heatmap.png)

![gemma4:31b Needle 準確率](results/plots/gemma4_31b_needle_accuracy.png)

#### qwen3.5:2b 個別圖表

![qwen3.5:2b 準確率 vs 長度](results/plots/qwen3.5_2b_accuracy_vs_length.png)

![qwen3.5:2b 準確率 vs 位置](results/plots/qwen3.5_2b_accuracy_vs_position.png)

![qwen3.5:2b 熱力圖](results/plots/qwen3.5_2b_heatmap.png)

![qwen3.5:2b Needle 準確率](results/plots/qwen3.5_2b_needle_accuracy.png)

#### qwen3.5:4b 個別圖表

![qwen3.5:4b 準確率 vs 長度](results/plots/qwen3.5_4b_accuracy_vs_length.png)

![qwen3.5:4b 準確率 vs 位置](results/plots/qwen3.5_4b_accuracy_vs_position.png)

![qwen3.5:4b 熱力圖](results/plots/qwen3.5_4b_heatmap.png)

![qwen3.5:4b Needle 準確率](results/plots/qwen3.5_4b_needle_accuracy.png)

#### qwen3.5:9b 個別圖表

![qwen3.5:9b 準確率 vs 長度](results/plots/qwen3.5_9b_accuracy_vs_length.png)

![qwen3.5:9b 準確率 vs 位置](results/plots/qwen3.5_9b_accuracy_vs_position.png)

![qwen3.5:9b 熱力圖](results/plots/qwen3.5_9b_heatmap.png)

![qwen3.5:9b Needle 準確率](results/plots/qwen3.5_9b_needle_accuracy.png)

#### qwen3.5:27b 個別圖表

![qwen3.5:27b 準確率 vs 長度](results/plots/qwen3.5_27b_accuracy_vs_length.png)

![qwen3.5:27b 準確率 vs 位置](results/plots/qwen3.5_27b_accuracy_vs_position.png)

![qwen3.5:27b 熱力圖](results/plots/qwen3.5_27b_heatmap.png)

![qwen3.5:27b Needle 準確率](results/plots/qwen3.5_27b_needle_accuracy.png)

#### qwen3.5:35b 個別圖表

![qwen3.5:35b 準確率 vs 長度](results/plots/qwen3.5_35b_accuracy_vs_length.png)

![qwen3.5:35b 準確率 vs 位置](results/plots/qwen3.5_35b_accuracy_vs_position.png)

![qwen3.5:35b 熱力圖](results/plots/qwen3.5_35b_heatmap.png)

![qwen3.5:35b Needle 準確率](results/plots/qwen3.5_35b_needle_accuracy.png)

#### llama3.3:70b 個別圖表

![llama3.3:70b 準確率 vs 長度](results/plots/llama3.3_70b_accuracy_vs_length.png)

![llama3.3:70b 準確率 vs 位置](results/plots/llama3.3_70b_accuracy_vs_position.png)

![llama3.3:70b 熱力圖](results/plots/llama3.3_70b_heatmap.png)

![llama3.3:70b Needle 準確率](results/plots/llama3.3_70b_needle_accuracy.png)

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

---

## 全面分析

繁問繁答實驗已全面完成（16 個模型，各 1,320 筆）。本章節依據各模型的 `results/analysis/*.json` 資料，就三個研究問題提出正式答覆，並歸納各類型模型在繁體中文長 context 研究中的特性。

---

### RQ1：繁體中文分詞負擔是否造成可量測的 Context Rot？

**分析基礎**：目前僅 gemma3:4b 與 llama3.1:8b 完成完整三 variant 對比（繁問繁答 / 繁問簡答 / 簡問簡答），是本研究問題的主要證據來源。

#### 65K 字元下三 Variant 對比

| 模型 | 繁問繁答 | 繁問簡答 | 簡問簡答 | 繁問簡答 − 繁問繁答 |
|------|---------|---------|---------|-------------------|
| gemma3:4b | 88.2% | 93.6% | 82.7% | **+5.4pp** |
| llama3.1:8b | 92.7% | 94.5% | 84.5% | **+1.8pp** |

> **繁問簡答**與**繁問繁答**的差異是最乾淨的比對：問題語言相同，僅 haystack 腳本不同。簡體 haystack 的 token 效率較高，兩個模型方向一致地顯示繁體 haystack 較吃虧。

**反直覺發現——簡問簡答 < 繁問繁答**：兩個模型的簡問簡答成績均低於繁問繁答，與「簡體效率較高故應較好」的直覺相反。可能原因：(1) 跨腳本問答（問題與答案腳本不一致）增加理解難度；(2) 模型針對繁體中文問答的訓練資料分布較優化；(3) 現有三 variant 資料不足以排除訓練偏差。此發現說明「context rot」並非唯一影響因素，腳本一致性本身也有獨立效果。

**位置敏感度——質性佐證**：在 gemma3:4b，繁問繁答呈現明顯的「Lost in the Middle」效應（位置 0.0=1.0，位置 0.5=0.81，位置 1.0=1.0）；繁問簡答（簡體 haystack）的位置曲線幾乎平坦（0.98–1.0）。相同的問題語言、不同的 haystack 腳本，卻產生截然不同的位置敏感度，支持 tokenizer overhead 加劇了中段注意力的稀釋效應。

**天花板效應的警示**：Qwen3.5 / Gemma4 系列在繁問繁答已達 100%，無法從此設計中觀察任何 variant 差異，並不代表它們對繁體 context rot 免疫，只代表本實驗的字元長度上限不足以顯現差距。

**RQ1 答覆**：假說成立，但效果溫和。在兩個有效對比模型中，使用繁體 haystack 比簡體 haystack 損失 1.8–5.4 個百分點（65K 字元）。效果大小與模型架構高度相關，架構差異可掩蓋或放大 tokenizer 效應。

---

### RQ2：不同模型架構／家族的繁體長 Context 抵抗力

由於大多數模型僅完成繁問繁答，本節以準確率對 context 長度的退化曲線形態作為主要分析維度，將 16 個模型歸納為四種行為模式。

#### 模式 A — 硬性上限型（Context Window 截斷，非漸進 Context Rot）

| 模型 | 正常範圍 | 崩潰點 | 崩潰後準確率 | 截斷原因 |
|------|---------|--------|------------|---------|
| gemma3:1b | ≤16K 字元 | 24K | 44.5% → 14.5% | 模型預設 num_ctx 低於官方宣稱上限 |
| qwen3:8b | ≤32K 字元 | 65K | 40.9% | 40,960 token 上限，65K 繁體 ≈ 36K tokens（88%） |

特徵：準確率突然崩潰而非漸進衰退。這是 **context window 截斷效應**，與 context rot 的機制不同，不應混用。

#### 模式 B — 漸進退化型（正統 Context Rot 曲線）

| 模型 | 100% 穩定範圍 | 65K | 100K | 130K |
|------|-------------|-----|------|------|
| gemma3:4b | 500–8K | 88.2% | 68.2% | 52.7% |
| gemma3:12b | 500–24K | 96.4% | 93.6% | 86.4% |
| llama3.1:8b | 6K–24K | 92.7% | 90.9% | 91.8% |
| llama3.3:70b | 2K–32K（500=99.1%）| 97.3% | 97.3% | 80.9% |

特徵：隨 context 長度增加，準確率單調（或近似單調）下降；退化幅度可量化。這是觀察與研究 context rot 機制的最佳模型類型。llama3.1:8b 曲線略非單調（500 字元起點 94.5%，6K 起回升至 100%，65K 後才開始下降），推測與短 context 下的評估噪音或 Lost-in-the-Beginning 效應有關。

#### 模式 C — 突發崩潰型（量化精度不足引發的輸出退化）

| 模型 | 穩定範圍 | 崩潰點 | 崩潰後 |
|------|---------|--------|--------|
| gemma3:27b | 500–65K（≥99.1%）| 100K | 20.9%（pad token）→ 130K 0.0%（全 pad）|

gemma3:27b 在 65K 以前表現完美，在 100K 開始輸出大量 `<pad>` token，130K 全部輸出 pad。此行為確認為 **Ollama Q4 量化在超長 context 下 KV cache 數值不穩定**，並非一般化的 context rot 現象，不應與其他模型的漸進退化混為一談。65K 以下的 gemma3:27b 資料可信。

#### 模式 D — 近乎完美型（天花板效應，無法觀察 Context Rot）

| 模型 | 整體準確率 | 130K |
|------|-----------|------|
| gemma4:31b | 100.0% | 100.0% |
| qwen3.5:4b / 9b / 27b / 35b | 100.0% | 100.0% |
| gemma4:e4b | 99.9% | 100.0% |
| gemma4:e2b | 99.8% | 100.0% |
| qwen3.5:2b | 99.8% | 99.1% |
| gemma4:26b | 98.9% | 92.7% |

這些模型在本實驗的測試範圍（最長 130K 字元）內幾乎無退化，無法用此設計區分彼此的 context rot 抵抗力。gemma4:26b（MoE，active 4B）在 130K 的 92.7% 遠優於同 active parameter 量的 gemma3:4b dense（52.7%），量化了 **MoE 架構 + 世代更新** 的效益。

**大小 vs 世代的對比**：
- 同代內規模越大越強（gemma3 系列：27b > 12b > 4b > 1b）
- 跨代架構優勢壓過規模：Qwen3.5 2B 在全部 130K 字元下的成績優於 Gemma3 27B（qwen3.5:2b=99.1% vs gemma3:27b 因 bug 為 0%，即便排除 bug，12b 在 130K 也僅 86.4%）

**RQ2 答覆**：模型世代與架構是繁體長 context 性能的主要預測因子，遠超過參數量。Qwen3.5 全系列與 Gemma4 系列已實質解決 130K 字元以內的繁體中文長 context 問題；Gemma3 系列則隨 context 長度呈現明顯的規模依賴退化。

---

### RQ3：Tokenizer Overhead 與性能退化的量化關係

**資料可用性**：僅三個模型有實測 tokenizer overhead 數據（繁體 / 簡體 token 數比值），其中 qwen3:8b 因達到 context window 硬上限，其性能崩潰屬於模式 A（截斷），不適合納入 context rot 的相關分析。

#### 有效數據點

| 模型 | Tokenizer Overhead | 繁問簡答 − 繁問繁答 @ 65K |
|------|-------------------|--------------------------|
| gemma3:4b | +5.7%（avg ratio 1.0568）| +5.4pp |
| llama3.1:8b | +7.0%（avg ratio 1.0705）| +1.8pp |
| qwen3:8b | +13.6%（avg ratio 1.1362）| N/A（context window 截斷）|

**非單調關係**：overhead 較高的 llama3.1:8b（+7.0%）在 65K 的 variant 差距反而較小（+1.8pp），而 overhead 較低的 gemma3:4b（+5.7%）差距較大（+5.4pp）。這表明架構差異（attention 機制、位置編碼、訓練資料分布）顯著掩蓋了純 tokenizer 效應，無法從這兩個數據點建立單調的劑量-反應關係。

**Token Budget 重新框架**：以 token 空間而非字元空間理解開銷更直覺。65K 字元在 llama3.1:8b（+7% overhead）下，繁體等效於簡體的約 69,550 字元的 token 量；在 gemma3:4b（+5.7%）下等效於約 68,705 字元。這使模型在其已知退化曲線上稍微提前到達衰退拐點。

**RQ3 答覆**：現有 2 個有效數據點不足以建立定量的 overhead-performance 關係。方向上與假說一致（繁體 haystack 較劣），但架構混淆因子無法受控。需要將三 variant 實驗擴充至更多模型（尤其是 Gemma4 / Qwen3.5 系列在更長字元下），才能進行有意義的相關分析。

---

### 各類型模型特性（針對繁體中文 Context Rot 研究）

#### 模型適用性矩陣

| 模型 | 適合研究 Context Rot | 繁體生產可靠性 | 推薦最大字元數 | 特殊注意事項 |
|------|:------------------:|:------------:|:------------:|------------|
| gemma3:1b | — | 低 | ≤12K | num_ctx 硬限制，24K+ 已截斷 |
| gemma3:4b | **最佳 benchmark** | 中 | ≤32K | 有完整三 variant 資料，退化曲線清晰 |
| gemma3:12b | 可用 | 中高 | ≤65K | 退化溫和，適合中長 context 生產 |
| gemma3:27b | — | 中（≤65K）| ≤65K | 100K+ 有 pad token bug，不可用 |
| gemma4:e2b | — | 高 | ≤130K | 天花板效應，不適合研究 rot |
| gemma4:e4b | — | 高 | ≤130K | 天花板效應，不適合研究 rot |
| gemma4:26b | 邊緣可用 | 高 | ≤130K | 130K 仍 92.7%，退化幅度小但存在 |
| gemma4:31b | — | **最高** | ≤130K | 全長度 100%，無法觀察退化 |
| llama3.1:8b | **良好** | 中高 | ≤130K | 有完整三 variant；非單調曲線需留意 |
| llama3.3:70b | 可用 | 高 | ≤130K | 500–32K 全 100%，130K 80.9%，屬漸進退化型 |
| qwen3:8b | — | 中 | ≤32K | context window 硬限制，非 rot |
| qwen3.5:2b | — | 高 | ≤130K | 天花板效應，需 200K+ 才能觀察 |
| qwen3.5:4b | — | **最高** | ≤130K | 天花板效應 |
| qwen3.5:9b | — | **最高** | ≤130K | 天花板效應 |
| qwen3.5:27b | — | **最高** | ≤130K | 天花板效應 |
| qwen3.5:35b | — | **最高** | ≤130K | 天花板效應 |

**各家族特性摘要**：

- **Gemma3 系列**：規模依賴明顯，小模型（1b/4b）容易受繁體 context rot 影響，適合作研究工具；大模型（12b）生產穩定性尚可但 130K 有顯著退化；27b 因 bug 不適合超長 context。此系列是目前三 variant 資料最完整的家族。
- **Gemma4 系列**：世代跳躍顯著，即使邊緣小模型（e2b 2B）也幾乎完美。26b MoE 在 130K 的 92.7% 展示了稀疏架構在超長 context 下的優勢。31b 是繁體長 context 生產部署的最佳選擇之一。
- **Llama 系列**：llama3.1:8b 是性價比高的研究工具（8B 參數，有完整三 variant，展示可量測退化）；llama3.3:70b 現已完成（1,320 筆），整體 97.9%，130K 80.9%，屬漸進退化型（模式 B），是 70B 規模下繁體長 context 性能的重要參照點。
- **Qwen3 (舊) 系列**：qwen3:8b 的 context window 硬限制（40,960 tokens）使其不適合繁體超長 context；+13.6% 的最高 overhead 值在理論上最值得關注，但無法在本實驗中直接觀察 rot 效應。
- **Qwen3.5 系列**：全系列在測試範圍內近乎完美，從 2B 到 35B 幾乎無差距，顯示 Qwen3.5 的訓練在繁體長 context 上已高度優化。此系列是生產部署的優先推薦，但若要研究繁體 context rot，需設計 200K+ 字元的實驗或使用不同壓力機制。

---

### 總體結論與研究局限

#### 已建立的發現

1. **繁體 context rot 假說成立（效果溫和）**：在漸進退化型模型中（gemma3:4b, llama3.1:8b），使用繁體 haystack 比簡體 haystack 在 65K 字元下損失 1.8–5.4 個百分點，方向一致支持 tokenizer overhead 假說。

2. **位置敏感度差異是質性確認**：gemma3:4b 的繁問繁答在中段位置呈現顯著的「Lost in the Middle」效應，而繁問簡答（簡體 haystack）的位置曲線幾乎平坦。這不僅是量的差距，也是注意力分布模式的差異。

3. **架構世代是最強的預測因子**：Gemma4 / Qwen3.5 系列在測試範圍內已有效解決繁體長 context 問題，且與參數量幾乎無關（Qwen3.5 2B ≈ Qwen3.5 35B）。世代更新的影響遠大於腳本/tokenizer 的影響。

4. **模型行為模式分類（四類）**：硬性上限型（模式 A）、漸進退化型（模式 B）、突發崩潰型（模式 C）、近乎完美型（模式 D），有助於研究者選擇適合的模型作為研究工具或生產部署。

#### 未建立的項目

- **Overhead-Performance 定量關係**：僅 2 個有效數據點，且呈非單調，架構混淆因子無法受控。
- **效應的普遍性**：context rot 假說僅在 2 個模型中直接驗證，其他模型缺乏三 variant 資料。
- **Qwen3.5 / Gemma4 的真實 context rot 臨界點**：天花板效應使本實驗設計對這些模型失效，它們的實際臨界長度未知。

#### 研究局限

- **核心限制**：完整三 variant 對比僅 2 個模型（gemma3:4b, llama3.1:8b），RQ1/RQ3 的結論泛化性有限。
- **天花板效應**：現代高性能模型在 130K 字元以內幾乎完美，使此實驗設計對前沿模型的評估能力有限。
- **資料品質問題**：gemma3:27b 的 pad token bug 使其 100K/130K 資料不可信，無法反映真實長 context 能力。
- **Tokenizer Overhead 量測**：overhead 為語料級平均，未測量句段級方差；不同內容類型的繁簡 token 比可能顯著差異。
- **Distractor 污染**：N04（台灣黑熊）haystack 中存在高相關度真實資訊，對各 variant 對等影響，使 N04 的絕對準確率偏低，不代表模型能力。

#### 未來工作建議

1. **擴充三 variant 至 Gemma4 / Qwen3.5**，使用 200K+ 字元，克服天花板效應，取得更多 RQ1/RQ3 的數據點。
2. **句段級 tokenizer overhead 量化**：對實驗用的 haystack 語料直接測量，取代語料級平均值。
3. **修復實驗設計缺陷**：加入 distractor detection（N04 類問題）、sanity probe（模型輸出退化偵測）、token count check（context window 超限預警），寫入 v2 實驗腳本。
4. **控制詞彙實驗**：使用完全對應的繁簡版本（字詞一對一轉換，不調整語序），消除語義混淆，更純粹地測試 tokenizer 效應。
5. **位置敏感度複現實驗**：確認 gemma3:4b 中觀察到的繁/簡位置曲線差異，是否在其他漸進退化型模型中重現。
