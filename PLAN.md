# 繁體中文 Context Rot 研究計劃

## 研究題目
Tokenizer Fragmentation and Context Rot: How Traditional Chinese Script Impacts Long-Context LLM Performance

## 一句話摘要
同一段語意，繁體中文因 tokenizer 碎片化產生更多 tokens，導致模型在長 context 下的性能衰退比簡體中文更嚴重。

---

## Phase 0：環境準備（第 1 天）

### 0.1 硬體確認
- [ ] ASUS Ascent GX10 開機，確認 DGX OS 正常
- [ ] 確認 128GB 統一記憶體可用：`nvidia-smi` 或 `free -h`
- [ ] 確認網路連線正常（需下載模型和維基百科）

### 0.2 安裝 Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # 背景執行
```

### 0.3 下載模型（第一組：8B 快速驗證用）
```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull gemma3:4b
```
預計下載時間：每個模型約 5-15 分鐘

### 0.4 下載模型（第二組：正式實驗用）
```bash
ollama pull qwen3:32b
ollama pull llama3.3:70b-q4_K_M
ollama pull gemma3:27b
```
預計下載時間：70B 模型約 30-60 分鐘

### 0.5 驗證模型可運行
```bash
ollama run qwen3:8b "你好，請用繁體中文回答：1+1等於多少？"
ollama run llama3.1:8b "你好，請用繁體中文回答：1+1等於多少？"
ollama run gemma3:4b "你好，請用繁體中文回答：1+1等於多少？"
```
確認三個模型都能正常回覆繁體中文。

### 0.6 安裝 Python 依賴
```bash
pip install opencc-python-reimplemented
```

### 0.7 下載實驗程式碼
將 context-rot-zh.zip 解壓到工作目錄。

### 完成標準
- [ ] 三個 8B 模型都能回覆繁體中文
- [ ] `python scripts/05_quick_test.py --model qwen3:8b` 跑通
- [ ] 看到繁簡體 token 數差異的輸出

---

## Phase 1：語料準備（第 2-3 天）

### 1.1 下載維基百科語料
```bash
python scripts/01_fetch_wiki_v2.py
```

### 1.2 檢查下載結果
確認以下統計數字：
- [ ] 中文條目：至少 40 篇成功下載
- [ ] 英文條目：至少 12 篇成功下載（有 en_title 的）
- [ ] 每篇文章至少 3,000 字元

### 1.3 檢查語料品質
打開 `data/wiki_raw_v2/corpus_metadata.json`，確認：
- [ ] 碎片化組的繁簡差異字密度排行在前
- [ ] 字形干擾組的對應部首密度明顯高於 baseline
- [ ] 每個方向至少有 5 篇可用文章

### 1.4 手動抽查
隨機打開 3 篇下載的文章，確認：
- [ ] 全部是繁體中文（不是簡體）
- [ ] 沒有 HTML 殘留標記
- [ ] 內容完整，不是截斷的

### 完成標準
- [ ] `corpus_metadata.json` 存在且內容完整
- [ ] 語料品質通過人工抽查

---

## Phase 2：建構實驗材料（第 3-4 天）

### 2.1 建構方向一（碎片化）的 haystack
```bash
python scripts/02_build_haystacks.py
```
此步驟會：
- 從維基百科文章拼接不同長度的 haystack
- 用 OpenCC 轉換繁簡體版本
- 在指定位置插入 needle
- 輸出到 `data/haystacks/experiments.jsonl`

### 2.2 驗證實驗材料
```bash
# 檢查產生的實驗數量
wc -l data/haystacks/experiments.jsonl

# 預期：7 長度 × 5 位置 × 10 試次 = 350 行
```

### 2.3 抽查繁簡對照品質
```python
# 手動抽查一筆
import json
with open("data/haystacks/experiments.jsonl") as f:
    exp = json.loads(f.readline())

print("繁體 needle:", exp["traditional"]["needle"])
print("簡體 needle:", exp["simplified"]["needle"])
print("繁體字元數:", exp["traditional"]["stats"]["char_count"])
print("簡體字元數:", exp["simplified"]["stats"]["char_count"])
# 字元數應該幾乎相同（OpenCC 只改字形，不改字數）
```

### 完成標準
- [ ] `experiments.jsonl` 存在且行數正確
- [ ] 繁簡體 needle 語意相同但字形不同
- [ ] 繁簡體字元數差異 < 1%

---

## Phase 3：探索性實驗（第 5-6 天）

### 目的
用 8B 模型快速跑一輪，找出繁簡差異最明顯的 context 長度區間。

### 3.1 粗粒度掃描
先只跑 4 個長度級別，加速探索：
```bash
# 修改 configs 或用 --max-experiments 限制數量
python scripts/03_run_experiment.py --model qwen3:8b --max-experiments 160
# 4 長度 × 5 位置 × 2 繁簡 × 10 試次 × (只跑部分) = ~160 次
# 預計時間：約 10-20 分鐘
```

### 3.2 快速分析
```bash
python scripts/04_analyze.py --model qwen3:8b
```

### 3.3 判斷結果
看輸出的準確率表格，找出：
- [ ] 繁簡差異在哪個長度開始出現？
- [ ] 差異在哪個長度最大？
- [ ] 差異的方向是否一致（繁體始終較差）？

### 3.4 記錄發現
在筆記本記下：
```
探索結果：
  差異開始出現的長度：_____ 字元
  差異最大的長度：_____ 字元
  繁體在最大差異處的準確率：_____%
  簡體在最大差異處的準確率：_____%
```

### 3.5 決定正式實驗的長度級別
根據探索結果，調整 context_lengths_chars：
```
假設差異在 2K~16K 最明顯：
  → 調整為：500, 2K, 3K, 4K, 6K, 8K, 12K, 16K, 32K
  在差異區間加密取樣
```

### 完成標準
- [ ] 找到繁簡差異最明顯的區間
- [ ] 確認差異方向一致（不是隨機波動）
- [ ] 決定正式實驗的長度級別

---

## Phase 4：正式實驗 - 碎片化方向（第 7-10 天）

### 4.1 用三個 8B 模型跑完整實驗
```bash
python scripts/03_run_experiment.py --model qwen3:8b
python scripts/03_run_experiment.py --model llama3.1:8b
python scripts/03_run_experiment.py --model gemma3:4b
```
每個模型約 2-4 小時，可以依序跑或跑過夜。

### 4.2 分析 8B 結果
```bash
python scripts/04_analyze.py --all
```
確認三個模型的結果趨勢一致。

### 4.3 用 32B~70B 模型跑正式實驗
```bash
python scripts/03_run_experiment.py --model qwen3:32b
# 預計 8-12 小時

python scripts/03_run_experiment.py --model gemma3:27b
# 預計 6-10 小時

python scripts/03_run_experiment.py --model llama3.3:70b-q4_K_M
# 預計 16-24 小時，建議跑過夜
# 如果中斷可用 --resume 繼續
```

### 4.4 記錄每次實驗的 token 數
03_run_experiment.py 會自動用 Ollama 的 tokenize API 記錄每筆實驗的實際 token 數。
這是關鍵數據——繁簡體在同一模型上的 token 數差異。

### 4.5 完整分析
```bash
python scripts/04_analyze.py --all
```

### 完成標準
- [ ] 6 個模型（3 個 8B + 3 個 32B~70B）全部跑完
- [ ] 繁簡差異在不同模型間一致
- [ ] Token 數差異有完整記錄
- [ ] 8B 和大模型的結論方向一致

---

## Phase 5：正式實驗 - 無詞界方向（第 11-14 天）

### 5.1 安裝斷詞工具
```bash
pip install ckip-transformers
# 或
pip install jieba
```

### 5.2 建構無詞界實驗材料
需要新寫一個腳本（基於 02_build_haystacks.py 修改）：
- 輸入：繁體 haystack（Phase 2 已產生）
- 處理：用 CKIP 斷詞後插入空格
- 輸出：原始版（無詞界）vs 加空格版（有詞界）

### 5.3 執行實驗
同 Phase 4 的流程，但比較的是「有空格 vs 無空格」而不是「繁體 vs 簡體」。

### 5.4 分析
觀察有無詞界對 context rot 的影響。

### 完成標準
- [ ] 有空格版和無空格版的 haystack 建構完成
- [ ] 至少 3 個模型跑完
- [ ] 有無詞界的準確率差異有統計結果

---

## Phase 6：穩健性檢驗（第 15-17 天）

### 6.1 準備不同領域語料
手動下載 3~4 個領域的公開繁體中文語料：
- [ ] 法律：全國法規資料庫（https://law.moj.gov.tw/）
- [ ] 新聞：公視新聞或中央社公開報導
- [ ] 文學：國家圖書館公共領域作品

### 6.2 對碎片化方向做穩健性檢驗
用不同領域語料重複 Phase 4 的核心實驗：
- 只跑差異最明顯的 3~4 個長度級別
- 只跑 2~3 個模型
- 觀察結論是否在不同領域語料下仍然成立

### 6.3 記錄領域差異
如果某些領域的繁簡差異更大或更小，這本身就是有趣的發現。

### 完成標準
- [ ] 至少 3 個領域的穩健性檢驗完成
- [ ] 結論在不同領域下一致（或記錄不一致的原因）

---

## Phase 7：統計分析（第 18-20 天）

### 7.1 描述性統計
- 繁體 vs 簡體的平均 token 數比值
- 各 context 長度下的準確率均值和標準差
- 各 needle 位置的準確率分布

### 7.2 統計顯著性檢驗
```python
from scipy import stats

# 對每個 context 長度
# 比較繁體 vs 簡體的準確率
# 用 McNemar's test（配對二元資料）
# 或 Fisher's exact test

for length in context_lengths:
    trad_correct = [...]  # 繁體的正確/錯誤列表
    simp_correct = [...]  # 簡體的正確/錯誤列表
    
    # McNemar's test
    result = stats.mcnemar(contingency_table)
    print(f"長度 {length}: p = {result.pvalue:.4f}")
```

### 7.3 效果量計算
```python
# Cohen's h 用於比較兩個比例
import numpy as np

def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

# 繁體準確率 vs 簡體準確率的效果量
```

### 7.4 Token 數與準確率的相關分析
```python
# 繁簡 token 數差異比 vs 準確率差異
# 觀察碎片化程度是否能預測 context rot 加速程度
```

### 完成標準
- [ ] 主要發現有 p-value 支持
- [ ] 效果量已計算
- [ ] Token 數差異與準確率差異的相關性已分析

---

## Phase 8：製圖（第 20-22 天）

### 8.1 核心圖表清單
1. **Fig 1: 衰退曲線對比圖**
   - X 軸：context 長度（字元數 + 對應的 token 數）
   - Y 軸：needle 檢索準確率
   - 兩條線：繁體（紅）、簡體（藍）
   - 陰影區域：標準差
   - 每個模型一個子圖

2. **Fig 2: Token 數差異圖**
   - X 軸：context 長度
   - Y 軸：繁/簡 token 數比值
   - 每個模型一條線
   - 顯示碎片化程度隨長度的變化

3. **Fig 3: 位置偏差熱力圖**
   - X 軸：needle 位置（10%~90%）
   - Y 軸：context 長度
   - 顏色：準確率
   - 繁體和簡體各一張，並排比較

4. **Fig 4: 跨模型一致性圖**
   - 所有模型的繁簡差異疊在一起
   - 顯示結論的穩健性

5. **Fig 5: 穩健性檢驗圖（如果做了 Phase 6）**
   - 不同領域語料下的繁簡差異
   - 驗證結論的推廣性

### 8.2 工具
```bash
pip install matplotlib seaborn
```

### 完成標準
- [ ] 5 張核心圖表完成
- [ ] 圖表清晰、標注完整
- [ ] 繁簡差異在圖上一目了然

---

## Phase 9：論文撰寫（第 23-30 天）

### 9.1 論文結構

```
Title:
  Tokenizer Fragmentation and Context Rot:
  How Traditional Chinese Script Impacts Long-Context LLM Performance

Abstract (250 字)
  問題 → 方法 → 主要發現 → 意義

1. Introduction (1.5 頁)
  - 長 context LLM 的實際應用越來越多
  - Chroma 的 context rot 研究揭示了衰退問題
  - 但現有研究全部基於英文
  - 繁體中文有獨特的 tokenizer 碎片化問題
  - 本文的貢獻

2. Background (1 頁)
  - BPE tokenizer 的運作機制
  - 繁簡體在 UTF-8 和 tokenizer 層面的差異
  - Context rot 的已知研究

3. Methodology (2 頁)
  - 實驗設計（自變量、因變量、控制變量）
  - 語料來源和建構方式
  - 模型選擇和執行環境
  - 評估指標

4. Experiments & Results (3 頁)
  4.1 Token 碎片化分析（繁簡 token 數差異）
  4.2 碎片化對 context rot 的影響（衰退曲線比較）
  4.3 位置偏差分析
  4.4 無詞界實驗結果（如果做了）
  4.5 穩健性檢驗（不同領域、不同模型大小）

5. Discussion (1.5 頁)
  - 碎片化加速 context rot 的機制假說
  - 對 RAG 系統設計的實際影響
  - 對 tokenizer 設計的建議
  - 局限性

6. Conclusion (0.5 頁)

References
Appendix
  - 完整的語料條目清單
  - 所有 needle 文本
  - 補充統計表格
```

### 9.2 撰寫順序
1. 先寫 Methodology（最確定的部分）
2. 再寫 Results（有圖表支撐）
3. 然後 Introduction 和 Background
4. 最後 Discussion 和 Abstract

### 完成標準
- [ ] 論文初稿完成（約 8-10 頁）
- [ ] 所有圖表嵌入
- [ ] References 格式正確

---

## Phase 10：釋出可復現材料（第 30-32 天）

### 10.1 GitHub Repository 內容
```
context-rot-zh/
├── README.md                     # 完整說明
├── requirements.txt              # 依賴
├── configs/
│   ├── wiki_articles_v2.json     # 語料設定（含選擇理由）
│   └── experiment_params.json    # 實驗參數
├── scripts/
│   ├── 01_fetch_wiki_v2.py       # 下載語料
│   ├── 02_build_haystacks.py     # 建構實驗材料
│   ├── 03_run_experiment.py      # 執行實驗
│   ├── 04_analyze.py             # 分析結果
│   └── 05_quick_test.py          # 快速測試
├── results/
│   └── (不放原始數據，太大)
└── paper/
    └── figures/                  # 論文圖表的生成腳本
```

### 10.2 釋出清單
- [ ] 所有程式碼推到 GitHub
- [ ] README 包含完整復現步驟
- [ ] 標記使用的 Wikipedia dump 日期
- [ ] 標記所有 random seed
- [ ] 標記 Ollama 和模型版本號

---

## 時間線總覽

```
     第 1 天   Phase 0  環境準備
  第 2-3 天   Phase 1  語料準備
  第 3-4 天   Phase 2  建構實驗材料
  第 5-6 天   Phase 3  探索性實驗（8B 快速掃描）
 第 7-10 天   Phase 4  正式實驗 - 碎片化方向
第 11-14 天   Phase 5  正式實驗 - 無詞界方向
第 15-17 天   Phase 6  穩健性檢驗
第 18-20 天   Phase 7  統計分析
第 20-22 天   Phase 8  製圖
第 23-30 天   Phase 9  論文撰寫
第 30-32 天   Phase 10 釋出可復現材料
```

---

## 風險與備案

### 風險 1：繁簡差異不顯著
可能結果：繁體和簡體的 context rot 曲線幾乎重疊。
備案：這本身也是有價值的發現（「tokenizer 碎片化不影響 context rot」），
      但需要確認 token 數差異確實存在（碎片化是真實的，只是不影響性能）。
      轉向強調 token 數效率的經濟意義（繁體花更多 token，成本更高，但品質相同）。

### 風險 2：模型中文能力不足
可能結果：某些模型在中文 needle 檢索上全面表現差，無法區分繁簡差異。
備案：排除該模型，增加其他中文能力更強的模型（如 deepseek-r1:32b）。
      在論文中說明排除理由。

### 風險 3：32K 字元超出某些模型的有效 context
可能結果：某些 8B 模型在長 context 下完全失效。
備案：降低最大 context 長度到 16K 字元。
      或只在大模型上跑長 context 的實驗。

### 風險 4：Ollama tokenize API 不支援某些模型
可能結果：無法取得精確 token 數。
備案：改用 Hugging Face tokenizer 離線計算。
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokens = tokenizer.encode(text)
token_count = len(tokens)
```

### 風險 5：70B 模型跑太慢
可能結果：完整實驗矩陣需要超過一週。
備案：減少試次（從 10 次降到 5 次），
      或只在差異最明顯的長度級別上跑 70B。

---

## 每個 Phase 結束時要做的事

1. 備份所有結果到外部儲存
2. 在筆記本記錄關鍵發現和決策理由
3. 確認可以從當前狀態重新開始下一個 Phase
4. git commit 所有程式碼變更
