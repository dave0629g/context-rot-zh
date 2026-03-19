# 繁體中文 Context Rot 研究：14 天衝刺計劃

---

## 時間線總覽

```
Week 1  建構 + 實驗
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1   環境 + 語料 + 實驗材料（一天搞定基礎建設）
Day 2   探索性實驗（8B 粗掃，找差異區間）
Day 3   正式實驗 - 8B 三模型（白天盯，晚上跑）
Day 4   正式實驗 - 大模型 qwen3:32b + gemma3:27b
Day 5   正式實驗 - llama3.3:70b（跑整天 + 過夜）
Day 6   無詞界實驗（方向二）
Day 7   穩健性檢驗（法律 + 新聞語料）

Week 2  分析 + 寫作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 8   統計分析（顯著性檢驗 + 效果量）
Day 9   製圖（5 張核心圖）
Day 10  寫 Methodology + Results
Day 11  寫 Introduction + Background
Day 12  寫 Discussion + Conclusion + Abstract
Day 13  校稿 + 補圖 + 格式
Day 14  釋出 GitHub repo + 最終提交
```

---

## DAY 1（週一）：基礎建設日

一天內完成環境、語料、實驗材料。這是最關鍵的一天。

### 上午：環境準備（2 小時）

```
□ GX10 開機，確認 DGX OS 正常
  $ nvidia-smi
  $ free -h

□ 安裝 Ollama
  $ curl -fsSL https://ollama.com/install.sh | sh
  $ ollama serve &

□ 拉 8B 模型（背景下載，邊拉邊做其他事）
  $ ollama pull qwen3:8b &
  $ ollama pull llama3.1:8b &
  $ ollama pull gemma3:4b &

□ 安裝 Python 依賴
  $ pip install opencc-python-reimplemented

□ 解壓實驗程式碼
  $ unzip context-rot-zh.zip
  $ cd context-rot-zh
```

### 中午前：快速驗證（30 分鐘）

```
□ 跑 quick test
  $ python scripts/05_quick_test.py --model qwen3:8b

□ 確認看到：
  - 繁體 tokens > 簡體 tokens            □ 是
  - 繁簡比值大約 1.05~1.25               □ 是
  - 模型能正確回答 needle 問題            □ 是

  → 如果任何一項不通過，停下來排查再繼續
```

### 下午：語料準備（2 小時）

```
□ 下載維基百科語料
  $ python scripts/01_fetch_wiki_v2.py

□ 確認下載結果
  - 中文條目 ≥ 40 篇                     □ 是
  - 英文條目 ≥ 12 篇                     □ 是
  - 碎片化組繁簡差異字密度排行正確        □ 是
  - 字形干擾組部首密度明顯高於 baseline   □ 是

□ 隨機開 3 篇文章人工確認
  - 全是繁體中文                          □ 是
  - 沒有 HTML 殘留                        □ 是
```

### 傍晚：建構實驗材料（1 小時）

```
□ 建構 haystack
  $ python scripts/02_build_haystacks.py

□ 確認
  $ wc -l data/haystacks/experiments.jsonl
  - 行數 = 350（7長度 × 5位置 × 10試次）  □ 是

□ 抽查一筆繁簡對照
  - 繁體 needle 和簡體 needle 語意相同     □ 是
  - 字元數差異 < 1%                        □ 是
```

### 晚上：開始拉大模型（背景跑）

```
□ 拉正式實驗用的大模型
  $ ollama pull qwen3:32b &
  $ ollama pull gemma3:27b &
  $ ollama pull llama3.3:70b-q4_K_M &
  （過夜下載，明天早上確認）

□ git init + 第一次 commit
  $ git init && git add -A && git commit -m "Day 1: setup complete"
```

### Day 1 關卡
```
■ quick test 通過
■ 語料下載完成且品質確認
■ experiments.jsonl 已產生
■ 大模型開始下載
```

---

## DAY 2（週二）：探索性實驗

### 上午：8B 粗掃（3 小時）

```
□ 確認大模型下載完成
  $ ollama list

□ 用 qwen3:8b 跑粗粒度掃描
  只跑 4 個長度：500, 4K, 16K, 32K
  $ python scripts/03_run_experiment.py --model qwen3:8b
  預計 1~2 小時

□ 快速分析
  $ python scripts/04_analyze.py --model qwen3:8b
```

### 下午：判讀 + 調參（2 小時）

```
□ 記錄探索結果
  差異開始出現的長度：_______ 字元
  差異最大的長度：_______ 字元
  繁體準確率（最大差異處）：_______%
  簡體準確率（最大差異處）：_______%
  差距：_______%

□ 決定正式實驗的長度級別
  在差異區間加密取樣
  例如差異在 2K~16K → 用 500, 2K, 3K, 4K, 6K, 8K, 12K, 16K, 32K

□ 更新 configs 中的 context_lengths_chars

□ 用另外兩個 8B 模型也粗掃一次確認趨勢
  $ python scripts/03_run_experiment.py --model llama3.1:8b --max-experiments 80
  $ python scripts/03_run_experiment.py --model gemma3:4b --max-experiments 80

□ 三個模型趨勢一致嗎？
  - 繁體始終比簡體差                      □ 是
  - 差異出現的區間大致相同                 □ 是
```

### Day 2 關卡
```
■ 找到繁簡差異最明顯的 context 長度區間
■ 三個 8B 模型趨勢一致
■ 正式實驗的長度級別已確定
```

---

## DAY 3（週三）：正式實驗 - 8B

### 整天：三個 8B 模型完整跑

```
□ 重建實驗材料（用新的長度級別）
  $ python scripts/02_build_haystacks.py

□ 依序跑三個模型
  $ python scripts/03_run_experiment.py --model qwen3:8b
  （約 2-4 小時）

  $ python scripts/03_run_experiment.py --model llama3.1:8b
  （約 2-4 小時）

  $ python scripts/03_run_experiment.py --model gemma3:4b
  （約 2-4 小時）

□ 初步分析
  $ python scripts/04_analyze.py --all

□ 確認
  - 三個模型都跑完                         □ 是
  - 結果檔存在且完整                       □ 是
  - 備份結果                               □ 是
```

### Day 3 關卡
```
■ 三個 8B 模型正式實驗完成
■ 初步分析結果確認趨勢
```

---

## DAY 4（週四）：正式實驗 - 中型模型

### 白天：qwen3:32b + gemma3:27b

```
□ qwen3:32b（約 8-12 小時）
  $ python scripts/03_run_experiment.py --model qwen3:32b
  早上開跑 → 傍晚結束

□ gemma3:27b（晚上開跑，過夜）
  $ python scripts/03_run_experiment.py --model gemma3:27b

□ 等待期間：準備無詞界實驗（Day 6 要用）
  $ pip install ckip-transformers
  開始寫無詞界版本的 haystack 建構腳本
```

### Day 4 關卡
```
■ qwen3:32b 完成
■ gemma3:27b 開始跑（過夜）
■ CKIP 斷詞工具安裝好
```

---

## DAY 5（週五）：正式實驗 - 大模型

### 整天 + 過夜：llama3.3:70b

```
□ 確認 gemma3:27b 過夜結果
  $ python scripts/04_analyze.py --model gemma3:27b
  完整嗎？ □ 是

□ llama3.3:70b-q4_K_M（16-24 小時）
  $ python scripts/03_run_experiment.py --model llama3.3:70b-q4_K_M
  早上開跑 → 跑過整個週末

  中途確認沒掛掉：
  $ tail -1 results/llama3.3:70b-q4_K_M_results.jsonl

□ 等待期間：完成無詞界 haystack 建構腳本
  把 02_build_haystacks.py 複製為 02b_build_haystacks_spaced.py
  加入 CKIP 斷詞 + 插入空格的邏輯

□ 等待期間：下載穩健性語料
  - 全國法規資料庫：挑 3-5 條長法條
  - 找 3-5 篇公開新聞長文
  存入 data/robustness/legal/ 和 data/robustness/news/
```

### Day 5 關卡
```
■ gemma3:27b 結果確認
■ llama3.3:70b 開始跑
■ 無詞界腳本寫好
■ 穩健性語料下載完
```

---

## DAY 6（週六）：無詞界實驗

### 上午：確認 70B + 建構無詞界材料

```
□ 確認 llama3.3:70b 完成（或還在跑）
  如果還在跑 → 用 --resume 讓它繼續

□ 建構無詞界實驗材料
  $ python scripts/02b_build_haystacks_spaced.py
  產出：原始版 vs 加空格版

□ 驗證
  - 兩個版本字元數差異（加空格版稍長）    □ 確認
  - 語意完全相同                           □ 確認
```

### 下午 + 晚上：無詞界實驗

```
□ 只跑差異最明顯的 3-4 個長度 + 2 個模型
  $ python scripts/03_run_experiment_spaced.py --model qwen3:8b
  $ python scripts/03_run_experiment_spaced.py --model qwen3:32b

□ 快速分析
  有空格 vs 無空格的準確率差異：_______%
```

### Day 6 關卡
```
■ 無詞界實驗至少跑完 2 個模型
■ 有初步結果
```

---

## DAY 7（週日）：穩健性檢驗 + 彙總

### 上午：穩健性檢驗

```
□ 用法律 + 新聞語料重複碎片化核心實驗
  只跑差異最明顯的 3 個長度
  只跑 2 個模型（qwen3:8b, qwen3:32b）
  $ python scripts/03_run_experiment.py --model qwen3:8b --corpus legal
  $ python scripts/03_run_experiment.py --model qwen3:8b --corpus news
```

### 下午：彙總所有結果

```
□ 跑完整分析
  $ python scripts/04_analyze.py --all

□ 整理核心發現（手寫筆記）
  發現 1：繁體比簡體多 ___% 的 tokens
  發現 2：繁體 context rot 加速 ___% 
  發現 3：無詞界效應 ___
  發現 4：穩健性 ___
  發現 5：8B vs 70B 結論是否一致 ___

□ 備份所有結果
  $ tar czf results_backup_day7.tar.gz results/

□ git commit
  $ git add -A && git commit -m "Day 7: all experiments done"
```

### Day 7 關卡（Week 1 結束）
```
■ 碎片化實驗：6 個模型完成
■ 無詞界實驗：2 個模型完成
■ 穩健性檢驗：2 個領域完成
■ 所有結果已備份
■ 核心發現已記錄
```

---

## DAY 8（週一）：統計分析

### 整天

```
□ 安裝統計工具
  $ pip install scipy matplotlib seaborn pandas

□ 碎片化方向統計
  - Token 數差異的描述統計（均值、中位數、SD）
  - 每個長度級別的 McNemar's test
  - Cohen's h 效果量
  - 記錄：哪些長度 p < 0.05？

  長度      p-value    Cohen's h    顯著？
  500       _______    _______      □
  2K        _______    _______      □
  4K        _______    _______      □
  8K        _______    _______      □
  16K       _______    _______      □
  32K       _______    _______      □

□ Token 數 vs 準確率的相關分析
  - 碎片化比值與準確率差異的 Pearson r
  - r = _______  p = _______

□ 無詞界方向統計
  - 有空格 vs 無空格的 McNemar's test

□ 穩健性統計
  - 法律/新聞語料下的效果量
  - 與維基百科結果的一致性
```

### Day 8 關卡
```
■ 所有統計檢驗完成
■ p-value 和效果量記錄完整
■ 知道哪些結果是統計顯著的
```

---

## DAY 9（週二）：製圖

### 整天

```
□ Fig 1：衰退曲線對比圖（最重要的圖）
  X: context 長度
  Y: 準確率
  繁體紅線 + 簡體藍線 + 標準差陰影
  6 個子圖（每個模型一個）
  → 完成 □

□ Fig 2：Token 數差異圖
  X: context 長度
  Y: 繁/簡 token 比值
  每個模型一條線
  → 完成 □

□ Fig 3：位置偏差熱力圖
  X: needle 位置
  Y: context 長度
  繁簡各一張並排
  → 完成 □

□ Fig 4：跨模型一致性
  所有模型的繁簡差異集中顯示
  → 完成 □

□ Fig 5：穩健性 + 無詞界
  不同語料/條件的結果
  → 完成 □

□ 所有圖存成 PDF 和 PNG
  $ ls paper/figures/
```

### Day 9 關卡
```
■ 5 張核心圖完成
■ 圖表標注清晰完整
■ 繁簡差異在圖上一目了然
```

---

## DAY 10（週三）：寫 Methodology + Results

### 整天

```
□ Section 3: Methodology（2 頁）
  - 實驗設計（自變量/因變量/控制變量）
  - 語料來源和建構（維基百科、OpenCC）
  - 為什麼用字元數控制而不是 token 數
  - 模型選擇（3 個 tokenizer 家族 × 2 個大小）
  - Needle 設計（為什麼用中文數字）
  - 評估指標
  → 完成 □

□ Section 4: Results（3 頁）
  4.1 Token 碎片化分析
      繁體比簡體多 N% tokens，表格
  4.2 碎片化對 context rot 的影響
      衰退曲線圖 + 統計顯著性
  4.3 位置偏差
      熱力圖 + 分析
  4.4 無詞界實驗
      有空格 vs 無空格結果
  4.5 穩健性檢驗
      不同語料結果
  → 完成 □
```

### Day 10 關卡
```
■ Methodology 寫完
■ Results 寫完（含圖表引用）
```

---

## DAY 11（週四）：寫 Introduction + Background

### 整天

```
□ Section 1: Introduction（1.5 頁）
  - 長 context 應用越來越多
  - Chroma context rot 研究（引用）
  - 現有研究全基於英文（research gap）
  - 繁體中文的 tokenizer 碎片化問題
  - 本文貢獻（3 點）
  → 完成 □

□ Section 2: Background（1 頁）
  - BPE tokenizer 運作機制
  - 繁簡體在 UTF-8 層面的差異
  - Context rot 的定義和已知研究
  - Tokenizer 效率對模型性能的已知影響
  → 完成 □

□ 找齊引用文獻
  - Chroma context rot 報告
  - BPE 原始論文（Sennrich et al. 2016）
  - SentencePiece 論文（Kudo & Richardson 2018）
  - NIAH 原始論文
  - NoLiMa 論文
  → 至少 15 篇 references
  → 完成 □
```

### Day 11 關卡
```
■ Introduction 寫完
■ Background 寫完
■ References 至少 15 篇
```

---

## DAY 12（週五）：Discussion + Conclusion + Abstract

### 整天

```
□ Section 5: Discussion（1.5 頁）
  - 碎片化加速 context rot 的機制假說
  - 對 RAG 系統設計的實際建議
  - 對 tokenizer 設計者的建議
  - 局限性（語料只是維基百科、只測了推理不是訓練）
  → 完成 □

□ Section 6: Conclusion（0.5 頁）
  → 完成 □

□ Abstract（250 字）
  問題 → 方法 → 三個主要發現 → 意義
  → 完成 □

□ 通讀全文一次
  - 邏輯連貫嗎？                          □ 是
  - 圖表引用正確嗎？                      □ 是
  - 數字一致嗎？                          □ 是
```

### Day 12 關卡
```
■ 論文初稿全部完成
■ 通讀過一次
```

---

## DAY 13（週六）：校稿 + 補強

### 整天

```
□ 第二次通讀，標記問題
  - 語法和拼寫                             □ 檢查完
  - 圖表標注是否清晰                       □ 檢查完
  - 數字和百分比是否一致                   □ 檢查完
  - 每個 claim 都有數據支持嗎             □ 檢查完

□ 補強弱點
  - 哪個段落論述最薄弱？修改之
  - 有沒有遺漏的 related work？補充之
  - 圖表需要調色或改版嗎？

□ 格式整理
  - 選定投稿目標的格式（ACL? EMNLP? Arxiv?）
  - 調整頁面佈局和字型
  - 確認頁數符合要求

□ 找一個人讀（如果可能）
  - 請同事或朋友讀一遍
  - 收集回饋
```

### Day 13 關卡
```
■ 論文二稿完成
■ 格式符合目標要求
```

---

## DAY 14（週日）：釋出 + 提交

### 上午：準備 GitHub repo

```
□ 整理 repo 結構
  context-rot-zh/
  ├── README.md（含完整復現步驟）
  ├── requirements.txt
  ├── configs/
  ├── scripts/
  ├── paper/figures/
  └── PLAN.md

□ README 包含
  - 一句話摘要
  - 復現步驟
  - 使用的 Wikipedia dump 日期
  - 所有 random seed
  - Ollama 版本 + 模型版本
  - 引用格式（BibTeX）

□ 推到 GitHub
  $ git push origin main
```

### 下午：最終提交

```
□ 論文最終版 PDF
  - 再讀一次 Abstract
  - 確認作者資訊
  - 確認 GitHub repo 連結在論文裡

□ 提交
  - Arxiv 或目標會議
  - 記錄提交時間和 submission ID

□ 慶祝 🎉
```

### Day 14 關卡
```
■ GitHub repo 公開
■ 論文提交
■ 完成
```

---

## 每日例行事項

```
每天早上第一件事：
  $ tail -5 results/*_results.jsonl    # 確認過夜的實驗還活著

每天結束前：
  $ git add -A && git commit -m "Day N: ___"
  $ cp -r results/ /備份路徑/           # 備份結果
```

---

## 如果卡住了

```
問題：模型跑太慢，70B 跑不完
解法：砍掉 70B，只用 8B + 32B，論文照樣成立

問題：繁簡差異不顯著
解法：這本身是發現。轉向寫「碎片化不影響 context rot」
      加上 token 成本效率分析（繁體花更多 token 但品質相同）

問題：某個模型中文能力太差
解法：排除該模型，換 deepseek-r1:32b 或其他

問題：維基百科某些條目下載失敗
解法：用其他同類條目替代，記錄在論文裡

問題：統計不顯著
解法：增加試次到 20 次，或合併相鄰長度級別增加樣本數

問題：寫不完
解法：先投 Arxiv preprint，不需要通過審稿
      之後再改投會議
```

---

## 最終交付物清單

```
□ 論文 PDF（8-10 頁）
□ GitHub repo（含所有程式碼和設定）
□ README（含完整復現步驟）
□ 5 張核心圖表
□ 統計分析結果
□ 實驗原始數據（jsonl 格式）
```
