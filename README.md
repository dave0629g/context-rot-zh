# 繁體中文 Context Rot 實驗

## 研究問題
繁體中文的 tokenizer 碎片化是否加速 context rot？

## 實驗設計

### 自變量
- 繁體 vs 簡體（同一段文字，OpenCC 轉換）

### 因變量
- Needle 檢索準確率（0 或 1）

### 控制變量
- 同一個模型（Ollama 本地運行）
- 同一段語意內容
- 同樣的 needle 位置
- 同樣的 context 長度（以字元數控制）

### 實驗矩陣
- 字形版本：繁體、簡體
- Context 長度：500, 1K, 2K, 4K, 8K, 16K, 32K 字元
- Needle 位置：10%, 30%, 50%, 70%, 90%
- 重複次數：每個組合 10 篇不同文章
- 模型：qwen3, llama3.1, gemma3（各自測試）

### 總實驗數
2 (繁/簡) × 7 (長度) × 5 (位置) × 10 (文章) × 3 (模型) = 2,100 次

## 執行步驟

### Step 0: 安裝依賴
```bash
pip install opencc-python-reimplemented wikipedia-api
```

### Step 1: 下載維基百科語料
```bash
python scripts/01_fetch_wiki.py
```

### Step 2: 建構 Haystack 和 Needle
```bash
python scripts/02_build_haystacks.py
```

### Step 3: 執行實驗
```bash
python scripts/03_run_experiment.py --model qwen3
```

### Step 4: 分析結果
```bash
python scripts/04_analyze.py
```

## 語料來源
- 維基百科繁體中文（透過 API 取得指定條目）
- 條目清單固定於 `configs/wiki_articles.json`
- 任何人可用相同清單復現

## 可復現性
- 所有 random seed 固定
- 所有條目 ID 記錄在設定檔
- 所有轉換腳本公開
