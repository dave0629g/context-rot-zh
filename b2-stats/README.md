# b2-stats — RQ1–RQ3 推論統計（含 H2 的 cell-level OLS / LRT）

本目錄收錄論文第四章推論統計之**可重現腳本與輸出**，資料來源為本 repo `results/` 之原始 JSONL（含 `h2_*.jsonl` simplified_q 補充資料）。所有 `is_correct` 於載入時以 `scripts/04_analyze.py` 的 `reevaluate()` 重算，與 `results/analysis/*.json` 口徑一致。

## 重現方式

```bash
# 於 repo 根目錄
python b2-stats/rq2_analysis.py    # RQ2 / H2：配對 t、cell-level OLS、VIF、LRT
python b2-stats/rq1_analysis.py    # RQ1：分組變項 OLS、Unique R²
python b2-stats/rq3_analysis.py    # RQ3：負擔比率 vs 起始點提前量
```

`compute_model_metrics.py` 會自動偵測 repo 根（支援「b2-stats/ 置於 repo 內」與「b2-stats/ 旁掛 context-rot-zh/ 子目錄」兩種佈局）。

## H2 核心結果（rq2_analysis.py → rq2_results.csv）

cell-level OLS（單位＝model × variant × length × position × needle，n=3,960；
2 個代表模型 gemma3:4b、llama3.1:8b × 3 變體 × 12 長度 × 11 位置 × 5 關鍵資訊）：

- **M1**: `prop_correct ~ log10(length_chars) + C(variant) + C(needle_id) + C(model)` → R²=.125
- **M2**: M1 + `token_overhead` → R²=.155
- **LRT (M2 vs M1)**: **χ²(1) = 137.69, p = 8.51×10⁻³², ΔR² = .030**
- `token_overhead` 係數 p = 1.02×10⁻³¹

> ⚠️ 重現須合併 `load_results(m) + load_results_h2(m)`（即 `results/*.jsonl` 與 `results/h2_*.jsonl`）。
> 僅用主 results 而漏掉 h2 補充資料，會低估為 χ²≈7.65。
