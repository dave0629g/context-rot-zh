# B2 推論統計結果摘要

設定（共用）：
- 樣本：退化型模型 = gemma3:4b、gemma3:12b、gemma3:27b（≤65K 字元）、llama3.1:8b、llama3.1:70b、llama3.3:70b。Qwen3.5 與 Gemma4 全系列因無衰減已排除。
- 衰減起始點（starting_point）：以 ≤8K 字元的平均準確率為 baseline，首次降幅 ≥ 5pp 的長度；右截尾以實驗範圍最大長度 imputation。
- decay_rate：對 (avg_tokens, accuracy%) 做最小平方法線性迴歸，斜率 × 1 = pp per 1000 tokens。
- 評估口徑：採用 04_analyze.py 的 reevaluate（含 OpenCC 簡繁轉換、中文數字正規化、同義詞替換），與 results/analysis/ JSON 一致。

---

## RQ1 結果（traditional 變項，n = 6）

模型層級指標：

| model        | family | scale_B | sp_tokens   | log10_sp_tokens | decay_rate (pp/1K tokens) | censored |
|--------------|--------|---------|-------------|-----------------|---------------------------|----------|
| gemma3:4b    | Gemma  | 4       | 49,062.28   | 4.6907          | -0.4622                   | False    |
| gemma3:12b   | Gemma  | 12      | 75,529.10   | 4.8781          | -0.1224                   | False    |
| gemma3:27b   | Gemma  | 27      | 49,062.28   | 4.6907          | -0.0167                   | True     |
| llama3.1:8b  | LLaMA  | 8       | 57,248.30   | 4.7578          | -0.0776                   | False    |
| llama3.1:70b | LLaMA  | 70      | 114,661.24  | 5.0594          | -0.1694                   | False    |
| llama3.3:70b | LLaMA  | 70      | 114,661.24  | 5.0594          | -0.1188                   | False    |

### DV1：Y = log10(starting_point_tokens)

完整模型（family + scale）：R² = 0.776，adj R² = 0.627，n = 6，F p = .106。
- 截距 = 4.692 (SE = 0.067, p < .001)
- C(tokenizer_family)[T.LLaMA] 係數 = 0.057 (SE = 0.110, p = .639)
- scale_B 係數 = 0.00424 (SE = 0.00197, p = .120)

對照模型：
- family-only：R² = 0.430，p = .157。
- scale-only：R² = 0.756，p = .024，scale_B 係數 = 0.00489 (SE = 0.00139, p = .024)。

Unique R²：
- tokenizer_family：0.020
- scale：0.346

### DV2：Y = decay_rate (pp / 1000 tokens)

完整模型（family + scale）：R² = 0.089，adj R² = −0.519，n = 6，F p = .870。
- 截距 = −0.211 (SE = 0.123, p = .184)
- C(tokenizer_family)[T.LLaMA] 係數 = 0.053 (SE = 0.202, p = .811)
- scale_B 係數 = 0.000741 (SE = 0.00361, p = .851)

對照模型：
- family-only：R² = 0.076，p = .597。
- scale-only：R² = 0.068，p = .618。

Unique R²：
- tokenizer_family：0.021
- scale：0.013

樣本限制：n = 6，DV1 在 scale-only 達到 p < .05；其餘 F p > .05，係數標準誤偏大。

---

## RQ2 結果

範圍：gemma3:4b 與 llama3.1:8b（固定樣本，不受 RQ1/RQ3 樣本擴充影響）。

### (a) 配對 t 檢定 + Cohen's dz（traditional vs simplified_q，配對單位：(model, length, position, needle_id)）

| model       | length | n_pairs | mean_trad | mean_simp_q | mean_diff | t      | df | p      | cohen_dz |
|-------------|-------:|--------:|----------:|------------:|----------:|-------:|---:|-------:|---------:|
| gemma3:4b   | 65,000 | 55      | 0.8818    | 0.9273      | -0.0455   | -1.695 | 54 | .096   | -0.229   |
| gemma3:4b   | 100,000| 55      | 0.6818    | 0.6818      |  0.0000   |  0.000 | 54 | 1.000  |  0.000   |
| gemma3:4b   | 130,000| 55      | 0.5273    | 0.5273      |  0.0000   |  0.000 | 54 | 1.000  |  0.000   |
| llama3.1:8b | 65,000 | 55      | 0.9273    | 0.9636      | -0.0364   | -1.659 | 54 | .103   | -0.224   |
| llama3.1:8b | 100,000| 55      | 0.9091    | 0.9182      | -0.0091   | -0.299 | 54 | .766   | -0.040   |
| llama3.1:8b | 130,000| 55      | 0.9182    | 0.8273      |  0.0909   |  2.324 | 54 | .024 * |  0.313   |

（mean_diff 為 traditional − simplified_q；負值代表繁體準確率較低。）

### (b) Cell-level OLS（n = 3,960；單位 = (model, length, position, needle, variant)）

VIF 共線性檢查：
- length_chars vs length_tokens：VIF = 95.07（兩變項共線性極高）。
- log10_length_chars vs token_overhead：VIF = 1.37（共線性可忽略）。

模型比較（巢狀）：
- M1：prop_correct ~ log10(length_chars) + C(variant) + C(needle_id) + C(model)
  - R² = 0.1249，log-likelihood = 1188.70，AIC = −2359.40，BIC = −2302.85
- M2：M1 + token_overhead
  - R² = 0.1548，log-likelihood = 1257.55，AIC = −2495.09，BIC = −2432.25

LRT (M2 vs M1)：χ²(1) = 137.69，p = 8.51 × 10⁻³². AIC 與 BIC 均下降。

M2 主要係數：
- 截距 = 1.1753 (SE = 0.0209, p < .001)
- log10(length_chars) 係數 = −0.0549 (SE = 0.0051, p = 4.90 × 10⁻²⁷)
- token_overhead 係數 = 2.61 × 10⁻⁵ (SE = 1.55 × 10⁻⁶, p = 1.02 × 10⁻³¹)
- C(variant)[T.traditional] 係數 = −0.0522 (SE = 0.0078, p = 2.81 × 10⁻¹¹)
- C(variant)[T.simplified_q] 係數 = −0.0057 (SE = 0.0069, p = .409)
- C(model)[T.llama3.1:8b] 係數 = 0.0518 (SE = 0.0056, p = 4.01 × 10⁻²⁰)
- C(needle_id) 控制項：N03 +0.0331 (p < .001)、N05 +0.0242 (p = .006)、其餘 p > .05

每模型 fertility (tokens / chars)：gemma3:4b = 0.7437，llama3.1:8b = 0.8826。

---

## RQ3 結果

家族 fertility（traditional，tokens / char）：Gemma = 0.7420，LLaMA = 0.9098。

每個退化型模型的衰減提前量（advance = sp_simp_q − sp_trad；正值代表繁體較早衰減）：

| model        | family | sp_trad_chars | sp_simp_q_chars | advance_chars | advance_tokens |
|--------------|--------|--------------:|----------------:|--------------:|---------------:|
| gemma3:4b    | Gemma  |        65,000 |          65,000 |             0 |              0 |
| gemma3:12b   | Gemma  |       100,000 |         100,000 |             0 |              0 |
| gemma3:27b   | Gemma  |        65,000 |          65,000 |     0 (截尾)  |              0 |
| llama3.1:8b  | LLaMA  |        65,000 |         100,000 |        35,000 |      31,844.20 |
| llama3.1:70b | LLaMA  |       130,000 |         130,000 |             0 |              0 |
| llama3.3:70b | LLaMA  |       130,000 |         130,000 |             0 |              0 |

家族平均：

| family | n_models | mean_advance_chars | sd_advance_chars | mean_advance_tokens | sd_advance_tokens |
|--------|---------:|-------------------:|-----------------:|--------------------:|------------------:|
| Gemma  | 3        |               0    |          0       |               0     |           0       |
| LLaMA  | 3        |          11,666.67 |     20,207.26    |          10,614.73  |      18,385.26    |

（長度採樣為離散 12 點 {500, 2K, 4K, 6K, 8K, 12K, 16K, 24K, 32K, 65K, 100K, 130K}；多數模型 sp_trad 與 sp_simp_q 落在同一 bucket，故 advance = 0；gemma3:27b 因 ≤65K 限制，繁簡均右截尾於 65K。）
