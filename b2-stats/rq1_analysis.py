"""
RQ1：分詞器家族（Gemma / LLaMA）vs 參數規模對 Context Rot 的解釋力。

樣本：退化型模型（gemma3:4b, gemma3:12b, llama3.1:8b, llama3.3:70b
        + gemma3:27b 在 65K 以下）

兩條 OLS：
  M_full_log_sp:  log10(starting_point_tokens) ~ tokenizer_family + scale_B
  M_full_decay :  decay_rate_pp_per_1k_tokens  ~ tokenizer_family + scale_B

對照模型計算 Unique R²：
  M_family : ~ tokenizer_family
  M_scale  : ~ scale_B
  M_full   : ~ tokenizer_family + scale_B

Unique R²(X) = R²(M_full) − R²(M_without_X)

輸出：rq1_results.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from compute_model_metrics import build_metrics_table, DEGRADING_MODELS


def fit_ols(df: pd.DataFrame, formula: str):
    res = smf.ols(formula, data=df).fit()
    return res


def model_summary(res, name: str) -> dict:
    return {
        "model": name,
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "n": int(res.nobs),
        "df_residual": int(res.df_resid),
        "f_pvalue": float(res.f_pvalue) if res.f_pvalue is not None else np.nan,
    }


def coef_rows(res, model_label: str, dv: str) -> list[dict]:
    rows = []
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    ci = res.conf_int(alpha=0.05)
    for term in params.index:
        rows.append({
            "dv": dv,
            "model": model_label,
            "term": term,
            "coef": float(params[term]),
            "std_err": float(bse[term]),
            "t": float(params[term] / bse[term]) if bse[term] else np.nan,
            "p_value": float(pvals[term]),
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
        })
    return rows


def unique_r2(full_r2: float, reduced_r2: float) -> float:
    return float(full_r2 - reduced_r2)


def run_one_dv(df: pd.DataFrame, dv_expr: str, dv_label: str) -> tuple[list, list]:
    """跑 family-only / scale-only / family+scale 三模型，回傳 coef 與 summary。"""
    coef_records = []
    summary_records = []

    f_family = f"{dv_expr} ~ C(tokenizer_family)"
    f_scale  = f"{dv_expr} ~ scale_B"
    f_full   = f"{dv_expr} ~ C(tokenizer_family) + scale_B"

    res_family = fit_ols(df, f_family)
    res_scale  = fit_ols(df, f_scale)
    res_full   = fit_ols(df, f_full)

    coef_records += coef_rows(res_family, "family_only",   dv_label)
    coef_records += coef_rows(res_scale,  "scale_only",    dv_label)
    coef_records += coef_rows(res_full,   "family+scale",  dv_label)

    summary_records.append({**model_summary(res_family, "family_only"),
                            "dv": dv_label})
    summary_records.append({**model_summary(res_scale,  "scale_only"),
                            "dv": dv_label})
    summary_records.append({**model_summary(res_full,   "family+scale"),
                            "dv": dv_label})

    # Unique R²：full − reduced (without family / without scale)
    uniq_family = unique_r2(res_full.rsquared, res_scale.rsquared)
    uniq_scale  = unique_r2(res_full.rsquared, res_family.rsquared)
    summary_records.append({
        "dv": dv_label, "model": "unique_r2",
        "r_squared": np.nan, "adj_r_squared": np.nan,
        "n": int(res_full.nobs), "df_residual": int(res_full.df_resid),
        "f_pvalue": np.nan,
        "unique_r2_family": uniq_family,
        "unique_r2_scale":  uniq_scale,
        "full_r2": float(res_full.rsquared),
    })
    return coef_records, summary_records


def main():
    # 主分析：drop_pp = 5
    tab = build_metrics_table(DEGRADING_MODELS, variant="traditional",
                              drop_pp=5.0, truncate_27b=True)
    tab["log10_sp_tokens"] = np.log10(tab["starting_point_tokens"])
    print("=== 模型層級指標（traditional, drop_pp=5）===")
    print(tab[["model", "tokenizer_family", "scale_B",
               "starting_point_tokens", "log10_sp_tokens",
               "decay_rate_pp_per_1k_tokens", "censored"]].to_string(index=False))

    coef_all = []
    summ_all = []

    # DV1: log10(starting_point_tokens)
    c1, s1 = run_one_dv(tab, "log10_sp_tokens", "log10_starting_point_tokens")
    coef_all += c1
    summ_all += s1

    # DV2: decay_rate
    c2, s2 = run_one_dv(tab, "decay_rate_pp_per_1k_tokens", "decay_rate_pp_per_1k_tokens")
    coef_all += c2
    summ_all += s2

    coef_df = pd.DataFrame(coef_all)
    summ_df = pd.DataFrame(summ_all)

    # 寫 CSV — 上半 = coef 表；下半 = summary 表
    out_path = "rq1_results.csv"
    with open(out_path, "w") as f:
        f.write("# RQ1 coefficients table\n")
        coef_df.to_csv(f, index=False)
        f.write("\n# RQ1 model summary table\n")
        summ_df.to_csv(f, index=False)
        f.write("\n# RQ1 raw model-level metrics (used as inputs)\n")
        tab.to_csv(f, index=False)

    print(f"\n寫入 {out_path}")
    print("\n=== Coefficients ===")
    print(coef_df.to_string(index=False))
    print("\n=== Summary ===")
    print(summ_df.to_string(index=False))

    return tab, coef_df, summ_df


if __name__ == "__main__":
    main()
