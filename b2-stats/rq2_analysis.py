"""
RQ2：Token 計數的獨立效應。

(a) 配對 t 檢定 + Cohen's d（dz）
    對象：gemma3:4b、llama3.1:8b
    長度：65K、100K、130K
    配對單位：(model, length, position, needle_id)
    比較：traditional vs simplified_q 的 prop_correct（每 cell 2 trials 的平均）

(b) Cell-level OLS：
    單位：model × length × position × needle × variant
    Y = prop_correct
    X = log10(length_chars) + token_overhead + C(variant) + C(needle_id) + C(model)
    token_overhead_cell = tokens_cell − fertility_model × chars_cell

    VIF 共線性檢查：length_chars vs length_tokens 應 VIF > 50；token_overhead 應 VIF < 5
    LRT 比較：
       M1: Y ~ log10(length_chars) + C(variant) + C(needle_id) + C(model)
       M2: M1 + token_overhead

輸出：rq2_results.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from compute_model_metrics import (
    load_results, load_results_h2, to_dataframe,
)


MODELS_RQ2 = ["gemma3:4b", "llama3.1:8b"]
LENGTHS_PAIR_T = [65000, 100000, 130000]


def build_cell_df(models: list[str]) -> pd.DataFrame:
    """以 (model, variant, length, position, needle_id) 為一 cell，
    prop_correct = 該 cell 內 trial 的平均 is_correct。
    回傳含 tokens_cell（cell 內平均 prompt token）的 DataFrame。"""
    parts = []
    for m in models:
        df = to_dataframe(load_results(m) + load_results_h2(m))
        df["model"] = m
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    cell = df.groupby(
        ["model", "variant", "length_chars", "position", "needle_id"],
        as_index=False,
    ).agg(
        prop_correct=("is_correct", "mean"),
        tokens_cell=("tokens_prompt", "mean"),
        n_trials=("is_correct", "size"),
    )
    return cell


def paired_t_and_d(trad: np.ndarray, simp_q: np.ndarray) -> dict:
    """配對 t 與 Cohen's dz。"""
    assert len(trad) == len(simp_q)
    n = len(trad)
    diff = trad - simp_q
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    if sd_diff == 0 or n < 2:
        return {"n_pairs": n, "mean_trad": float(np.mean(trad)),
                "mean_simp_q": float(np.mean(simp_q)),
                "mean_diff": mean_diff, "sd_diff": sd_diff,
                "t": np.nan, "df": n - 1, "p_value": np.nan,
                "cohen_dz": np.nan}
    t_stat, p = stats.ttest_rel(trad, simp_q)
    dz = mean_diff / sd_diff
    return {"n_pairs": n, "mean_trad": float(np.mean(trad)),
            "mean_simp_q": float(np.mean(simp_q)),
            "mean_diff": mean_diff, "sd_diff": sd_diff,
            "t": float(t_stat), "df": n - 1,
            "p_value": float(p), "cohen_dz": float(dz)}


def run_paired_t(cell: pd.DataFrame) -> pd.DataFrame:
    """對每個 (model, length) 配對 t + Cohen's d。"""
    rows = []
    for m in MODELS_RQ2:
        for L in LENGTHS_PAIR_T:
            sub = cell[(cell["model"] == m) & (cell["length_chars"] == L) &
                       (cell["variant"].isin(["traditional", "simplified_q"]))]
            wide = sub.pivot_table(
                index=["position", "needle_id"],
                columns="variant",
                values="prop_correct",
            ).dropna()
            trad = wide["traditional"].to_numpy()
            simp_q = wide["simplified_q"].to_numpy()
            r = paired_t_and_d(trad, simp_q)
            r["model"] = m
            r["length_chars"] = L
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["model", "length_chars", "n_pairs", "mean_trad", "mean_simp_q",
            "mean_diff", "sd_diff", "t", "df", "p_value", "cohen_dz"]
    return df[cols]


def fertility_by_model(cell: pd.DataFrame) -> dict:
    """每個 model 的 fertility = mean(tokens / chars)，
    使用所有 (variant, length, position, needle) cells。"""
    cell = cell.copy()
    cell["fert"] = cell["tokens_cell"] / cell["length_chars"]
    return cell.groupby("model")["fert"].mean().to_dict()


def add_overhead_cols(cell: pd.DataFrame) -> pd.DataFrame:
    cell = cell.copy()
    fert = fertility_by_model(cell)
    cell["fertility_model"] = cell["model"].map(fert)
    cell["token_overhead"] = (
        cell["tokens_cell"] - cell["fertility_model"] * cell["length_chars"]
    )
    cell["log10_length_chars"] = np.log10(cell["length_chars"])
    cell["length_tokens"] = cell["tokens_cell"]
    return cell


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float).to_numpy()
    X_with_const = np.column_stack([np.ones(len(X)), X])
    rows = []
    for i, c in enumerate(cols):
        v = variance_inflation_factor(X_with_const, i + 1)
        rows.append({"variable": c, "VIF": float(v)})
    return pd.DataFrame(rows)


def likelihood_ratio_test(res_full, res_reduced) -> dict:
    """LRT: −2 * (logLik_reduced − logLik_full) ~ chi²(df_diff)"""
    ll_full = float(res_full.llf)
    ll_red = float(res_reduced.llf)
    df_diff = int(res_full.df_model - res_reduced.df_model)
    lr = 2.0 * (ll_full - ll_red)
    p = float(stats.chi2.sf(lr, df_diff)) if df_diff > 0 else np.nan
    return {"LR_statistic": lr, "df_diff": df_diff, "p_value": p,
            "ll_M1_reduced": ll_red, "ll_M2_full": ll_full,
            "aic_M1": float(res_reduced.aic), "aic_M2": float(res_full.aic),
            "bic_M1": float(res_reduced.bic), "bic_M2": float(res_full.bic)}


def coef_rows(res, model_label: str) -> list[dict]:
    rows = []
    p = res.params
    se = res.bse
    pv = res.pvalues
    for term in p.index:
        rows.append({
            "model": model_label,
            "term": term,
            "coef": float(p[term]),
            "std_err": float(se[term]),
            "p_value": float(pv[term]),
        })
    return rows


def main():
    cell = build_cell_df(MODELS_RQ2)
    print("=== Cell counts ===")
    print(cell.groupby(["model", "variant"]).size().unstack(fill_value=0))

    # (a) 配對 t + Cohen's dz
    pt = run_paired_t(cell)
    print("\n=== (a) 配對 t + Cohen's dz ===")
    print(pt.to_string(index=False))

    # (b) Cell-level OLS
    cell_oh = add_overhead_cols(cell)
    # 把 needle_id 變類別字串、把 model 變類別字串
    cell_oh["needle_id"] = cell_oh["needle_id"].astype(str)
    cell_oh["model"] = cell_oh["model"].astype(str)
    cell_oh["variant"] = cell_oh["variant"].astype(str)

    # VIF：對連續變項做共線性檢查
    print("\n=== VIF: length_chars vs length_tokens（驗證共線性） ===")
    vif_raw = vif_table(cell_oh, ["length_chars", "length_tokens"])
    print(vif_raw.to_string(index=False))
    print("\n=== VIF: log10(length_chars) + token_overhead（模型內變項） ===")
    vif_model = vif_table(cell_oh, ["log10_length_chars", "token_overhead"])
    print(vif_model.to_string(index=False))

    # M1: 不含 token_overhead
    m1_formula = (
        "prop_correct ~ log10_length_chars + C(variant) + C(needle_id) + C(model)"
    )
    # M2: 加 token_overhead
    m2_formula = (
        "prop_correct ~ log10_length_chars + token_overhead "
        "+ C(variant) + C(needle_id) + C(model)"
    )
    res_m1 = smf.ols(m1_formula, data=cell_oh).fit()
    res_m2 = smf.ols(m2_formula, data=cell_oh).fit()

    print(f"\n=== M1 R²={res_m1.rsquared:.4f}, n={int(res_m1.nobs)} ===")
    print(f"=== M2 R²={res_m2.rsquared:.4f}, n={int(res_m2.nobs)} ===")

    lrt = likelihood_ratio_test(res_m2, res_m1)
    print("\n=== LRT (M2 vs M1) ===")
    print(lrt)

    m2_coef = coef_rows(res_m2, "M2_full")
    m1_coef = coef_rows(res_m1, "M1_no_overhead")

    coef_df = pd.DataFrame(m1_coef + m2_coef)
    print("\n=== OLS coefficients ===")
    print(coef_df.to_string(index=False))

    # 寫 CSV
    out_path = "rq2_results.csv"
    with open(out_path, "w") as f:
        f.write("# RQ2 (a) Paired t + Cohen's dz (traditional vs simplified_q)\n")
        pt.to_csv(f, index=False)
        f.write("\n# RQ2 (b) VIF: length_chars vs length_tokens (raw)\n")
        vif_raw.to_csv(f, index=False)
        f.write("\n# RQ2 (b) VIF: log10_length_chars + token_overhead (model variables)\n")
        vif_model.to_csv(f, index=False)
        f.write("\n# RQ2 (b) OLS coefficients\n")
        coef_df.to_csv(f, index=False)
        f.write("\n# RQ2 (b) Model summary\n")
        pd.DataFrame([
            {"model": "M1_no_overhead", "r_squared": res_m1.rsquared,
             "adj_r_squared": res_m1.rsquared_adj, "n": int(res_m1.nobs),
             "df_residual": int(res_m1.df_resid),
             "ll": float(res_m1.llf), "aic": float(res_m1.aic),
             "bic": float(res_m1.bic)},
            {"model": "M2_full", "r_squared": res_m2.rsquared,
             "adj_r_squared": res_m2.rsquared_adj, "n": int(res_m2.nobs),
             "df_residual": int(res_m2.df_resid),
             "ll": float(res_m2.llf), "aic": float(res_m2.aic),
             "bic": float(res_m2.bic)},
        ]).to_csv(f, index=False)
        f.write("\n# RQ2 (b) LRT (M2 vs M1)\n")
        pd.DataFrame([lrt]).to_csv(f, index=False)
        # 額外：fertility 表
        f.write("\n# RQ2 (b) fertility by model (tokens/chars)\n")
        fert = fertility_by_model(cell)
        pd.DataFrame([{"model": k, "fertility": v} for k, v in fert.items()])\
            .to_csv(f, index=False)

    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
