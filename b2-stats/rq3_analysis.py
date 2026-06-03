"""
RQ3：繁體 vs 簡體的衰減提前量。

對每個退化型模型計算：
  sp_trad        = 衰減起始點（字元）— traditional 變項
  sp_simp_q      = 衰減起始點（字元）— simplified_q 變項
  advance_chars  = sp_simp_q − sp_trad（正值代表繁體較早衰減）
  advance_tokens = advance_chars × fertility_family（家族層 token/char）

報告每個 family 的平均 advance_chars 與 advance_tokens。

輸出：rq3_results.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from compute_model_metrics import (
    DEGRADING_MODELS, MODEL_META,
    load_results, load_results_h2, to_dataframe,
    compute_length_accuracy, compute_starting_point,
)


def family_fertility(models_in_family: list[str], variant: str = "traditional") -> float:
    """家族 fertility = 家族內模型在所有 (length, position, needle) 之 mean(tokens / chars)。

    使用 traditional（繁體）變項，與「字元 → token」換算的物理含義一致。"""
    parts = []
    for m in models_in_family:
        df = to_dataframe(load_results(m) + load_results_h2(m))
        df = df[df["variant"] == variant].copy()
        df["fert"] = df["tokens_prompt"] / df["length_chars"]
        parts.append(df["fert"])
    s = pd.concat(parts, ignore_index=True)
    return float(s.mean())


def model_sp(model: str, variant: str, drop_pp: float = 5.0,
             truncate_27b: bool = True) -> dict:
    df = to_dataframe(load_results(model) + load_results_h2(model))
    max_length = 65000 if (truncate_27b and model == "gemma3:27b") else None
    g = compute_length_accuracy(df, variant, max_length=max_length)
    return compute_starting_point(g, drop_pp=drop_pp)


def main():
    drop_pp = 5.0
    # 1) 家族 fertility
    families = {}
    for m in DEGRADING_MODELS:
        fam = MODEL_META[m]["family"]
        families.setdefault(fam, []).append(m)
    fert_family = {fam: family_fertility(ms) for fam, ms in families.items()}
    print("=== Family fertility (traditional) ===")
    print(fert_family)

    # 2) 每個模型的 sp_trad / sp_simp_q / advance
    rows = []
    for m in DEGRADING_MODELS:
        meta = MODEL_META[m]
        fam = meta["family"]
        sp_t = model_sp(m, "traditional", drop_pp=drop_pp)
        sp_q = model_sp(m, "simplified_q", drop_pp=drop_pp)
        adv_chars = float(sp_q["sp_chars"]) - float(sp_t["sp_chars"])
        adv_tokens = adv_chars * fert_family[fam]
        rows.append({
            "model": m,
            "family": fam,
            "scale_B": meta["scale_B"],
            "sp_trad_chars": sp_t["sp_chars"],
            "sp_simp_q_chars": sp_q["sp_chars"],
            "advance_chars": adv_chars,
            "fertility_family": fert_family[fam],
            "advance_tokens": adv_tokens,
            "trad_censored": sp_t["censored"],
            "simp_q_censored": sp_q["censored"],
            "trad_baseline_pct": sp_t["baseline_pct"],
            "simp_q_baseline_pct": sp_q["baseline_pct"],
            "drop_pp": drop_pp,
        })
    per_model = pd.DataFrame(rows)
    print("\n=== Per-model advance ===")
    print(per_model.to_string(index=False))

    # 3) 每個 family 的平均 advance
    family_summary = per_model.groupby("family", as_index=False).agg(
        n_models=("model", "size"),
        mean_advance_chars=("advance_chars", "mean"),
        sd_advance_chars=("advance_chars", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        mean_advance_tokens=("advance_tokens", "mean"),
        sd_advance_tokens=("advance_tokens", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        fertility_family=("fertility_family", "first"),
    )
    print("\n=== Family-level advance summary ===")
    print(family_summary.to_string(index=False))

    # 寫 CSV
    out_path = "rq3_results.csv"
    with open(out_path, "w") as f:
        f.write("# RQ3 per-model advance (chars / tokens)\n")
        per_model.to_csv(f, index=False)
        f.write("\n# RQ3 family-level mean advance\n")
        family_summary.to_csv(f, index=False)
        f.write("\n# RQ3 family fertility (traditional, tokens per char)\n")
        pd.DataFrame([{"family": k, "fertility_family": v}
                      for k, v in fert_family.items()]).to_csv(f, index=False)

    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
