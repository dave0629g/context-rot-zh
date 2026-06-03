"""
檢查每個 *_results.jsonl 是否「重存 reeval 後仍可用」。

對每個 (model, variant)：
  1. 從 raw JSONL（main + h2_）載入並套用最新 reevaluate()。
  2. 計算 accuracy_by_length。
  3. 與 results/analysis/{model}_analysis.json 的 accuracy_by_length 比對。

判定：
  - 完全一致（誤差 < 0.005）→ OK：raw 欄位 stale，但 reeval 後可用。
  - 不一致（誤差 >= 0.005）→ 不可用：raw 內容與既有 analysis 對不上，可能是
    舊版實驗（pre-rulebase-fix）未重存、或實驗版本錯位。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from compute_model_metrics import (
    load_results, load_results_h2, to_dataframe, REPO_RESULTS,
)


def load_analysis_json(model: str) -> dict | None:
    p = REPO_RESULTS / "analysis" / f"{model}_analysis.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_models_with_results() -> list[str]:
    out = []
    for p in sorted(REPO_RESULTS.glob("*_results.jsonl")):
        name = p.name.replace("_results.jsonl", "")
        if name.startswith("h2_"):
            continue
        out.append(name)
    return out


def check_model(model: str) -> list[dict]:
    rows = []
    raw = load_results(model)
    h2 = load_results_h2(model)
    df = to_dataframe(raw + h2)
    analysis = load_analysis_json(model)
    if analysis is None:
        return [{"model": model, "variant": "(all)", "status": "missing_analysis_json",
                 "note": "找不到對應 analysis JSON"}]
    abl_json = analysis.get("accuracy_by_length", {})
    for variant in sorted(df["variant"].unique()):
        sub = df[df["variant"] == variant]
        if len(sub) == 0:
            continue
        g = sub.groupby("length_chars").agg(prop=("is_correct", "mean")).reset_index()
        ours = {int(r["length_chars"]): float(r["prop"]) for _, r in g.iterrows()}
        theirs = {int(k): float(v) for k, v in abl_json.get(variant, {}).items()}
        all_lens = sorted(set(ours) | set(theirs))
        diffs = []
        for L in all_lens:
            a = ours.get(L); b = theirs.get(L)
            if a is None or b is None:
                diffs.append((L, a, b, np.nan))
            else:
                diffs.append((L, a, b, a - b))
        max_abs = max((abs(d[3]) for d in diffs if d[3] is not None and not np.isnan(d[3])),
                      default=np.nan)
        status = "OK" if (not np.isnan(max_abs) and max_abs < 0.005) else "MISMATCH"
        rows.append({
            "model": model,
            "variant": variant,
            "n_lengths_ours": len(ours),
            "n_lengths_json": len(theirs),
            "max_abs_diff": max_abs,
            "status": status,
            "detail_lengths_mismatch": "; ".join(
                f"{L}:ours={a:.3f},json={b:.3f}" if (a is not None and b is not None) else f"{L}:miss"
                for (L, a, b, d) in diffs
                if (d is None or np.isnan(d) or abs(d) >= 0.005)
            ),
        })
    return rows


def check_raw_vs_reeval(model: str) -> dict:
    """Raw evaluation.is_correct vs reeval 後 is_correct 的差異規模。"""
    raw = load_results(model) + load_results_h2(model)
    # 不重算
    df_raw = to_dataframe(raw, reeval=False)
    # 重算
    raw2 = json.loads(json.dumps(raw))  # 深拷貝，避免汙染
    df_re = to_dataframe(raw2, reeval=True)
    df = df_raw.copy()
    df["is_correct_reeval"] = df_re["is_correct"].to_numpy()
    df["changed"] = (df["is_correct"] != df["is_correct_reeval"]).astype(int)
    n = len(df)
    total_changed = int(df["changed"].sum())
    overall_raw = float(df["is_correct"].mean())
    overall_re = float(df["is_correct_reeval"].mean())
    return {"model": model, "n_total": n,
            "n_changed_by_reeval": total_changed,
            "pct_changed": round(100 * total_changed / max(n, 1), 2),
            "raw_overall_acc": round(overall_raw, 4),
            "reeval_overall_acc": round(overall_re, 4),
            "delta_acc": round(overall_re - overall_raw, 4)}


def main():
    models = list_models_with_results()
    print(f"檢查 {len(models)} 個模型 result 檔案：")
    all_rows = []
    for m in models:
        try:
            all_rows.extend(check_model(m))
        except Exception as e:
            all_rows.append({"model": m, "variant": "(error)", "status": "error",
                             "note": str(e)})
    cons = pd.DataFrame(all_rows)
    print("\n=== 一致性檢查（reeval 後 vs analysis JSON）===")
    print(cons.to_string(index=False))

    # raw vs reeval 規模
    delta_rows = []
    for m in models:
        try:
            delta_rows.append(check_raw_vs_reeval(m))
        except Exception as e:
            delta_rows.append({"model": m, "n_total": 0,
                               "n_changed_by_reeval": -1, "pct_changed": -1,
                               "raw_overall_acc": np.nan, "reeval_overall_acc": np.nan,
                               "delta_acc": np.nan})
    delta = pd.DataFrame(delta_rows)
    print("\n=== Raw is_correct vs reeval 後差異 ===")
    print(delta.to_string(index=False))

    # 輸出
    cons.to_csv("results_consistency.csv", index=False)
    delta.to_csv("results_raw_vs_reeval.csv", index=False)
    print("\n寫入 results_consistency.csv 與 results_raw_vs_reeval.csv")

    # 摘要
    bad = cons[cons["status"] == "MISMATCH"]
    print("\n=== 不可用（reeval 後仍對不上 analysis JSON）的 (model, variant) ===")
    if len(bad) == 0:
        print("（無）— 所有 raw JSONL 透過 reevaluate() 重算後皆與 analysis JSON 一致。")
    else:
        print(bad[["model", "variant", "max_abs_diff", "detail_lengths_mismatch"]].to_string(index=False))


if __name__ == "__main__":
    main()
