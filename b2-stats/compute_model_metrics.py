"""
從 raw JSONL 計算每個 (model, variant) 的：
- accuracy_by_length: dict[length_chars] = (prop_correct, avg_tokens)
- starting_point_chars: 首次降 >= drop_pp 的長度（字元）
- starting_point_tokens: 對應的平均 token 數
- decay_rate_pp_per_1k_tokens: 對 token 數做線性迴歸
- fertility: avg_tokens / chars 的全長平均
"""
from __future__ import annotations
import json
import sys
import importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


# 自動偵測 repo 根，支援兩種佈局：
#   (a) 本機：b2-stats/ 旁掛 context-rot-zh/ 子目錄
#   (b) 置於 repo 內 b2-stats/：repo 根為上一層
_here = Path(__file__).resolve().parent
if (_here / "context-rot-zh" / "scripts" / "04_analyze.py").exists():
    REPO_ROOT = _here / "context-rot-zh"
else:
    REPO_ROOT = _here.parent
REPO_RESULTS = REPO_ROOT / "results"

# 載入既有 04_analyze.py 的 reevaluate 函式，使 is_correct 與 analysis JSON 一致
_spec = importlib.util.spec_from_file_location("_analyze",
                                                REPO_ROOT / "scripts" / "04_analyze.py")
_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyze)
reevaluate = _analyze.reevaluate

# 所有可能模型；後續按 RQ 篩
ALL_MODELS = [
    "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b",
    "gemma4:e2b", "gemma4:e4b", "gemma4:26b", "gemma4:31b",
    "llama3.1:8b", "llama3.1:70b", "llama3.3:70b",
    "qwen3:8b",
    "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b", "qwen3.5:35b",
]

# 退化型模型（完整集合，6 個）
DEGRADING_MODELS = [
    "gemma3:4b", "gemma3:12b", "gemma3:27b",
    "llama3.1:8b", "llama3.1:70b", "llama3.3:70b",
]

# RQ1 / RQ3 之 family×scale 分析樣本（n=5）。
# 排除 llama3.1:70b：其與 llama3.3:70b 同屬 LLaMA 家族、同為 70B 參數規模，
# 於 (family, scale_B) 設計矩陣中佔據完全相同之 (LLaMA, 70.0) 格位；且二者衰減
# 起始點於本實驗長度範圍內均為右截尾上限值（log10 sp_tokens 相同），對 DV1 為
# 重複觀測。為避免單一 (家族×規模) 組合在僅 5 點之迴歸中被加倍加權、扭曲 Unique R²
# 之分配，保留較新世代之 llama3.3:70b、排除 llama3.1:70b（與 gemma3:1b 之排除並列）。
ANALYSIS_MODELS = [m for m in DEGRADING_MODELS if m != "llama3.1:70b"]

# 模型家族與參數規模（B）
MODEL_META = {
    "gemma3:1b":   {"family": "Gemma", "scale_B": 1.0},
    "gemma3:4b":   {"family": "Gemma", "scale_B": 4.0},
    "gemma3:12b":  {"family": "Gemma", "scale_B": 12.0},
    "gemma3:27b":  {"family": "Gemma", "scale_B": 27.0},
    "gemma4:e2b":  {"family": "Gemma", "scale_B": 2.0},
    "gemma4:e4b":  {"family": "Gemma", "scale_B": 4.0},
    "gemma4:26b":  {"family": "Gemma", "scale_B": 26.0},
    "gemma4:31b":  {"family": "Gemma", "scale_B": 31.0},
    "llama3.1:8b": {"family": "LLaMA", "scale_B": 8.0},
    "llama3.1:70b":{"family": "LLaMA", "scale_B": 70.0},
    "llama3.3:70b":{"family": "LLaMA", "scale_B": 70.0},
    "qwen3:8b":    {"family": "Qwen",  "scale_B": 8.0},
    "qwen3.5:2b":  {"family": "Qwen",  "scale_B": 2.0},
    "qwen3.5:4b":  {"family": "Qwen",  "scale_B": 4.0},
    "qwen3.5:9b":  {"family": "Qwen",  "scale_B": 9.0},
    "qwen3.5:27b": {"family": "Qwen",  "scale_B": 27.0},
    "qwen3.5:35b": {"family": "Qwen",  "scale_B": 35.0},
}


def load_results(model: str) -> list[dict]:
    fn = REPO_RESULTS / f"{model}_results.jsonl"
    if not fn.exists():
        raise FileNotFoundError(fn)
    rows = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_results_h2(model: str) -> list[dict]:
    """簡體問題（simplified_q）變項的補充結果。"""
    fn = REPO_RESULTS / f"h2_{model}_results.jsonl"
    if not fn.exists():
        return []
    rows = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_dataframe(rows: list[dict], reeval: bool = True) -> pd.DataFrame:
    """攤平成扁平 DataFrame。

    reeval=True 時用 04_analyze.reevaluate 重算 is_correct，
    保持與既有 analysis JSON 的口徑一致（修正繁簡比對偏差）。
    """
    if reeval:
        for r in rows:
            if r.get("skipped"):
                continue
            r.setdefault("evaluation", {})
            r["evaluation"]["is_correct"] = bool(reevaluate(r))
    flat = []
    for r in rows:
        eval_ = r.get("evaluation") or {}
        flat.append({
            "experiment_id": r.get("experiment_id"),
            "model": r.get("model"),
            "variant": r.get("variant"),
            "length_chars": r.get("context_length_chars"),
            "position": r.get("needle_position"),
            "trial": r.get("trial"),
            "needle_id": r.get("needle_id"),
            "tokens_prompt": r.get("token_count_prompt"),
            "is_correct": int(bool(eval_.get("is_correct", False))),
            "skipped": bool(r.get("skipped", False)),
        })
    df = pd.DataFrame(flat)
    # 篩掉 skipped
    df = df[~df["skipped"]].copy()
    return df


def compute_length_accuracy(df: pd.DataFrame, variant: str,
                            max_length: int | None = None) -> pd.DataFrame:
    """回傳 length_chars, prop_correct, avg_tokens"""
    sub = df[df["variant"] == variant].copy()
    if max_length is not None:
        sub = sub[sub["length_chars"] <= max_length]
    g = sub.groupby("length_chars", as_index=False).agg(
        prop_correct=("is_correct", "mean"),
        avg_tokens=("tokens_prompt", "mean"),
        n=("is_correct", "size"),
    )
    return g.sort_values("length_chars").reset_index(drop=True)


def compute_starting_point(g: pd.DataFrame, drop_pp: float = 5.0,
                           baseline_lengths: tuple[int, ...] = (500, 2000, 4000, 6000, 8000)
                           ) -> dict:
    """衰減起始點：首次降 >= drop_pp 的長度。

    baseline 為 baseline_lengths 中存在點的平均 prop。
    若全範圍均未觸發，回傳 sp_chars=NaN, sp_tokens=NaN, censored=True。
    """
    if len(g) < 3:
        return {"sp_chars": np.nan, "sp_tokens": np.nan, "censored": True,
                "baseline_pct": np.nan, "drop_pp": drop_pp}
    base_pts = g[g["length_chars"].isin(baseline_lengths)]
    if len(base_pts) == 0:
        baseline = g["prop_correct"].iloc[:3].mean()
    else:
        baseline = base_pts["prop_correct"].mean()
    baseline_pct = baseline * 100.0
    threshold = baseline_pct - drop_pp
    sp_chars = np.nan
    sp_tokens = np.nan
    for _, row in g.iterrows():
        pct = row["prop_correct"] * 100.0
        if pct < threshold:
            sp_chars = float(row["length_chars"])
            sp_tokens = float(row["avg_tokens"])
            break
    censored = bool(np.isnan(sp_chars))
    if censored:
        # 右截尾：在實驗範圍內未觸發；以實驗最大長度作為下限保守估計
        last = g.iloc[-1]
        sp_chars = float(last["length_chars"])
        sp_tokens = float(last["avg_tokens"])
    return {"sp_chars": sp_chars, "sp_tokens": sp_tokens,
            "censored": censored, "baseline_pct": baseline_pct,
            "drop_pp": drop_pp}


def compute_decay_rate(g: pd.DataFrame) -> dict:
    """以 token 為 X 跑線性迴歸：
    Y (pp) = a + b * (tokens / 1000)
    decay_rate = b（pp per 1000 tokens；負值代表衰減）。
    """
    if len(g) < 3:
        return {"slope_pp_per_1k_tokens": np.nan, "r_squared": np.nan,
                "n_points": len(g)}
    xs = (g["avg_tokens"].to_numpy() / 1000.0)
    ys = g["prop_correct"].to_numpy() * 100.0
    n = len(xs)
    sx = xs.sum(); sy = ys.sum()
    sxx = (xs * xs).sum()
    sxy = (xs * ys).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"slope_pp_per_1k_tokens": np.nan, "r_squared": np.nan,
                "n_points": n}
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = slope * xs + intercept
    ss_res = ((ys - y_pred) ** 2).sum()
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope_pp_per_1k_tokens": float(slope),
            "r_squared": float(r2),
            "n_points": int(n)}


def family_fertility(df: pd.DataFrame, variant: str = "traditional") -> float:
    """平均 tokens / chars。"""
    sub = df[df["variant"] == variant].copy()
    sub = sub[(sub["length_chars"] > 0) & (sub["tokens_prompt"] > 0)]
    return float((sub["tokens_prompt"] / sub["length_chars"]).mean())


def build_metrics_table(models: list[str], variant: str = "traditional",
                         drop_pp: float = 5.0,
                         truncate_27b: bool = True) -> pd.DataFrame:
    rows = []
    for m in models:
        raw = load_results(m) + load_results_h2(m)
        df = to_dataframe(raw)
        max_length = 65000 if (truncate_27b and m == "gemma3:27b") else None
        g = compute_length_accuracy(df, variant, max_length=max_length)
        sp = compute_starting_point(g, drop_pp=drop_pp)
        dr = compute_decay_rate(g)
        meta = MODEL_META[m]
        rows.append({
            "model": m,
            "tokenizer_family": meta["family"],
            "scale_B": meta["scale_B"],
            "variant": variant,
            "starting_point_chars": sp["sp_chars"],
            "starting_point_tokens": sp["sp_tokens"],
            "decay_rate_pp_per_1k_tokens": dr["slope_pp_per_1k_tokens"],
            "decay_r_squared": dr["r_squared"],
            "n_length_points": dr["n_points"],
            "baseline_pct": sp["baseline_pct"],
            "drop_pp": sp["drop_pp"],
            "censored": sp["censored"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 分析樣本 traditional 5pp（n=5；排除 llama3.1:70b，見 ANALYSIS_MODELS 註解）
    tab = build_metrics_table(ANALYSIS_MODELS, variant="traditional", drop_pp=5.0)
    print(tab.to_string(index=False))
