"""
繁體中文 Context Rot 實驗 — 互動式結果瀏覽器

執行：
  streamlit run app.py

部署：
  Streamlit Community Cloud → 連結此 GitHub repo → 指定 app.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from importlib import import_module
_analyze = import_module("04_analyze")
reevaluate = _analyze.reevaluate
compute_rot_coefficient = _analyze.compute_rot_coefficient
compute_breakpoint = _analyze.compute_breakpoint
compute_long_context_accuracy = _analyze.compute_long_context_accuracy

# ── 常數 ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_META = {
    "gemma3:1b":   dict(label="Gemma 3 1B",   family="Gemma 3",      color="#A8C0FF"),
    "gemma3:4b":   dict(label="Gemma 3 4B",   family="Gemma 3",      color="#6690E0"),
    "gemma3:12b":  dict(label="Gemma 3 12B",  family="Gemma 3",      color="#3A62B8"),
    "gemma3:27b":  dict(label="Gemma 3 27B",  family="Gemma 3",      color="#1A3A8C"),
    "gemma4:e2b":  dict(label="Gemma 4 E2B",  family="Gemma 4 Edge", color="#FFB060"),
    "gemma4:e4b":  dict(label="Gemma 4 E4B",  family="Gemma 4 Edge", color="#D07010"),
    "gemma4:26b":  dict(label="Gemma 4 26B",  family="Gemma 4",      color="#E06040"),
    "gemma4:31b":  dict(label="Gemma 4 31B",  family="Gemma 4",      color="#8B2500"),
    "llama3.1:8b": dict(label="Llama 3.1 8B", family="Llama",        color="#E07070"),
    "llama3.3:70b":dict(label="Llama 3.3 70B",family="Llama",        color="#8B0000"),
    "qwen3:8b":    dict(label="Qwen3 8B",     family="Qwen3",        color="#2E8B4A"),
    "qwen3.5:2b":  dict(label="Qwen3.5 2B",  family="Qwen3.5",       color="#C09AE8"),
    "qwen3.5:4b":  dict(label="Qwen3.5 4B",  family="Qwen3.5",       color="#A070D0"),
    "qwen3.5:9b":  dict(label="Qwen3.5 9B",  family="Qwen3.5",       color="#7840B8"),
    "qwen3.5:27b": dict(label="Qwen3.5 27B", family="Qwen3.5",       color="#5010A0"),
    "qwen3.5:35b": dict(label="Qwen3.5 35B", family="Qwen3.5",       color="#2E0060"),
}
FAMILIES = {
    "Gemma 3":      ["gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"],
    "Gemma 4 Edge": ["gemma4:e2b", "gemma4:e4b"],
    "Gemma 4":      ["gemma4:26b", "gemma4:31b"],
    "Llama":        ["llama3.1:8b", "llama3.3:70b"],
    "Qwen3":        ["qwen3:8b"],
    "Qwen3.5":      ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b", "qwen3.5:35b"],
}
ALL_VARIANTS = ["繁問繁答", "繁問簡答", "簡問簡答"]
VARIANT_DASH  = {"繁問繁答": "solid", "繁問簡答": "dash", "簡問簡答": "dot"}
NEEDLE_LABELS = {"N01": "N01 金額", "N02": "N02 人名", "N03": "N03 面積",
                 "N04": "N04 數量", "N05": "N05 百分比"}
DEFAULT_MODELS = {"gemma3:4b", "llama3.1:8b", "gemma4:e2b"}


# ── 資料載入 ──────────────────────────────────────────────────────────────────

def _jsonl_max_mtime() -> float:
    """回傳所有 JSONL 結果檔的最新修改時間，作為快取失效 key。"""
    mtimes = [f.stat().st_mtime for f in RESULTS_DIR.glob("*_results.jsonl")]
    return max(mtimes) if mtimes else 0.0


@st.cache_data(show_spinner="載入並評估資料（首次約需 10–20 秒）...")
def load_all_data(mtime_key: float = 0.0) -> dict:
    """回傳 {model: {variant_label: [records]}}
    mtime_key 由呼叫端傳入，確保 JSONL 更新後快取自動失效。
    """
    result = {}
    variant_source = [
        ("traditional",  lambda m: RESULTS_DIR / f"{m}_results.jsonl",    "繁問繁答"),
        ("simplified",   lambda m: RESULTS_DIR / f"{m}_results.jsonl",    "繁問簡答"),
        ("simplified_q", lambda m: RESULTS_DIR / f"h2_{m}_results.jsonl", "簡問簡答"),
    ]

    all_models = set()
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_results.jsonl") and not fname.startswith("h2_"):
            model = fname.replace("_results.jsonl", "")
            if (RESULTS_DIR / fname).stat().st_size > 0:
                all_models.add(model)

    for model in sorted(all_models):
        model_data = defaultdict(list)
        for vkey, path_fn, label in variant_source:
            fpath = path_fn(model)
            if not fpath.exists() or fpath.stat().st_size == 0:
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("skipped") or r.get("variant") != vkey:
                            continue
                        r["_correct"] = reevaluate(r)
                        model_data[label].append(r)
                    except Exception:
                        pass
        if model_data:
            result[model] = dict(model_data)
    return result


def acc_by(records: list, key_fn) -> dict:
    d = defaultdict(lambda: [0, 0])
    for r in records:
        k = key_fn(r)
        d[k][1] += 1
        d[k][0] += int(r["_correct"])
    return {k: c / t * 100 for k, (c, t) in d.items() if t > 0}


def fmt_length(n: int) -> str:
    return f"{n // 1000}K" if n >= 1000 else str(n)


# ── 圖表函式 ──────────────────────────────────────────────────────────────────

def chart_length(all_data, models, variants) -> go.Figure:
    fig = go.Figure()
    all_vals = []
    for model in models:
        for variant in variants:
            recs = all_data.get(model, {}).get(variant, [])
            if not recs:
                continue
            acc = acc_by(recs, lambda r: r["context_length_chars"])
            lengths = sorted(acc)
            all_vals.extend(acc.values())
            fig.add_trace(go.Scatter(
                x=[fmt_length(l) for l in lengths],
                y=[acc[l] for l in lengths],
                mode="lines+markers",
                name=f"{MODEL_META.get(model, {}).get('label', model)} {variant}",
                line=dict(color=MODEL_META.get(model, {}).get("color", "#888"),
                          dash=VARIANT_DASH.get(variant, "solid"), width=2),
                marker=dict(size=7),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}字元<br>準確率: %{y:.1f}%<extra></extra>",
            ))
    y_min = max(0, min(all_vals) - 5) if all_vals else 60
    fig.update_layout(
        title="準確率 vs Context 長度",
        xaxis_title="Context 長度（字元）",
        yaxis_title="準確率（%）",
        yaxis_range=[y_min, 102],
        hovermode="x unified",
        height=500,
        legend=dict(x=1.02, y=1),
        margin=dict(r=220),
    )
    return fig


def chart_position(all_data, models, variants) -> go.Figure:
    fig = go.Figure()
    for model in models:
        for variant in variants:
            recs = all_data.get(model, {}).get(variant, [])
            if not recs:
                continue
            acc = acc_by(recs, lambda r: r["needle_position"])
            positions = sorted(acc)
            fig.add_trace(go.Scatter(
                x=[p * 100 for p in positions],
                y=[acc[p] for p in positions],
                mode="lines+markers",
                name=f"{MODEL_META.get(model, {}).get('label', model)} {variant}",
                line=dict(color=MODEL_META.get(model, {}).get("color", "#888"),
                          dash=VARIANT_DASH.get(variant, "solid"), width=2),
                marker=dict(size=7),
                hovertemplate="<b>%{fullData.name}</b><br>位置: %{x:.0f}%<br>準確率: %{y:.1f}%<extra></extra>",
            ))
    fig.update_layout(
        title="準確率 vs Needle 位置",
        xaxis_title="Needle 位置（%）",
        yaxis_title="準確率（%）",
        xaxis=dict(tickmode="linear", tick0=0, dtick=10),
        hovermode="x unified",
        height=500,
        legend=dict(x=1.02, y=1),
        margin=dict(r=220),
    )
    return fig


def chart_heatmap(all_data, model, variant) -> go.Figure | None:
    recs = all_data.get(model, {}).get(variant, [])
    if not recs:
        return None
    d = defaultdict(lambda: [0, 0])
    for r in recs:
        k = (r["context_length_chars"], r["needle_position"])
        d[k][1] += 1
        d[k][0] += int(r["_correct"])
    lengths = sorted({k[0] for k in d})
    positions = sorted({k[1] for k in d})
    matrix = [
        [round(d[(l, p)][0] / d[(l, p)][1] * 100, 1) if (l, p) in d else None
         for p in positions]
        for l in lengths
    ]
    text = [[f"{v:.0f}" if v is not None else "" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, text=text, texttemplate="%{text}",
        x=[f"{int(p * 100)}%" for p in positions],
        y=[fmt_length(l) for l in lengths],
        colorscale="RdYlGn", zmin=0, zmax=100,
        hovertemplate="長度: %{y}<br>位置: %{x}<br>準確率: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"NIAH Heatmap：{MODEL_META.get(model, {}).get('label', model)} — {variant}",
        xaxis_title="Needle 位置",
        yaxis_title="Context 長度",
        height=500,
    )
    return fig


def chart_needle(all_data, models, variants) -> go.Figure:
    needles = sorted(NEEDLE_LABELS)
    fig = go.Figure()
    for model in models:
        for variant in variants:
            recs = all_data.get(model, {}).get(variant, [])
            if not recs:
                continue
            acc = acc_by(recs, lambda r: r["needle_id"])
            fig.add_trace(go.Bar(
                x=[NEEDLE_LABELS[n] for n in needles],
                y=[acc.get(n, 0) for n in needles],
                name=f"{MODEL_META.get(model, {}).get('label', model)} {variant}",
                marker_color=MODEL_META.get(model, {}).get("color", "#888"),
                opacity=0.85,
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1f}%<extra></extra>",
            ))
    fig.update_layout(
        title="各 Needle 準確率", barmode="group",
        yaxis_title="準確率（%）", yaxis_range=[0, 105],
        height=450,
    )
    return fig


def chart_token_overhead(all_data, models) -> go.Figure | None:
    labels, values, colors = [], [], []
    for model in models:
        trad = {r["experiment_id"]: r.get("token_count_prompt", 0)
                for r in all_data.get(model, {}).get("繁問繁答", [])}
        simp = {r["experiment_id"]: r.get("token_count_prompt", 0)
                for r in all_data.get(model, {}).get("繁問簡答", [])}
        pairs = [(trad[e], simp[e]) for e in set(trad) & set(simp)
                 if trad[e] > 0 and simp[e] > 0]
        if not pairs:
            continue
        overhead = (sum(t / s for t, s in pairs) / len(pairs) - 1) * 100
        labels.append(MODEL_META.get(model, {}).get("label", model))
        values.append(round(overhead, 2))
        colors.append(MODEL_META.get(model, {}).get("color", "#888"))
    if not labels:
        return None
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"+{v:.1f}%" for v in values], textposition="outside",
        hovertemplate="<b>%{x}</b><br>繁體多用 %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Tokenizer Overhead：繁體比簡體多用的 Token（%）",
        yaxis_title="多用比例（%）",
        yaxis_range=[0, max(values) * 1.5 + 1],
        height=400,
    )
    return fig


LABEL_TO_VARIANT = {"繁問繁答": "traditional", "繁問簡答": "simplified", "簡問簡答": "simplified_q"}
VARIANT_TO_LABEL = {v: k for k, v in LABEL_TO_VARIANT.items()}


def _to_flat_records(all_data: dict, model: str) -> list:
    """將 Streamlit 格式的資料轉回 04_analyze 格式"""
    flat = []
    for label, recs in all_data.get(model, {}).items():
        vkey = LABEL_TO_VARIANT.get(label)
        if not vkey:
            continue
        for r in recs:
            r["variant"] = vkey
            r["evaluation"] = {"is_correct": r["_correct"]}
            flat.append(r)
    return flat


def chart_rot_coefficient(all_data, models) -> go.Figure:
    """衰退速率比較：各模型的 rot coefficient（斜率）"""
    rows = []
    for model in models:
        flat = _to_flat_records(all_data, model)
        if not flat:
            continue
        rot = compute_rot_coefficient(flat)
        label = MODEL_META.get(model, {}).get("label", model)
        for vkey, rc in rot.items():
            rows.append({
                "model": model, "label": label,
                "variant": VARIANT_TO_LABEL.get(vkey, vkey),
                "slope": rc["slope_pp_per_1k_chars"],
                "r_squared": rc["r_squared"],
            })

    if not rows:
        return None
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for variant in ALL_VARIANTS:
        vdf = df[df["variant"] == variant]
        if vdf.empty:
            continue
        fig.add_trace(go.Bar(
            x=vdf["label"], y=vdf["slope"],
            name=variant,
            text=[f"{s:.2f}" for s in vdf["slope"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b> %{fullData.name}<br>"
                "斜率: %{y:.3f} pp/1K字元<br>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="衰退速率比較（Rot Coefficient）",
        yaxis_title="斜率（pp / 1K 字元）",
        xaxis_title="",
        barmode="group",
        height=500,
        legend=dict(x=1.02, y=1),
        margin=dict(r=180),
    )
    # 加一條 y=0 的參考線
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def chart_breakpoint(all_data, models) -> go.Figure:
    """斷點分析：各模型準確率首次大幅下降的位置"""
    rows = []
    for model in models:
        flat = _to_flat_records(all_data, model)
        if not flat:
            continue
        bps = compute_breakpoint(flat)
        label = MODEL_META.get(model, {}).get("label", model)
        for vkey, bp in bps.items():
            vlabel = VARIANT_TO_LABEL.get(vkey, vkey)
            rows.append({
                "model": model, "label": label,
                "variant": vlabel,
                "breakpoint": bp["breakpoint_drop"],
                "total_drop": bp["total_drop_pp"],
                "acc_at_max": bp["acc_at_max_length"],
            })

    if not rows:
        return None
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for variant in ALL_VARIANTS:
        vdf = df[df["variant"] == variant]
        if vdf.empty:
            continue
        # 將 None/NaN 斷點顯示為 ">130K"（未觸發）
        bp_vals = []
        bp_text = []
        for _, r in vdf.iterrows():
            bp = r["breakpoint"]
            if bp is None or (isinstance(bp, float) and pd.isna(bp)):
                bp_vals.append(140000)  # 超出範圍表示未觸發
                bp_text.append(">130K")
            else:
                bp_vals.append(int(bp))
                bp_text.append(fmt_length(int(bp)))

        fig.add_trace(go.Bar(
            x=vdf["label"], y=bp_vals,
            name=variant,
            text=bp_text, textposition="outside",
            hovertemplate=(
                "<b>%{x}</b> %{fullData.name}<br>"
                "斷點: %{text}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="斷點分析：準確率首次下降 ≥10pp 的位置",
        yaxis_title="上下文長度（字元）",
        yaxis=dict(
            tickvals=[0, 8000, 16000, 32000, 65000, 100000, 130000, 140000],
            ticktext=["0", "8K", "16K", "32K", "65K", "100K", "130K", ">130K"],
        ),
        barmode="group",
        height=500,
        legend=dict(x=1.02, y=1),
        margin=dict(r=180),
    )
    return fig


def chart_sensitivity(all_data, models) -> go.Figure:
    """模型敏感度散佈圖：衰退斜率 vs 總落差"""
    rows = []
    for model in models:
        flat = _to_flat_records(all_data, model)
        if not flat:
            continue
        rot = compute_rot_coefficient(flat)
        bps = compute_breakpoint(flat)
        label = MODEL_META.get(model, {}).get("label", model)
        color = MODEL_META.get(model, {}).get("color", "#888")
        for vkey in rot:
            if vkey not in bps:
                continue
            vlabel = VARIANT_TO_LABEL.get(vkey, vkey)
            rows.append({
                "label": label, "variant": vlabel,
                "slope": rot[vkey]["slope_pp_per_1k_chars"],
                "total_drop": bps[vkey]["total_drop_pp"],
                "color": color,
            })

    if not rows:
        return None

    fig = go.Figure()
    markers = {"繁問繁答": "circle", "繁問簡答": "diamond", "簡問簡答": "square"}
    for variant in ALL_VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant]
        if not vrows:
            continue
        fig.add_trace(go.Scatter(
            x=[r["slope"] for r in vrows],
            y=[r["total_drop"] for r in vrows],
            mode="markers+text",
            marker=dict(
                size=14,
                symbol=markers.get(variant, "circle"),
                color=[r["color"] for r in vrows],
                line=dict(width=1, color="black"),
            ),
            text=[r["label"] for r in vrows],
            textposition="top center",
            textfont=dict(size=9),
            name=variant,
            hovertemplate=(
                "<b>%{text}</b> %{fullData.name}<br>"
                "斜率: %{x:.3f} pp/1K字元<br>"
                "總落差: %{y:.1f}pp<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="模型敏感度：衰退斜率 vs 總落差",
        xaxis_title="衰退斜率（pp / 1K 字元）— 越負越快衰退",
        yaxis_title="總落差（pp）— 基線 vs 最長上下文",
        height=550,
        legend=dict(x=1.02, y=1),
        margin=dict(r=180),
    )
    return fig


def chart_breakpoint_shift(all_data, models) -> go.Figure:
    """斷點偏移：繁體 vs 簡體的斷點位置差異"""
    rows = []
    for model in models:
        flat = _to_flat_records(all_data, model)
        if not flat:
            continue
        bps = compute_breakpoint(flat)
        label = MODEL_META.get(model, {}).get("label", model)
        color = MODEL_META.get(model, {}).get("color", "#888")

        bp_trad = bps.get("traditional", {}).get("breakpoint_drop")
        bp_simp = bps.get("simplified_q", bps.get("simplified", {})).get("breakpoint_drop")

        drop_trad = bps.get("traditional", {}).get("total_drop_pp", 0)
        drop_simp = bps.get("simplified_q", bps.get("simplified", {})).get("total_drop_pp", 0)

        if bp_trad is None and bp_simp is None:
            # 兩者都未觸發，跳過
            continue

        rows.append({
            "label": label, "color": color,
            "bp_trad": bp_trad, "bp_simp": bp_simp,
            "drop_trad": drop_trad, "drop_simp": drop_simp,
        })

    if not rows:
        return None

    fig = go.Figure()
    labels = [r["label"] for r in rows]
    max_val = 140000

    fig.add_trace(go.Bar(
        x=labels,
        y=[r["bp_trad"] if r["bp_trad"] else max_val for r in rows],
        name="繁問繁答",
        text=[fmt_length(r["bp_trad"]) if r["bp_trad"] else ">130K" for r in rows],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels,
        y=[r["bp_simp"] if r["bp_simp"] else max_val for r in rows],
        name="簡問簡答",
        text=[fmt_length(r["bp_simp"]) if r["bp_simp"] else ">130K" for r in rows],
        textposition="outside",
    ))

    fig.update_layout(
        title="斷點偏移：繁體 vs 簡體的退化起始位置",
        yaxis_title="上下文長度（字元）",
        yaxis=dict(
            tickvals=[0, 8000, 16000, 32000, 65000, 100000, 130000, 140000],
            ticktext=["0", "8K", "16K", "32K", "65K", "100K", "130K", ">130K"],
        ),
        barmode="group",
        height=500,
    )
    return fig


def length_table(all_data, models, variants) -> pd.DataFrame:
    rows = []
    for model in models:
        for variant in variants:
            recs = all_data.get(model, {}).get(variant, [])
            if not recs:
                continue
            acc = acc_by(recs, lambda r: r["context_length_chars"])
            for length, pct in sorted(acc.items()):
                rows.append({
                    "模型": MODEL_META.get(model, {}).get("label", model),
                    "Variant": variant,
                    "Context 長度": f"{length:,}",
                    "準確率": f"{pct:.1f}%",
                    "_sort_len": length,
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["模型", "Variant", "_sort_len"]).drop(columns="_sort_len")
    return df.reset_index(drop=True)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Context Rot 繁中實驗",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 繁體中文 Context Rot 實驗結果瀏覽器")
st.caption("Needle-in-a-Haystack：比較繁體 vs 簡體 context 對 LLM 長文檢索能力的影響")

all_data = load_all_data(mtime_key=_jsonl_max_mtime())
# 只保留有資料且在 MODEL_META 中定義的模型，按家族順序排列
available_models = [
    m for fam_models in FAMILIES.values()
    for m in fam_models
    if m in all_data
]
data_variants = {
    m: sorted(all_data[m].keys(), key=lambda v: ALL_VARIANTS.index(v) if v in ALL_VARIANTS else 99)
    for m in available_models
}

# ── 側邊欄 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    chart_type = st.radio(
        "圖表類型",
        ["準確率 vs 長度", "準確率 vs 位置", "熱力圖", "各 Needle 準確率",
         "Tokenizer Overhead", "衰退速率", "斷點分析", "模型敏感度", "斷點偏移"],
    )

    st.divider()
    st.subheader("🤖 選擇模型")

    selected_models = []
    for family, fam_models in FAMILIES.items():
        avail = [m for m in fam_models if m in all_data]
        if not avail:
            continue
        expanded = any(m in DEFAULT_MODELS for m in avail)
        with st.expander(family, expanded=expanded):
            for m in avail:
                cnt = sum(len(v) for v in all_data[m].values())
                vs  = ", ".join(data_variants[m])
                if st.checkbox(
                    MODEL_META[m]["label"],
                    value=m in DEFAULT_MODELS,
                    key=f"model_{m}",
                    help=f"{cnt} 筆 | {vs}",
                ):
                    selected_models.append(m)

    # variant 選擇（熱力圖和深度分析圖表不顯示）
    NO_VARIANT_CHARTS = {"熱力圖", "衰退速率", "斷點分析", "模型敏感度", "斷點偏移"}
    if chart_type not in NO_VARIANT_CHARTS:
        st.divider()
        st.subheader("📋 選擇 Variant")
        sel_variants = [
            v for v in ALL_VARIANTS
            if st.checkbox(v, value=(v == "繁問繁答"), key=f"var_{v}")
        ]

# ── 主要區域 ──────────────────────────────────────────────────────────────────

if chart_type == "熱力圖":
    col1, col2 = st.columns(2)
    hm_model = col1.selectbox(
        "模型",
        available_models,
        format_func=lambda m: MODEL_META.get(m, {}).get("label", m),
    )
    hm_variant = col2.selectbox("Variant", data_variants.get(hm_model, ALL_VARIANTS))
    fig = chart_heatmap(all_data, hm_model, hm_variant)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{hm_model} 沒有 {hm_variant} 資料")

elif not selected_models:
    st.info("← 請在左側勾選至少一個模型")

elif chart_type not in NO_VARIANT_CHARTS and chart_type != "Tokenizer Overhead" and not sel_variants:
    st.info("← 請在左側勾選至少一個 Variant")

else:
    if chart_type == "準確率 vs 長度":
        fig = chart_length(all_data, selected_models, sel_variants)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 數值表格"):
            df = length_table(all_data, selected_models, sel_variants)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)

    elif chart_type == "準確率 vs 位置":
        st.plotly_chart(chart_position(all_data, selected_models, sel_variants),
                        use_container_width=True)

    elif chart_type == "各 Needle 準確率":
        st.plotly_chart(chart_needle(all_data, selected_models, sel_variants),
                        use_container_width=True)

    elif chart_type == "Tokenizer Overhead":
        fig = chart_token_overhead(all_data, selected_models)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要同時有「繁問繁答」和「繁問簡答」資料的模型才能計算 token overhead。")

    elif chart_type == "衰退速率":
        fig = chart_rot_coefficient(all_data, selected_models)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("斜率越負代表衰退越快（每增加 1K 字元損失的準確率百分點）。"
                       "數值接近 0 表示長上下文下準確率穩定。")
        else:
            st.info("無足夠資料計算衰退係數")

    elif chart_type == "斷點分析":
        fig = chart_breakpoint(all_data, selected_models)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("斷點定義：以最短 3 個長度的平均準確率為基線，"
                       "準確率首次下降 ≥10 百分點的上下文長度。\">130K\" 表示在實驗範圍內未觸發。")
        else:
            st.info("無足夠資料進行斷點分析")

    elif chart_type == "模型敏感度":
        fig = chart_sensitivity(all_data, selected_models)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("左下角 = 衰退快且落差大（最脆弱）。"
                       "右上角靠近原點 = 穩定。不同標記形狀代表不同 variant。")
        else:
            st.info("無足夠資料")

    elif chart_type == "斷點偏移":
        fig = chart_breakpoint_shift(all_data, selected_models)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("比較同一模型在繁體 vs 簡體下的退化起始位置。"
                       "繁體斷點更早 = tokenizer 碎片化可能提前引發退化。")
        else:
            st.info("所選模型在實驗範圍內均未觸發斷點")
