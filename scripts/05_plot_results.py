"""
Step 5: 產生實驗結果視覺化圖表（三 variant 版）

每個模型產出：
  1. {model}_accuracy_vs_length.png  - 準確率 vs Context 長度（3 條線）
  2. {model}_accuracy_vs_position.png - 準確率 vs Needle 位置（3 條線）
  3. {model}_heatmap.png              - 熱力圖 3 面板（繁問繁答/簡問簡答/繁問簡答）
  4. {model}_needle_accuracy.png      - 各 Needle 準確率分組長條圖

跨模型比較：
  5. compare_accuracy_vs_length.png   - 多模型對比（每 variant 一面板）
  6. compare_token_ratio.png          - 繁/簡 Tokenizer overhead
  7. compare_65k_accuracy.png         - 65k 長度各模型 × 各 variant

用法:
  python scripts/05_plot_results.py
  python scripts/05_plot_results.py --models gemma3:4b llama3.1:8b
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import numpy as np

# 載入 reevaluate
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_analyze = import_module("04_analyze")
reevaluate = _analyze.reevaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

# 中文字型
for _fp in ["/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

# 標記符號：空心/實心交替，確保黑白列印可辨識
MARKER_STYLES = [
    dict(marker="o",  fillstyle="full",  linestyle="-"),
    dict(marker="o",  fillstyle="none",  linestyle="--"),
    dict(marker="s",  fillstyle="full",  linestyle="-"),
    dict(marker="s",  fillstyle="none",  linestyle="--"),
    dict(marker="^",  fillstyle="full",  linestyle="-."),
    dict(marker="^",  fillstyle="none",  linestyle="-."),
    dict(marker="D",  fillstyle="full",  linestyle=":"),
    dict(marker="D",  fillstyle="none",  linestyle=":"),
    dict(marker="v",  fillstyle="full",  linestyle="-"),
    dict(marker="P",  fillstyle="full",  linestyle="--"),
]

VARIANT_LABELS = {"繁問繁答": "#2E86AB", "簡問簡答": "#A23B72", "繁問簡答": "#F18F01"}
VARIANT_MARKERS = {
    "繁問繁答": dict(marker="o",  fillstyle="full",  linestyle="-"),
    "簡問簡答": dict(marker="s",  fillstyle="none",  linestyle="--"),
    "繁問簡答": dict(marker="^",  fillstyle="full",  linestyle="-."),
}
MODEL_COLORS = {
    "gemma3:4b":   "#4C72B0",
    "llama3.1:8b": "#C44E52",
    "qwen3:8b":    "#55A868",
    "qwen3.5:35b": "#8172B2",
    "gemma3:27b":  "#CCB974",
    "llama3.3:70b":"#64B5CD",
}
MODEL_MARKERS = {
    # Gemma 家族：圓形，小=空心，大=實心
    "gemma3:4b":    dict(marker="o",  fillstyle="none",  linestyle="--"),
    "gemma3:27b":   dict(marker="o",  fillstyle="full",  linestyle="-"),
    # Llama 家族：正方形，小=空心，大=實心
    "llama3.1:8b":  dict(marker="s",  fillstyle="none",  linestyle="--"),
    "llama3.3:70b": dict(marker="s",  fillstyle="full",  linestyle="-"),
    # Qwen 家族：上三角，小=空心，大=實心
    "qwen3:8b":     dict(marker="^",  fillstyle="none",  linestyle="--"),
    "qwen3.5:35b":  dict(marker="^",  fillstyle="full",  linestyle="-"),
}
MODEL_LABELS = {
    "gemma3:4b": "Gemma 3 4B", "llama3.1:8b": "Llama 3.1 8B",
    "qwen3:8b": "Qwen3 8B", "qwen3.5:35b": "Qwen3.5 35B",
    "gemma3:27b": "Gemma 3 27B", "llama3.3:70b": "Llama 3.3 70B",
}


# ── 資料載入 ──────────────────────────────────────────────────────────────────

def load_model_data(model: str) -> dict:
    """載入模型的三種 variant 資料，回傳 {variant_label: [records]}"""
    sources = [
        ("繁問繁答", f"results/{model}_results.jsonl", "traditional"),
        ("簡問簡答", f"results/h2_{model}_results.jsonl", "simplified_q"),
        ("繁問簡答", f"results/{model}_results.jsonl", "simplified"),
    ]
    data = {}
    for label, path, vk in sources:
        full_path = os.path.join(os.path.dirname(__file__), "..", path)
        if not os.path.exists(full_path):
            continue
        records = []
        with open(full_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("skipped") or r.get("variant") != vk:
                        continue
                    r["_correct"] = reevaluate(r)
                    records.append(r)
                except:
                    pass
        if records:
            data[label] = records
    return data


def acc_by_key(records, key_fn):
    d = defaultdict(lambda: [0, 0])
    for r in records:
        k = key_fn(r)
        d[k][1] += 1
        d[k][0] += int(r["_correct"])
    return {k: c / t * 100 for k, (c, t) in d.items() if t > 0}


# ── 1. 準確率 vs Context 長度 ────────────────────────────────────────────────

def plot_accuracy_vs_length(model, data, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    mlabel = MODEL_LABELS.get(model, model)
    ax.set_title(f"{mlabel}：準確率 vs Context 長度", fontsize=13, fontweight="bold")

    for variant, color in VARIANT_LABELS.items():
        if variant not in data:
            continue
        acc = acc_by_key(data[variant], lambda r: r["context_length_chars"])
        lengths = sorted(acc)
        st = VARIANT_MARKERS[variant]
        ax.plot(range(len(lengths)), [acc[l] for l in lengths],
                color=color, label=variant, linewidth=2, markersize=7,
                marker=st["marker"], fillstyle=st["fillstyle"],
                linestyle=st["linestyle"], markeredgewidth=1.5)

    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lengths],
                       rotation=45, ha="right")
    ax.set_xlabel("Context 長度（字元）")
    ax.set_ylabel("準確率（%）")
    ax.set_ylim(70, 102)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 2. 準確率 vs Needle 位置 ─────────────────────────────────────────────────

def plot_accuracy_vs_position(model, data, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    mlabel = MODEL_LABELS.get(model, model)
    ax.set_title(f"{mlabel}：準確率 vs Needle 位置", fontsize=13, fontweight="bold")

    for variant, color in VARIANT_LABELS.items():
        if variant not in data:
            continue
        acc = acc_by_key(data[variant], lambda r: r["needle_position"])
        positions = sorted(acc)
        st = VARIANT_MARKERS[variant]
        ax.plot([p * 100 for p in positions], [acc[p] for p in positions],
                color=color, label=variant, linewidth=2, markersize=7,
                marker=st["marker"], fillstyle=st["fillstyle"],
                linestyle=st["linestyle"], markeredgewidth=1.5)

    ax.set_xlabel("Needle 位置（%）")
    ax.set_ylabel("準確率（%）")
    ax.set_xlim(-5, 105)
    ax.set_ylim(80, 102)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 3. 熱力圖（3 面板） ──────────────────────────────────────────────────────

def plot_heatmap(model, data, out_path):
    available = [v for v in VARIANT_LABELS if v in data]
    n = len(available)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6))
    if n == 1:
        axes = [axes]
    mlabel = MODEL_LABELS.get(model, model)
    fig.suptitle(f"NIAH Heatmap：{mlabel}", fontsize=14, fontweight="bold")

    for ax, variant in zip(axes, available):
        acc_2d = defaultdict(dict)
        for r in data[variant]:
            l, p = r["context_length_chars"], r["needle_position"]
            key = (l, p)
            if key not in acc_2d:
                acc_2d[key] = [0, 0]
            acc_2d[key][1] += 1
            acc_2d[key][0] += int(r["_correct"])

        lengths = sorted(set(k[0] for k in acc_2d))
        positions = sorted(set(k[1] for k in acc_2d))
        matrix = np.full((len(lengths), len(positions)), np.nan)
        for i, l in enumerate(lengths):
            for j, p in enumerate(positions):
                if (l, p) in acc_2d:
                    c, t = acc_2d[(l, p)]
                    matrix[i, j] = c / t * 100

        im = ax.imshow(matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f"{int(p*100)}%" for p in positions], fontsize=7)
        ax.set_yticks(range(len(lengths)))
        ax.set_yticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lengths], fontsize=8)
        ax.set_xlabel("Needle 位置")
        ax.set_ylabel("Context 長度")
        ax.set_title(variant, fontsize=11)

        for i in range(len(lengths)):
            for j in range(len(positions)):
                val = matrix[i, j]
                if not np.isnan(val):
                    tc = "white" if val < 40 or val > 85 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=6.5, color=tc)
        plt.colorbar(im, ax=ax, label="準確率（%）", fraction=0.03)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 4. 各 Needle 準確率 ──────────────────────────────────────────────────────

NEEDLE_LABELS = {"N01": "N01 金額", "N02": "N02 人名", "N03": "N03 面積",
                 "N04": "N04 數量", "N05": "N05 百分比"}

def plot_needle_accuracy(model, data, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    mlabel = MODEL_LABELS.get(model, model)
    ax.set_title(f"{mlabel}：各 Needle 準確率", fontsize=13, fontweight="bold")

    available = [v for v in VARIANT_LABELS if v in data]
    needles = sorted(NEEDLE_LABELS.keys())
    x = np.arange(len(needles))
    width = 0.25

    for i, variant in enumerate(available):
        acc = acc_by_key(data[variant], lambda r: r["needle_id"])
        vals = [acc.get(n, 0) for n in needles]
        color = VARIANT_LABELS[variant]
        bars = ax.bar(x + (i - len(available) / 2 + 0.5) * width, vals,
                      width * 0.9, label=variant, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([NEEDLE_LABELS[n] for n in needles])
    ax.set_ylabel("準確率（%）")
    ax.set_ylim(85, 103)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 5. 跨模型：準確率 vs Context 長度 ────────────────────────────────────────

def plot_compare_length(all_data, out_path):
    variants = ["繁問繁答", "簡問簡答", "繁問簡答"]
    available = [v for v in variants if any(v in d for d in all_data.values())]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("跨模型比較：準確率 vs Context 長度", fontsize=14, fontweight="bold")

    for ax, variant in zip(axes, available):
        for model, data in all_data.items():
            if variant not in data:
                continue
            acc = acc_by_key(data[variant], lambda r: r["context_length_chars"])
            lengths = sorted(acc)
            color = MODEL_COLORS.get(model, "#666666")
            label = MODEL_LABELS.get(model, model)
            st = MODEL_MARKERS.get(model, MARKER_STYLES[0])
            ax.plot(range(len(lengths)), [acc[l] for l in lengths],
                    color=color, label=label, linewidth=2, markersize=6,
                    marker=st["marker"], fillstyle=st["fillstyle"],
                    linestyle=st["linestyle"], markeredgewidth=1.5)

        ax.set_title(variant, fontsize=11)
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lengths],
                           rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Context 長度")
        ax.set_ylabel("準確率（%）")
        ax.set_ylim(70, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 6. Token 比率 ────────────────────────────────────────────────────────────

def plot_token_ratio(all_data, out_path):
    models = []
    ratios = []
    for model, data in all_data.items():
        trad = {r["experiment_id"]: r.get("token_count_prompt", 0)
                for r in data.get("繁問繁答", [])}
        simp = {r["experiment_id"]: r.get("token_count_prompt", 0)
                for r in data.get("繁問簡答", [])}
        r_list = []
        for eid in set(trad) & set(simp):
            if trad[eid] > 0 and simp[eid] > 0:
                r_list.append(trad[eid] / simp[eid])
        if r_list:
            models.append(model)
            ratios.append(sum(r_list) / len(r_list))

    fig, ax = plt.subplots(figsize=(max(5, len(models) * 2), 4))
    colors = [MODEL_COLORS.get(m, "#666") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    bars = ax.bar(labels, [(r - 1) * 100 for r in ratios], color=colors, width=0.4)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"+{(r-1)*100:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("繁體比簡體多用的 Token（%）")
    ax.set_title("Tokenizer Overhead：繁體 vs 簡體", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max((r - 1) * 100 for r in ratios) * 1.5 + 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 7. 65k 準確率比較 ────────────────────────────────────────────────────────

def plot_65k_comparison(all_data, out_path):
    variants = ["繁問繁答", "簡問簡答", "繁問簡答"]
    models = list(all_data.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.5), 5))
    ax.set_title("65,000 字元下各模型 × 各 Variant 準確率", fontsize=13, fontweight="bold")

    for i, variant in enumerate(variants):
        vals = []
        for model in models:
            data = all_data[model]
            if variant in data:
                acc = acc_by_key(data[variant], lambda r: r["context_length_chars"])
                vals.append(acc.get(65000, 0))
            else:
                vals.append(0)
        color = VARIANT_LABELS[variant]
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.9,
                      label=variant, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("準確率（%）")
    ax.set_ylim(60, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── 8. 字元數 → Token 數對照圖 ───────────────────────────────────────────────

CTX_WINDOWS = {
    "gemma3:4b": 131072, "llama3.1:8b": 131072,
    "qwen3:8b": 40960, "qwen3.5:35b": 262144,
    "gemma3:27b": 131072, "llama3.3:70b": 131072,
}

def plot_token_map(all_data, out_path):
    """字元數 vs 實際 token 數：顯示各模型 tokenizer 效率差異與 context window 上限"""
    from collections import defaultdict

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("字元數 → 實際 Token 數（繁體，各模型 Tokenizer 對照）",
                 fontsize=13, fontweight="bold")

    for model, data in all_data.items():
        trad = data.get("繁問繁答", [])
        if not trad:
            continue

        # 計算每個長度的平均 token 數
        by_len = defaultdict(list)
        for r in trad:
            tp = r.get("token_count_prompt", 0)
            if tp > 0:
                by_len[r["context_length_chars"]].append(tp)

        lengths = sorted(by_len)
        avg_tokens = [sum(by_len[l]) / len(by_len[l]) for l in lengths]

        color = MODEL_COLORS.get(model, "#666666")
        label = MODEL_LABELS.get(model, model)
        st = MODEL_MARKERS.get(model, MARKER_STYLES[0])
        ax.plot(lengths, avg_tokens, color=color, label=label,
                linewidth=2, markersize=6,
                marker=st["marker"], fillstyle=st["fillstyle"],
                linestyle=st["linestyle"], markeredgewidth=1.5)

        # 在最後一個點標注數值
        if avg_tokens:
            ax.annotate(f"{avg_tokens[-1]/1000:.0f}K",
                        (lengths[-1], avg_tokens[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=8, color=color)

    # 畫 context window 上限線
    drawn_limits = set()
    for model in all_data:
        ctx = CTX_WINDOWS.get(model, 0)
        if ctx > 0 and ctx not in drawn_limits:
            ax.axhline(ctx, linestyle="--", alpha=0.5, color="#999999", linewidth=1)
            ax.text(ax.get_xlim()[0], ctx + 1500,
                    f"Context Window {ctx:,}",
                    fontsize=8, color="#999999")
            drawn_limits.add(ctx)

    ax.set_xlabel("Context 長度（字元）")
    ax.set_ylabel("實際 Token 數")
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{l//1000}K" if l >= 1000 else str(l) for l in lengths],
                       rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  已儲存: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gemma3:4b", "llama3.1:8b"])
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    all_data = {}
    for model in args.models:
        print(f"載入 {model}...")
        data = load_model_data(model)
        if data:
            all_data[model] = data
            for label, records in data.items():
                print(f"  {label}: {len(records)} 筆")

    if not all_data:
        print("沒有可用的資料")
        return

    # 每個模型的圖表
    for model, data in all_data.items():
        safe = model.replace(":", "_")
        print(f"\n產生 {model} 圖表...")
        plot_accuracy_vs_length(model, data,
            os.path.join(PLOT_DIR, f"{safe}_accuracy_vs_length.png"))
        plot_accuracy_vs_position(model, data,
            os.path.join(PLOT_DIR, f"{safe}_accuracy_vs_position.png"))
        plot_heatmap(model, data,
            os.path.join(PLOT_DIR, f"{safe}_heatmap.png"))
        plot_needle_accuracy(model, data,
            os.path.join(PLOT_DIR, f"{safe}_needle_accuracy.png"))

    # 跨模型比較
    if len(all_data) >= 2:
        print(f"\n產生跨模型比較圖表...")
        plot_compare_length(all_data,
            os.path.join(PLOT_DIR, "compare_accuracy_vs_length.png"))

    plot_token_ratio(all_data,
        os.path.join(PLOT_DIR, "compare_token_ratio.png"))

    if len(all_data) >= 2:
        plot_65k_comparison(all_data,
            os.path.join(PLOT_DIR, "compare_65k_accuracy.png"))

    plot_token_map(all_data,
        os.path.join(PLOT_DIR, "compare_token_map.png"))

    print(f"\n所有圖表已儲存至 results/plots/")


if __name__ == "__main__":
    main()
