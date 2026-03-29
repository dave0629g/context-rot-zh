"""
Step 5: 產生實驗結果視覺化圖表

產生：
  1. accuracy_vs_length.png  - 準確率 vs Context 長度（雙模型比較）
  2. accuracy_vs_position.png - 準確率 vs Needle 位置（雙模型比較）
  3. heatmap_{model}.png      - 每個模型的 (長度 × 位置) 熱力圖（繁/簡對比）

用法:
  python scripts/05_plot_results.py
  python scripts/05_plot_results.py --models gemma3:4b llama3.1:8b
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # 無 display 環境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import numpy as np

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "analysis")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

# 中文字型：優先使用楷書（標楷體風格），fallback 為 Noto Sans CJK
# AR PL UKai TW 為系統內最接近標楷體的楷書字型
_CJK_LOADED = False
for _fp in ["/usr/share/fonts/truetype/arphic/ukai.ttc",          # AR PL UKai (楷書)
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _font_name = fm.FontProperties(fname=_fp).get_name()
        matplotlib.rcParams["font.family"] = _font_name
        _CJK_LOADED = True
        break

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

MODEL_COLORS = {
    "gemma3:4b":   ("#4C72B0", "#88BBDD"),   # 藍
    "llama3.1:8b": ("#C44E52", "#EE9999"),   # 紅
    "qwen3:8b":    ("#55A868", "#AADDBB"),   # 綠
    "qwen3.5:35b": ("#8172B2", "#BBAADD"),   # 紫
}
DEFAULT_COLORS = ("#666666", "#AAAAAA")

MODEL_LABELS = {
    "gemma3:4b":   "Gemma 3 4B",
    "llama3.1:8b": "Llama 3.1 8B",
    "qwen3:8b":    "Qwen3 8B",
    "qwen3.5:35b": "Qwen3.5 35B",
}


def load_analysis(model: str) -> dict | None:
    path = os.path.join(ANALYSIS_DIR, f"{model}_analysis.json")
    if not os.path.exists(path):
        print(f"找不到分析檔: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    # JSON keys 一律是字串，把長度轉回 int、位置轉回 float
    for section in ["accuracy_by_length", "heatmap"]:
        for variant in data.get(section, {}):
            data[section][variant] = {
                int(k): v for k, v in data[section][variant].items()
            }
            if section == "heatmap":
                data[section][variant] = {
                    length: {float(p): acc for p, acc in pos_dict.items()}
                    for length, pos_dict in data[section][variant].items()
                }
    for variant in data.get("accuracy_by_position", {}):
        data["accuracy_by_position"][variant] = {
            float(k): v for k, v in data["accuracy_by_position"][variant].items()
        }
    return data


# ── 1. 準確率 vs Context 長度 ────────────────────────────────────────────────

def plot_accuracy_vs_length(models_data: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Context Rot: Accuracy vs Context Length", fontsize=14, fontweight="bold")

    for ax, variant, title in zip(axes, ["traditional", "simplified"],
                                  ["Traditional Chinese (繁體)", "Simplified Chinese (簡體)"]):
        for model, data in models_data.items():
            acc = data["accuracy_by_length"].get(variant, {})
            lengths = sorted(acc.keys())
            values = [acc[l] * 100 for l in lengths]
            colors = MODEL_COLORS.get(model, DEFAULT_COLORS)
            label = MODEL_LABELS.get(model, model)
            ax.plot(range(len(lengths)), values, "o-", color=colors[0],
                    label=label, linewidth=2, markersize=5)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lengths],
                           rotation=45, ha="right")
        ax.set_xlabel("Context Length (chars)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(50, 105)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_path}")


# ── 2. 準確率 vs Needle 位置 ─────────────────────────────────────────────────

def plot_accuracy_vs_position(models_data: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Context Rot: Accuracy vs Needle Position", fontsize=14, fontweight="bold")

    for ax, variant, title in zip(axes, ["traditional", "simplified"],
                                  ["Traditional Chinese (繁體)", "Simplified Chinese (簡體)"]):
        for model, data in models_data.items():
            acc = data["accuracy_by_position"].get(variant, {})
            positions = sorted(acc.keys())
            values = [acc[p] * 100 for p in positions]
            colors = MODEL_COLORS.get(model, DEFAULT_COLORS)
            label = MODEL_LABELS.get(model, model)
            ax.plot([p * 100 for p in positions], values, "o-", color=colors[0],
                    label=label, linewidth=2, markersize=5)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Needle Position (%)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xlim(-5, 105)
        ax.set_ylim(50, 105)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_path}")


# ── 3. 繁 vs 簡 準確率差異（折線） ──────────────────────────────────────────

def plot_trad_simp_gap(models_data: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Traditional vs Simplified Accuracy Gap", fontsize=14, fontweight="bold")

    for ax, x_key, xlabel, title in [
        (axes[0], "accuracy_by_length",
         "Context Length (chars)", "Gap by Context Length"),
        (axes[1], "accuracy_by_position",
         "Needle Position (%)", "Gap by Needle Position"),
    ]:
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        for model, data in models_data.items():
            trad = data[x_key].get("traditional", {})
            simp = data[x_key].get("simplified", {})
            keys = sorted(set(trad) & set(simp))
            gaps = [(trad[k] - simp[k]) * 100 for k in keys]

            colors = MODEL_COLORS.get(model, DEFAULT_COLORS)
            label = MODEL_LABELS.get(model, model)

            if x_key == "accuracy_by_length":
                x_vals = range(len(keys))
                ax.set_xticks(x_vals)
                ax.set_xticklabels(
                    [f"{k//1000}k" if k >= 1000 else str(k) for k in keys],
                    rotation=45, ha="right")
            else:
                x_vals = [k * 100 for k in keys]
                ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

            ax.plot(x_vals, gaps, "o-", color=colors[0],
                    label=label, linewidth=2, markersize=5)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Traditional − Simplified (pp)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_path}")


# ── 4. 熱力圖（長度 × 位置） ─────────────────────────────────────────────────

def plot_heatmap(model: str, data: dict, out_path: str):
    heatmap = data["heatmap"]
    label = MODEL_LABELS.get(model, model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"NIAH Heatmap: {label}", fontsize=14, fontweight="bold")

    for ax, variant, title in zip(axes, ["traditional", "simplified"],
                                  ["Traditional Chinese (繁體)", "Simplified Chinese (簡體)"]):
        hm = heatmap.get(variant, {})
        lengths = sorted(hm.keys())
        all_positions = sorted({p for d in hm.values() for p in d})

        matrix = np.full((len(lengths), len(all_positions)), np.nan)
        for i, length in enumerate(lengths):
            for j, pos in enumerate(all_positions):
                val = hm.get(length, {}).get(pos, None)
                if val is not None:
                    matrix[i, j] = val * 100

        im = ax.imshow(matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn", vmin=0, vmax=100)

        ax.set_xticks(range(len(all_positions)))
        ax.set_xticklabels([f"{int(p*100)}%" for p in all_positions], fontsize=8)
        ax.set_yticks(range(len(lengths)))
        ax.set_yticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lengths], fontsize=8)
        ax.set_xlabel("Needle Position", fontsize=10)
        ax.set_ylabel("Context Length (chars)", fontsize=10)
        ax.set_title(title, fontsize=11)

        # 數值標注
        for i in range(len(lengths)):
            for j in range(len(all_positions)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 40 or val > 85 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=6.5, color=text_color)

        plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.03)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_path}")


# ── 5. Token 比 bar chart ────────────────────────────────────────────────────

def plot_token_ratio(models_data: dict, out_path: str):
    labels = [MODEL_LABELS.get(m, m) for m in models_data]
    ratios = [d["token_stats"]["avg_ratio_trad_over_simp"] for d in models_data.values()]
    colors = [MODEL_COLORS.get(m, DEFAULT_COLORS)[0] for m in models_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [(r - 1) * 100 for r in ratios], color=colors, width=0.4)

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"+{(r-1)*100:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Extra tokens in Traditional vs Simplified (%)", fontsize=10)
    ax.set_title("Tokenizer Overhead: Traditional vs Simplified Chinese", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(0, max((r - 1) * 100 for r in ratios) * 1.4 + 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gemma3:4b", "llama3.1:8b"])
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    models_data = {}
    for model in args.models:
        data = load_analysis(model)
        if data:
            models_data[model] = data

    if not models_data:
        print("沒有可用的分析資料")
        return

    print(f"載入模型: {list(models_data.keys())}")

    # 生成所有圖表
    plot_accuracy_vs_length(models_data,
        os.path.join(PLOT_DIR, "accuracy_vs_length.png"))

    plot_accuracy_vs_position(models_data,
        os.path.join(PLOT_DIR, "accuracy_vs_position.png"))

    plot_trad_simp_gap(models_data,
        os.path.join(PLOT_DIR, "trad_simp_gap.png"))

    for model, data in models_data.items():
        plot_heatmap(model, data,
            os.path.join(PLOT_DIR, f"heatmap_{model.replace(':', '_')}.png"))

    plot_token_ratio(models_data,
        os.path.join(PLOT_DIR, "token_ratio.png"))

    print(f"\n所有圖表已儲存至 results/plots/")


if __name__ == "__main__":
    main()
