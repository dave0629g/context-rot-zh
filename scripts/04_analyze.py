"""
Step 4: 分析實驗結果

讀取實驗結果，產生：
  1. 繁體 vs 簡體的 context rot 衰退曲線
  2. Token 數差異統計
  3. 各 needle 位置的準確率熱力圖
  4. 統計顯著性檢驗

用法:
  python scripts/04_analyze.py
  python scripts/04_analyze.py --model qwen3
  python scripts/04_analyze.py --all

輸出: results/analysis/ 目錄下的圖表和統計報告
"""

import argparse
import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")


def load_results(model: str) -> list[dict]:
    """載入指定模型的結果"""
    path = os.path.join(RESULTS_DIR, f"{model}_results.jsonl")
    if not os.path.exists(path):
        print(f"找不到結果檔案: {path}")
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_accuracy_by_length(results: list[dict]) -> dict:
    """
    計算各 context 長度下的準確率

    回傳格式：
    {
        "traditional": {500: 0.95, 1000: 0.90, ...},
        "simplified":  {500: 0.98, 1000: 0.95, ...},
    }
    """
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in results:
        variant = r["variant"]
        length = r["context_length_chars"]
        is_correct = r["evaluation"]["is_correct"]

        counts[variant][length]["total"] += 1
        counts[variant][length]["correct"] += int(is_correct)

    accuracy = {}
    for variant in ["traditional", "simplified"]:
        accuracy[variant] = {}
        for length in sorted(counts[variant].keys()):
            c = counts[variant][length]
            if c["total"] > 0:
                accuracy[variant][length] = round(c["correct"] / c["total"], 4)

    return accuracy


def compute_accuracy_by_position(results: list[dict]) -> dict:
    """計算各 needle 位置的準確率"""
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in results:
        variant = r["variant"]
        position = r["needle_position"]
        is_correct = r["evaluation"]["is_correct"]

        counts[variant][position]["total"] += 1
        counts[variant][position]["correct"] += int(is_correct)

    accuracy = {}
    for variant in ["traditional", "simplified"]:
        accuracy[variant] = {}
        for pos in sorted(counts[variant].keys()):
            c = counts[variant][pos]
            if c["total"] > 0:
                accuracy[variant][pos] = round(c["correct"] / c["total"], 4)

    return accuracy


def compute_token_stats(results: list[dict]) -> dict:
    """統計繁體 vs 簡體的 token 數差異"""
    pairs = defaultdict(dict)

    for r in results:
        exp_id = r["experiment_id"]
        variant = r["variant"]
        token_count = r.get("token_count_actual", -1)
        if token_count > 0:
            pairs[exp_id][variant] = token_count

    ratios = []
    for exp_id, data in pairs.items():
        if "traditional" in data and "simplified" in data:
            ratio = data["traditional"] / data["simplified"]
            ratios.append({
                "experiment_id": exp_id,
                "traditional_tokens": data["traditional"],
                "simplified_tokens": data["simplified"],
                "ratio": round(ratio, 4),
            })

    if not ratios:
        return {"message": "無 token 計數資料"}

    avg_ratio = sum(r["ratio"] for r in ratios) / len(ratios)
    max_ratio = max(r["ratio"] for r in ratios)
    min_ratio = min(r["ratio"] for r in ratios)

    return {
        "pair_count": len(ratios),
        "avg_ratio_trad_over_simp": round(avg_ratio, 4),
        "max_ratio": round(max_ratio, 4),
        "min_ratio": round(min_ratio, 4),
        "interpretation": (
            f"繁體平均比簡體多 {(avg_ratio - 1) * 100:.1f}% 的 tokens"
            if avg_ratio > 1
            else f"繁體平均比簡體少 {(1 - avg_ratio) * 100:.1f}% 的 tokens"
        ),
    }


def compute_heatmap_data(results: list[dict]) -> dict:
    """計算 (長度 × 位置) 的準確率矩陣"""
    counts = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"correct": 0, "total": 0})
        )
    )

    for r in results:
        variant = r["variant"]
        length = r["context_length_chars"]
        position = r["needle_position"]
        is_correct = r["evaluation"]["is_correct"]

        counts[variant][length][position]["total"] += 1
        counts[variant][length][position]["correct"] += int(is_correct)

    heatmap = {}
    for variant in ["traditional", "simplified"]:
        heatmap[variant] = {}
        for length in sorted(counts[variant].keys()):
            heatmap[variant][length] = {}
            for pos in sorted(counts[variant][length].keys()):
                c = counts[variant][length][pos]
                if c["total"] > 0:
                    heatmap[variant][length][pos] = round(
                        c["correct"] / c["total"], 4
                    )

    return heatmap


def print_accuracy_table(accuracy: dict, label: str):
    """印出準確率表格"""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    lengths = sorted(
        set(list(accuracy.get("traditional", {}).keys())
            + list(accuracy.get("simplified", {}).keys()))
    )

    if not lengths:
        print("  （無資料）")
        return

    # 表頭
    print(f"  {'長度':>10s}  {'繁體':>8s}  {'簡體':>8s}  {'差異':>8s}")
    print(f"  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for length in lengths:
        trad = accuracy.get("traditional", {}).get(length, None)
        simp = accuracy.get("simplified", {}).get(length, None)

        trad_str = f"{trad * 100:6.1f}%" if trad is not None else "   N/A"
        simp_str = f"{simp * 100:6.1f}%" if simp is not None else "   N/A"

        if trad is not None and simp is not None:
            diff = (trad - simp) * 100
            diff_str = f"{diff:+6.1f}%"
        else:
            diff_str = "   N/A"

        print(f"  {length:>10,}  {trad_str}  {simp_str}  {diff_str}")


def generate_report(model: str, results: list[dict]):
    """產生完整的分析報告"""

    print(f"\n{'═' * 60}")
    print(f"  Context Rot 分析報告: {model}")
    print(f"  實驗結果數: {len(results)}")
    print(f"{'═' * 60}")

    # 1. Token 數統計
    token_stats = compute_token_stats(results)
    print(f"\n📊 Token 數差異統計")
    print(f"  比較組數: {token_stats.get('pair_count', 0)}")
    print(f"  繁/簡 token 比: {token_stats.get('avg_ratio_trad_over_simp', 'N/A')}")
    print(f"  最大比值: {token_stats.get('max_ratio', 'N/A')}")
    print(f"  最小比值: {token_stats.get('min_ratio', 'N/A')}")
    print(f"  解讀: {token_stats.get('interpretation', 'N/A')}")

    # 2. 按長度的準確率
    acc_by_length = compute_accuracy_by_length(results)
    print_accuracy_table(acc_by_length, "📈 準確率 vs Context 長度（字元）")

    # 3. 按位置的準確率
    acc_by_position = compute_accuracy_by_position(results)
    print_accuracy_table(acc_by_position, "📍 準確率 vs Needle 位置")

    # 4. 熱力圖數據
    heatmap = compute_heatmap_data(results)

    # 儲存完整分析結果
    analysis = {
        "model": model,
        "total_results": len(results),
        "token_stats": token_stats,
        "accuracy_by_length": acc_by_length,
        "accuracy_by_position": acc_by_position,
        "heatmap": heatmap,
    }

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    output_path = os.path.join(ANALYSIS_DIR, f"{model}_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"\n完整分析已儲存至: {output_path}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="分析 Context Rot 實驗結果")
    parser.add_argument("--model", help="指定分析某個模型")
    parser.add_argument("--all", action="store_true", help="分析所有模型")
    args = parser.parse_args()

    if args.all:
        # 掃描 results 目錄下的所有結果檔
        models = []
        for f in os.listdir(RESULTS_DIR):
            if f.endswith("_results.jsonl"):
                models.append(f.replace("_results.jsonl", ""))
    elif args.model:
        models = [args.model]
    else:
        print("請指定 --model 或 --all")
        return

    all_analyses = {}
    for model in models:
        results = load_results(model)
        if results:
            analysis = generate_report(model, results)
            all_analyses[model] = analysis

    # 如果有多個模型，做跨模型比較
    if len(all_analyses) > 1:
        print(f"\n{'═' * 60}")
        print(f"  跨模型比較")
        print(f"{'═' * 60}")

        print(f"\n  {'模型':>15s}  {'繁/簡token比':>12s}  {'繁體最長準確率':>14s}  {'簡體最長準確率':>14s}")
        print(f"  {'─' * 15}  {'─' * 12}  {'─' * 14}  {'─' * 14}")

        for model, analysis in all_analyses.items():
            ratio = analysis["token_stats"].get("avg_ratio_trad_over_simp", "N/A")
            acc_trad = analysis["accuracy_by_length"].get("traditional", {})
            acc_simp = analysis["accuracy_by_length"].get("simplified", {})

            max_len = max(list(acc_trad.keys()) + list(acc_simp.keys()), default=0)
            trad_at_max = acc_trad.get(max_len, None)
            simp_at_max = acc_simp.get(max_len, None)

            trad_str = f"{trad_at_max*100:.1f}%" if trad_at_max else "N/A"
            simp_str = f"{simp_at_max*100:.1f}%" if simp_at_max else "N/A"

            print(f"  {model:>15s}  {str(ratio):>12s}  {trad_str:>14s}  {simp_str:>14s}")


if __name__ == "__main__":
    main()
