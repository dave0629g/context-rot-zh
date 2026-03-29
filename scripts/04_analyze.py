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
import re
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")


def chinese_num_to_float(text: str) -> list[float]:
    """
    從文字中提取中文數字並轉為阿拉伯數字。

    支援：三點七 → 3.7、四百七十三億 → 47300000000、
         八百五十 → 850、百分之九十二點六 → 92.6
    """
    DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    # 簡體對應
    DIGITS.update({"两": 2})

    UNITS = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "億": 100000000}
    UNITS.update({"万": 10000, "亿": 100000000})

    results = []

    # 移除「百分之」前綴，但標記為百分比
    clean = text.replace("百分之", "")

    # 尋找連續的中文數字片段
    pattern = re.compile(
        r"[零一二兩两三四五六七八九十百千萬億万亿點点]+")
    for m in pattern.finditer(clean):
        s = m.group().replace("点", "點")
        # 解析整數部分和小數部分
        if "點" in s:
            int_part_s, dec_part_s = s.split("點", 1)
        else:
            int_part_s, dec_part_s = s, ""

        # 解析整數
        val = 0
        cur = 0
        has_digit = False
        for c in int_part_s:
            if c in DIGITS:
                cur = DIGITS[c]
                has_digit = True
            elif c in UNITS:
                u = UNITS[c]
                if u >= 10000:
                    # 萬/億 是大單位，把已累積的值乘上去
                    val = (val + max(cur, 1)) * u
                    cur = 0
                else:
                    val += max(cur, 1) * u
                    cur = 0

        val += cur

        if not has_digit and val == 0:
            continue

        # 小數部分：逐位讀取
        if dec_part_s:
            dec = 0.0
            for i, c in enumerate(dec_part_s):
                if c in DIGITS:
                    dec += DIGITS[c] * (10 ** -(i + 1))
            val += dec

        if val > 0:
            results.append(val)

    return results


def extract_all_numbers(text: str) -> set[float]:
    """
    從文字中提取所有數值，支援：
      - 純阿拉伯數字：473、3.7、92.6
      - 純中文數字：四百七十三億、三點七
      - 混合格式：473億、3.7萬
    回傳統一的 float 集合
    """
    UNIT_MAP = {"十": 10, "百": 100, "千": 1000,
                "萬": 10000, "億": 100000000,
                "万": 10000, "亿": 100000000}
    nums = set()

    # 1. 混合格式：阿拉伯數字 + 中文單位（473億 → 47300000000）
    for m in re.finditer(r"(\d+\.?\d*)\s*([十百千萬億万亿])", text):
        try:
            base = float(m.group(1))
            unit = UNIT_MAP.get(m.group(2), 1)
            nums.add(base * unit)
        except ValueError:
            pass

    # 2. 純阿拉伯數字
    for m in re.finditer(r"\d+\.?\d*", text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass

    # 3. 純中文數字
    nums.update(chinese_num_to_float(text))

    return nums


# 同義詞組：每組用一個 canonical token 取代所有變體
# 替換順序：長的先換，避免子串衝突（「新台幣」要比「台幣」先換）
SYNONYMS = [
    (["新臺幣", "新台幣", "臺幣", "台幣", "元"], "＄"),
    (["隻", "只", "頭"], "＃"),
]


def normalize_for_match(text: str) -> str:
    """正規化文字以提升比對命中率：同義詞替換為統一 token"""
    result = text
    for words, token in SYNONYMS:
        for word in words:  # 已按長度遞減排列
            result = result.replace(word, token)
    return result


def reevaluate(r: dict) -> bool:
    """
    用修正後的邏輯重算 is_correct。

    改進：
      1. 簡體 variant 額外做 OpenCC 轉換後比對
      2. 中文數字 → 阿拉伯數字正規化（三點七 ↔ 3.7）
      3. 同義詞比對（台幣 ↔ 元、隻 ↔ 只）
    """
    response = (r.get("model_response") or "").strip().lower()
    expected = r["expected_answer"].strip().lower()

    if not response:
        return False

    # 候選期望答案（原始 + 簡體轉換 + 同義詞正規化）
    candidates = [expected, normalize_for_match(expected)]
    try:
        import opencc
        exp_simp = opencc.OpenCC("t2s").convert(expected)
        candidates.append(exp_simp)
        candidates.append(normalize_for_match(exp_simp))
    except ImportError:
        pass

    resp_normalized = normalize_for_match(response)

    # 1. 字串包含比對（含同義詞正規化）
    exact_match = any(c in response or c in resp_normalized
                      for c in candidates)

    # 2. 數字比對：統一提取所有數值（阿拉伯、中文、混合格式）
    exp_nums = extract_all_numbers(expected)
    resp_nums = extract_all_numbers(response)
    number_match = bool(exp_nums and exp_nums.issubset(resp_nums))

    return exact_match or number_match


def load_results(model: str, eval_file: str = None, reeval: bool = False) -> list[dict]:
    """載入指定模型的結果，可選擇用 LLM 評估檔覆蓋原始判斷"""
    path = os.path.join(RESULTS_DIR, f"{model}_results.jsonl")
    if not os.path.exists(path):
        print(f"找不到結果檔案: {path}")
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # 排除已標記為跳過的記錄
    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        results = [r for r in results if not r.get("skipped")]
        print(f"排除 {len(skipped)} 筆 context_length_exceeded 記錄")

    if reeval:
        for r in results:
            r["evaluation"]["is_correct"] = reevaluate(r)
        print(f"已重新評估（修正版文字比對）：{len(results)} 筆")

    if eval_file:
        if not os.path.exists(eval_file):
            print(f"找不到評估檔案: {eval_file}")
            return results
        # 建立 (experiment_id, variant) → is_correct 的覆蓋表
        overrides = {}
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    overrides[(e["experiment_id"], e["variant"])] = e["is_correct"]
        applied = 0
        for r in results:
            key = (r["experiment_id"], r["variant"])
            if key in overrides:
                r["evaluation"]["is_correct"] = overrides[key]
                applied += 1
        print(f"已套用 LLM 評估：{applied}/{len(results)} 筆")

    return results


def detect_truncated_lengths(results: list[dict]) -> list[int]:
    """
    偵測可能因 context window 截斷而失真的實驗長度。

    判斷依據：同一長度下，所有記錄的 token_count_prompt 完全相同（零變異），
    代表全部撞到模型上限被截斷，而非自然的 token 數差異。
    """
    from collections import defaultdict
    length_tokens = defaultdict(list)
    for r in results:
        tp = r.get("token_count_prompt")
        if tp and tp > 0:
            length_tokens[r["context_length_chars"]].append(tp)

    truncated = []
    for length, tokens in length_tokens.items():
        if len(tokens) >= 2 and len(set(tokens)) == 1:
            truncated.append(length)
    return sorted(truncated)


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
        if token_count <= 0:
            token_count = r.get("token_count_prompt", -1)
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

    # 偵測截斷長度並警告
    truncated_lengths = detect_truncated_lengths(results)
    if truncated_lengths:
        lengths_str = ", ".join(f"{l:,}" for l in truncated_lengths)
        print(f"\n⚠️  警告：以下長度的 token_count_prompt 全部相同，")
        print(f"   疑似超出模型 context window 而被截斷，結果不可信：")
        print(f"   {lengths_str} 字元")

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
    parser.add_argument("--eval-file", default=None,
                        help="LLM 評估檔路徑（由 05_llm_judge.py 產生），覆蓋原始判斷")
    parser.add_argument("--reeval", action="store_true",
                        help="用修正版文字比對重算 is_correct（修正繁簡偏差，不需 LLM）")
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
        results = load_results(model, eval_file=args.eval_file, reeval=args.reeval)
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
