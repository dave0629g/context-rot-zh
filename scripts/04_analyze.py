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


def _load_jsonl(path: str) -> list[dict]:
    """載入 JSONL 檔案，排除 skipped 記錄"""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        results = [r for r in results if not r.get("skipped")]
        print(f"  排除 {len(skipped)} 筆 context_length_exceeded 記錄 ({os.path.basename(path)})")
    return results


def load_results(model: str, eval_file: str = None, reeval: bool = False) -> list[dict]:
    """載入指定模型的結果（含 h2_ 檔案），可選擇用 LLM 評估檔覆蓋原始判斷"""
    path = os.path.join(RESULTS_DIR, f"{model}_results.jsonl")
    h2_path = os.path.join(RESULTS_DIR, f"h2_{model}_results.jsonl")

    if not os.path.exists(path):
        print(f"找不到結果檔案: {path}")
        return []

    results = _load_jsonl(path)

    # 合併 h2_ 結果（simplified_q variant）
    if os.path.exists(h2_path):
        h2_results = _load_jsonl(h2_path)
        print(f"合併 h2_ 結果：{len(h2_results)} 筆 simplified_q")
        results.extend(h2_results)

    if reeval:
        for r in results:
            r["evaluation"]["is_correct"] = reevaluate(r)
        # 統計各 variant 的 reeval 數量
        from collections import Counter
        vc = Counter(r["variant"] for r in results)
        parts = ", ".join(f"{v}={n}" for v, n in sorted(vc.items()))
        print(f"已重新評估（修正版文字比對）：{len(results)} 筆 ({parts})")

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
    for variant in ["traditional", "simplified", "simplified_q"]:
        if variant not in counts:
            continue
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
    for variant in ["traditional", "simplified", "simplified_q"]:
        if variant not in counts:
            continue
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
    for variant in ["traditional", "simplified", "simplified_q"]:
        if variant not in counts:
            continue
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


def compute_rot_coefficient(results: list[dict]) -> dict:
    """
    計算每個 variant 的衰退係數（rot coefficient）。

    方法：以短上下文（≤8000 字元）的平均準確率為基線，
    對所有長度做線性迴歸，斜率即為每增加 1K 字元的準確率變化（pp/1K chars）。
    同時回傳 R² 以評估線性假設的合理性。
    """
    acc = compute_accuracy_by_length(results)
    rot = {}
    for variant, length_acc in acc.items():
        points = sorted(length_acc.items())
        if len(points) < 3:
            continue
        xs = [l / 1000 for l, _ in points]  # 轉為 K chars
        ys = [a * 100 for _, a in points]    # 轉為百分比

        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))

        denom = n * sxx - sx * sx
        if denom == 0:
            continue
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        # R²
        y_mean = sy / n
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 短上下文基線（≤8K）
        baseline_pts = [a for l, a in points if l <= 8000]
        baseline = sum(baseline_pts) / len(baseline_pts) * 100 if baseline_pts else ys[0]

        rot[variant] = {
            "slope_pp_per_1k_chars": round(slope, 4),
            "intercept": round(intercept, 2),
            "r_squared": round(r_squared, 4),
            "baseline_pct": round(baseline, 2),
            "points_count": n,
        }
    return rot


def compute_breakpoint(results: list[dict], threshold_drop: float = 10.0) -> dict:
    """
    偵測每個 variant 的性能斷點（breakpoint）。

    定義：以最短 3 個長度的平均準確率為基線，
    斷點為準確率首次下降超過 threshold_drop (pp) 的上下文長度。
    同時回傳準確率首次低於 80% 的長度。
    """
    acc = compute_accuracy_by_length(results)
    breakpoints = {}
    for variant, length_acc in acc.items():
        points = sorted(length_acc.items())
        if len(points) < 3:
            continue

        # 基線：最短 3 個長度的平均
        baseline = sum(a for _, a in points[:3]) / 3 * 100

        bp_drop = None
        bp_80 = None
        for length, a in points:
            pct = a * 100
            if bp_drop is None and baseline - pct >= threshold_drop:
                bp_drop = length
            if bp_80 is None and pct < 80:
                bp_80 = length

        breakpoints[variant] = {
            "baseline_pct": round(baseline, 2),
            "drop_threshold_pp": threshold_drop,
            "breakpoint_drop": bp_drop,        # 首次掉 threshold_drop pp 的長度
            "breakpoint_below_80": bp_80,       # 首次低於 80% 的長度
            "acc_at_max_length": round(points[-1][1] * 100, 2),
            "max_length": points[-1][0],
            "total_drop_pp": round(baseline - points[-1][1] * 100, 2),
        }
    return breakpoints


def compute_token_overhead_by_length(results: list[dict]) -> dict:
    """
    按上下文長度計算繁體 vs 簡體的 token overhead。

    回傳 {context_length: {avg_ratio, trad_tokens, simp_tokens, overhead_pct}}
    """
    from collections import defaultdict

    pairs = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r.get("skipped"):
            continue
        tp = r.get("token_count_prompt") or r.get("token_count_actual") or 0
        if tp > 0:
            pairs[r["experiment_id"]][r["variant"]] = {
                "tokens": tp,
                "length": r["context_length_chars"],
            }

    length_data = defaultdict(lambda: {"trad": [], "simp": []})
    for exp_id, variants in pairs.items():
        if "traditional" in variants and "simplified" in variants:
            length = variants["traditional"]["length"]
            length_data[length]["trad"].append(variants["traditional"]["tokens"])
            length_data[length]["simp"].append(variants["simplified"]["tokens"])

    result = {}
    for length in sorted(length_data):
        d = length_data[length]
        if d["trad"] and d["simp"]:
            avg_trad = sum(d["trad"]) / len(d["trad"])
            avg_simp = sum(d["simp"]) / len(d["simp"])
            result[length] = {
                "avg_trad_tokens": round(avg_trad, 1),
                "avg_simp_tokens": round(avg_simp, 1),
                "overhead_pct": round((avg_trad / avg_simp - 1) * 100, 2),
                "pair_count": min(len(d["trad"]), len(d["simp"])),
            }
    return result


def compute_accuracy_by_needle(results: list[dict]) -> dict:
    """計算各 variant 各 needle 的準確率"""
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in results:
        variant = r["variant"]
        needle = r.get("needle_id", "unknown")
        is_correct = r["evaluation"]["is_correct"]
        counts[variant][needle]["total"] += 1
        counts[variant][needle]["correct"] += int(is_correct)

    result = {}
    for variant in ["traditional", "simplified", "simplified_q"]:
        if variant not in counts:
            continue
        result[variant] = {}
        for needle in sorted(counts[variant]):
            c = counts[variant][needle]
            if c["total"] > 0:
                result[variant][needle] = round(c["correct"] / c["total"], 4)
    return result


def compute_long_context_accuracy(results: list[dict], min_length: int = 65000) -> dict:
    """計算長上下文（≥min_length）下各 variant 的準確率"""
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["context_length_chars"] >= min_length:
            v = r["variant"]
            counts[v]["total"] += 1
            counts[v]["correct"] += int(r["evaluation"]["is_correct"])
    return {
        v: round(c["correct"] / c["total"], 4) if c["total"] > 0 else None
        for v, c in counts.items()
    }


def print_accuracy_table(accuracy: dict, label: str):
    """印出準確率表格（支援 traditional / simplified / simplified_q）"""
    print(f"\n{'─' * 75}")
    print(f"  {label}")
    print(f"{'─' * 75}")

    all_keys = set()
    for v in ["traditional", "simplified", "simplified_q"]:
        all_keys.update(accuracy.get(v, {}).keys())
    lengths = sorted(all_keys)

    if not lengths:
        print("  （無資料）")
        return

    has_simp = bool(accuracy.get("simplified"))
    has_simp_q = bool(accuracy.get("simplified_q"))

    # 動態表頭
    header = f"  {'長度':>10s}  {'繁問繁答':>8s}"
    sep = f"  {'─' * 10}  {'─' * 8}"
    if has_simp:
        header += f"  {'繁問簡答':>8s}"
        sep += f"  {'─' * 8}"
    if has_simp_q:
        header += f"  {'簡問簡答':>8s}"
        sep += f"  {'─' * 8}"
    header += f"  {'繁-簡q':>8s}" if has_simp_q else (f"  {'差異':>8s}" if has_simp else "")
    sep += f"  {'─' * 8}" if (has_simp_q or has_simp) else ""
    print(header)
    print(sep)

    for length in lengths:
        trad = accuracy.get("traditional", {}).get(length, None)
        simp = accuracy.get("simplified", {}).get(length, None)
        simp_q = accuracy.get("simplified_q", {}).get(length, None)

        trad_str = f"{trad * 100:6.1f}%" if trad is not None else "   N/A"
        row = f"  {length:>10,}  {trad_str}"

        if has_simp:
            simp_str = f"{simp * 100:6.1f}%" if simp is not None else "   N/A"
            row += f"  {simp_str}"

        if has_simp_q:
            simp_q_str = f"{simp_q * 100:6.1f}%" if simp_q is not None else "   N/A"
            row += f"  {simp_q_str}"

        # 差異欄：優先用 trad vs simp_q，否則用 trad vs simp
        ref = simp_q if has_simp_q else (simp if has_simp else None)
        if trad is not None and ref is not None:
            diff = (trad - ref) * 100
            diff_str = f"{diff:+6.1f}%"
        else:
            diff_str = "   N/A"
        if has_simp_q or has_simp:
            row += f"  {diff_str}"

        print(row)


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

    # 5. 衰退係數
    rot_coeff = compute_rot_coefficient(results)
    print(f"\n📉 衰退係數（pp / 1K 字元）")
    for v, rc in rot_coeff.items():
        print(f"  {v}: slope={rc['slope_pp_per_1k_chars']:.3f}, "
              f"R²={rc['r_squared']:.3f}, baseline={rc['baseline_pct']:.1f}%")

    # 6. 斷點偵測
    breakpoints = compute_breakpoint(results)
    print(f"\n📍 斷點位置（首次掉 10pp）")
    for v, bp in breakpoints.items():
        bp_str = f"{bp['breakpoint_drop']:,}" if bp['breakpoint_drop'] else "未觸發"
        print(f"  {v}: {bp_str} 字元（基線={bp['baseline_pct']:.1f}%, "
              f"最長={bp['acc_at_max_length']:.1f}%, 落差={bp['total_drop_pp']:.1f}pp）")

    # 7. Token overhead by length
    overhead_by_len = compute_token_overhead_by_length(results)

    # 8. 各 needle 準確率
    acc_by_needle = compute_accuracy_by_needle(results)

    # 9. 長上下文準確率
    long_ctx_acc = compute_long_context_accuracy(results)

    # 儲存完整分析結果
    analysis = {
        "model": model,
        "total_results": len(results),
        "token_stats": token_stats,
        "accuracy_by_length": acc_by_length,
        "accuracy_by_position": acc_by_position,
        "heatmap": heatmap,
        "rot_coefficient": rot_coeff,
        "breakpoints": breakpoints,
        "token_overhead_by_length": overhead_by_len,
        "accuracy_by_needle": acc_by_needle,
        "long_context_accuracy": long_ctx_acc,
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
        # 掃描 results 目錄下的所有結果檔（h2_ 檔案會自動合併，不另外列出）
        models = []
        for f in os.listdir(RESULTS_DIR):
            if f.endswith("_results.jsonl") and not f.startswith("h2_"):
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

        print(f"\n  {'模型':>15s}  {'繁/簡token比':>12s}  {'繁問繁答@max':>14s}  {'簡問簡答@max':>14s}  {'差距':>8s}")
        print(f"  {'─' * 15}  {'─' * 12}  {'─' * 14}  {'─' * 14}  {'─' * 8}")

        for model, analysis in all_analyses.items():
            ratio = analysis["token_stats"].get("avg_ratio_trad_over_simp", "N/A")
            acc_trad = analysis["accuracy_by_length"].get("traditional", {})
            acc_simp_q = analysis["accuracy_by_length"].get("simplified_q", {})
            acc_simp = analysis["accuracy_by_length"].get("simplified", {})

            # 用 simp_q 優先，沒有則用 simp
            acc_ref = acc_simp_q or acc_simp

            all_keys = list(acc_trad.keys()) + list(acc_ref.keys())
            max_len = max(all_keys, default=0)
            trad_at_max = acc_trad.get(max_len, None)
            ref_at_max = acc_ref.get(max_len, None)

            trad_str = f"{trad_at_max*100:.1f}%" if trad_at_max else "N/A"
            ref_str = f"{ref_at_max*100:.1f}%" if ref_at_max else "N/A"

            if trad_at_max is not None and ref_at_max is not None:
                diff = (trad_at_max - ref_at_max) * 100
                diff_str = f"{diff:+.1f}%"
            else:
                diff_str = "N/A"

            print(f"  {model:>15s}  {str(ratio):>12s}  {trad_str:>14s}  {ref_str:>14s}  {diff_str:>8s}")


if __name__ == "__main__":
    main()
