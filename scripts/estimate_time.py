"""
估算所有實驗的剩餘時間

根據已完成實驗的 elapsed_seconds 統計，推算每個模型每個 variant 的：
  - 已完成筆數 / 總筆數
  - 已花費 GPU 時間
  - 預估剩餘時間
  - 預估總時間

用法: python scripts/estimate_time.py
"""

import json
import os
import unicodedata
from collections import defaultdict


def wlen(s):
    w = 0
    for c in s:
        ea = unicodedata.east_asian_width(c)
        w += 2 if ea in ('W', 'F') else 1
    return w


def wljust(s, width):
    return s + ' ' * max(0, width - wlen(s))


def wrjust(s, width):
    return ' ' * max(0, width - wlen(s)) + s

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOTAL = 1320  # 12 長度 × 11 位置 × 10 trials

# ANSI 顏色（dark theme）
GREEN  = "\033[32m"   # 完成
RED    = "\033[31m"   # 進行中
YELLOW = "\033[33m"   # 暫停
DIM    = "\033[90m"   # 未開始（灰色）
RESET  = "\033[0m"


def is_running(model: str, variant_key: str) -> bool:
    """檢查是否有對應的實驗程序正在跑（精確比對 model + variant）"""
    import subprocess
    try:
        lines = subprocess.check_output(["ps", "aux"], text=True).splitlines()
        for line in lines:
            if model not in line:
                continue
            if variant_key == "simplified_q":
                if "06_hypothesis2" in line:
                    return True
            elif variant_key == "traditional":
                if "03_run_experiment" in line and "--variant traditional" in line:
                    return True
                # --variant both 或無 --variant 時也算 traditional
                if "03_run_experiment" in line and "--variant" not in line:
                    return True
            elif variant_key == "simplified":
                if "03_run_experiment" in line and "--variant simplified" in line:
                    return True
        return False
    except:
        return False

MODELS = [
    "gemma3:4b", "llama3.1:8b", "qwen3:8b",
    "qwen3.5:35b", "gemma3:27b", "llama3.3:70b",
]

VARIANTS = [
    ("繁問繁答", "traditional",   lambda m: f"results/{m}_results.jsonl"),
    ("簡問簡答", "simplified_q",  lambda m: f"results/h2_{m}_results.jsonl"),
    ("繁問簡答", "simplified",    lambda m: f"results/{m}_results.jsonl"),
]

# 各 context 長度的筆數（11 positions × 10 trials = 110 筆/長度）
LENGTHS = [500, 2000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 65000, 100000, 130000]
PER_LENGTH = 110  # 11 × 10


def scan_results(path, variant_key):
    """掃描結果檔，回傳 (by_len, skipped_lengths)
    by_len: {context_length: [elapsed_seconds, ...]}
    skipped_lengths: set of lengths that were skipped"""
    by_len = defaultdict(list)
    skipped_lengths = set()
    if not os.path.exists(path):
        return by_len, skipped_lengths
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("variant") != variant_key:
                    continue
                if r.get("skipped"):
                    skipped_lengths.add(r["context_length_chars"])
                    continue
                by_len[r["context_length_chars"]].append(r["elapsed_seconds"])
            except:
                pass
    return by_len, skipped_lengths


def fmt_time(sec):
    if sec <= 0:
        return "—"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def estimate_missing_length(known_avgs, target_length):
    """用已知長度的平均時間，線性外插估算未知長度的耗時"""
    if not known_avgs:
        return None
    # 用最接近的兩個已知長度做線性外插
    lengths = sorted(known_avgs.keys())
    if len(lengths) == 1:
        # 只有一個已知：按比例縮放
        l0 = lengths[0]
        return known_avgs[l0] * (target_length / l0)
    # 找最接近的兩個
    for i in range(len(lengths) - 1):
        if lengths[i] <= target_length <= lengths[i + 1]:
            l0, l1 = lengths[i], lengths[i + 1]
            t0, t1 = known_avgs[l0], known_avgs[l1]
            ratio = (target_length - l0) / (l1 - l0)
            return t0 + (t1 - t0) * ratio
    # 超出範圍：用最後兩個外插
    l0, l1 = lengths[-2], lengths[-1]
    t0, t1 = known_avgs[l0], known_avgs[l1]
    slope = (t1 - t0) / (l1 - l0)
    return t1 + slope * (target_length - l1)


def main():
    # 收集所有已知的耗時資料（跨模型、跨 variant 共用同一個模型的資料）
    model_data = {}
    for model in MODELS:
        model_data[model] = {}
        for label, vk, path_fn in VARIANTS:
            path = os.path.join(os.path.dirname(__file__), "..", path_fn(model))
            by_len, skipped = scan_results(path, vk)
            model_data[model][label] = {"by_len": by_len, "skipped": skipped}

    # 找同系列模型的參考資料（用於估算未開始的模型）
    # gemma3:4b → gemma3:27b (按參數量比例)
    # llama3.1:8b → llama3.3:70b
    # qwen3:8b → qwen3.5:35b
    SIZE_RATIO = {
        "qwen3.5:35b": ("qwen3:8b", 35 / 8),
        "gemma3:27b":  ("gemma3:4b", 27 / 4),
        "llama3.3:70b": ("llama3.1:8b", 70 / 8),
    }

    # 欄位寬度
    C = [10, 16, 8, 8, 8]
    headers = ["variant", "完成", "已花費", "剩餘估計", "總估計"]

    print("═" * 62)
    print("  實驗時間估算")
    print("═" * 62)

    grand_done_sec = 0
    grand_remain_sec = 0

    def print_header():
        hdr = "    " + "  ".join(wljust(h, c) if i == 0 else wrjust(h, c) for i, (h, c) in enumerate(zip(headers, C)))
        sep = "    " + "  ".join("─" * c for c in C)
        print(hdr)
        print(sep)

    for model in MODELS:
        print(f"\n  ▌ {model}")
        print_header()

        for label, vk, path_fn in VARIANTS:
            by_len = model_data[model][label]["by_len"]
            skipped_lengths = model_data[model][label]["skipped"]
            done = sum(len(v) for v in by_len.values())
            skipped = sum(PER_LENGTH for l in skipped_lengths if l not in by_len)
            done_sec = sum(sum(v) for v in by_len.values())
            grand_done_sec += done_sec

            # 計算每個長度的平均耗時
            known_avgs = {l: sum(v) / len(v) for l, v in by_len.items() if v}

            # 估算剩餘（排除已知 SKIP 的長度）
            remain_sec = 0
            remain_count = 0
            for length in LENGTHS:
                if length in skipped_lengths:
                    continue  # 此長度已知會 SKIP，不計入剩餘時間
                done_at_len = len(by_len.get(length, []))
                remain_at_len = PER_LENGTH - done_at_len
                if remain_at_len <= 0:
                    continue
                remain_count += remain_at_len

                avg = None
                if length in known_avgs:
                    avg = known_avgs[length]
                elif known_avgs:
                    est = estimate_missing_length(known_avgs, length)
                    if est and est > 0:
                        avg = est

                # 用參考模型估算
                if avg is None and model in SIZE_RATIO:
                    ref_model, ratio = SIZE_RATIO[model]
                    ref_data = model_data[ref_model]
                    ref_by_len = ref_data.get(label, {}).get("by_len", {})
                    ref_avgs = {l: sum(v) / len(v) for l, v in ref_by_len.items() if v}
                    if length in ref_avgs:
                        avg = ref_avgs[length] * ratio
                    elif ref_avgs:
                        est = estimate_missing_length(ref_avgs, length)
                        if est and est > 0:
                            avg = est * ratio

                if avg is None or avg <= 0:
                    avg = 60  # 完全未知，用 60s 預設

                remain_sec += remain_at_len * max(avg, 0.1)

            grand_remain_sec += remain_sec
            total_sec = done_sec + remain_sec
            processed = done + skipped

            # 狀態顏色
            if processed >= TOTAL:
                color = GREEN   # 完成
            elif done == 0 and skipped == 0:
                color = DIM     # 未開始（灰色）
            elif is_running(model, vk):
                color = RED     # 進行中
            else:
                color = YELLOW  # 暫停

            # 完成欄：顯示 skip 數量
            if skipped > 0:
                done_str = f"{done}+{skipped}s/{TOTAL}"
            else:
                done_str = f"{done}/{TOTAL}"

            cols = [label, done_str, fmt_time(done_sec),
                    fmt_time(remain_sec), fmt_time(total_sec)]
            line = "    " + "  ".join(
                wljust(v, c) if i == 0 else wrjust(v, c)
                for i, (v, c) in enumerate(zip(cols, C)))
            print(f"{color}{line}{RESET}")

    print()
    print("─" * 62)
    print(f"  合計已花費: {fmt_time(grand_done_sec)}  "
          f"剩餘估計: {fmt_time(grand_remain_sec)}  "
          f"總估計: {fmt_time(grand_done_sec + grand_remain_sec)}")
    print()


if __name__ == "__main__":
    main()
