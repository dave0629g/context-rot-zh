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
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOTAL = 1320  # 12 長度 × 11 位置 × 10 trials

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
    """掃描結果檔，回傳 {context_length: [elapsed_seconds, ...]}"""
    by_len = defaultdict(list)
    if not os.path.exists(path):
        return by_len
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("variant") != variant_key or r.get("skipped"):
                    continue
                by_len[r["context_length_chars"]].append(r["elapsed_seconds"])
            except:
                pass
    return by_len


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
            model_data[model][label] = scan_results(path, vk)

    # 找同系列模型的參考資料（用於估算未開始的模型）
    # gemma3:4b → gemma3:27b (按參數量比例)
    # llama3.1:8b → llama3.3:70b
    # qwen3:8b → qwen3.5:35b
    SIZE_RATIO = {
        "qwen3.5:35b": ("qwen3:8b", 35 / 8),
        "gemma3:27b":  ("gemma3:4b", 27 / 4),
        "llama3.3:70b": ("llama3.1:8b", 70 / 8),
    }

    print("═" * 78)
    print("  實驗時間估算")
    print("═" * 78)

    grand_done_sec = 0
    grand_remain_sec = 0

    for model in MODELS:
        print(f"\n  ▌ {model}")
        print(f"    {'variant':<10}  {'完成':>12}  {'已花費':>8}  {'剩餘估計':>8}  {'總估計':>8}")
        print(f"    {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")

        for label, vk, path_fn in VARIANTS:
            by_len = model_data[model][label]
            done = sum(len(v) for v in by_len.values())
            done_sec = sum(sum(v) for v in by_len.values())
            grand_done_sec += done_sec

            # 計算每個長度的平均耗時
            known_avgs = {l: sum(v) / len(v) for l, v in by_len.items() if v}

            # 估算剩餘
            remain_sec = 0
            for length in LENGTHS:
                done_at_len = len(by_len.get(length, []))
                remain_at_len = PER_LENGTH - done_at_len
                if remain_at_len <= 0:
                    continue

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
                    ref_by_len = ref_data.get(label, {})
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
            remain = TOTAL - done

            print(f"    {label:<10}  {done:>5}/{TOTAL:<5}  {fmt_time(done_sec):>8}  "
                  f"{fmt_time(remain_sec):>8}  {fmt_time(total_sec):>8}")

    print()
    print("─" * 78)
    print(f"  合計已花費: {fmt_time(grand_done_sec)}    "
          f"剩餘估計: {fmt_time(grand_remain_sec)}    "
          f"總估計: {fmt_time(grand_done_sec + grand_remain_sec)}")
    print()


if __name__ == "__main__":
    main()
