#!/usr/bin/env python3
"""比對論文裡關鍵數字 vs b2-stats/ CSV"""
import csv
import os

stats_dir = "b2-stats"

# 論文表 4-3 衰減係數（B2 C 段寫進論文的）
PAPER_DECAY = {
    "gemma3:4b":    -0.46,
    "gemma3:12b":   -0.12,
    "gemma3:27b":   -0.02,
    "llama3.1:8b":  -0.08,
    "llama3.3:70b": -0.12,
}

print("=" * 70)
print("檢查 1：表 4-3 衰減係數 vs rq1_results.csv")
print("=" * 70)
print(f"{'model':20s} {'paper':>10s} {'csv':>10s} {'diff':>10s} {'status':>10s}")
print("-" * 70)

with open(f"../../{stats_dir}/rq1_results.csv") as f:
    for row in csv.DictReader(f):
        m = row.get("model", "")
        if m in PAPER_DECAY:
            csv_val = float(row.get("decay_rate", 0))
            paper_val = PAPER_DECAY[m]
            diff = abs(csv_val - paper_val)
            status = "✅" if diff < 0.01 else "⚠️"
            print(f"{m:20s} {paper_val:>10.4f} {csv_val:>10.4f} {diff:>10.4f} {status:>10s}")

# 論文表 4-16 配對 t 檢定 6 列
PAPER_PAIRED_T = [
    ("gemma3:4b",    65000,  2.32, .024, +0.31),  # 別的 cell 也補進去
    ("gemma3:4b",   100000,  0.00, 1.00,  0.00),
    ("gemma3:4b",   130000,  0.00, 1.00,  0.00),
    ("llama3.1:8b",  65000, -1.66, .103, -0.22),
    ("llama3.1:8b", 100000, -0.30, .766, -0.04),
    ("llama3.1:8b", 130000,  2.32, .024, +0.31),
]

print("\n" + "=" * 70)
print("檢查 2：表 4-16 配對 t vs rq2_results.csv")
print("=" * 70)

# 先看 csv 結構
try:
    with open(f"../../{stats_dir}/rq2_results.csv") as f:
        reader = csv.DictReader(f)
        print(f"rq2_results.csv 欄位: {reader.fieldnames}")
        for i, row in enumerate(reader):
            if i < 10:
                print(f"  row {i+1}: {dict(row)}")
            else:
                break
except FileNotFoundError:
    print("⚠️ rq2_results.csv 不存在於 b2-stats/")

# 論文表 4-19 advance（雙 onset 測量）
PAPER_ADVANCE = {
    "gemma3:4b":    (0, 0),
    "gemma3:12b":   (0, 0),
    "gemma3:27b":   (0, 0),
    "llama3.1:8b":  (35000, 31844),
    "llama3.3:70b": (0, 0),
}

print("\n" + "=" * 70)
print("檢查 3：表 4-19 advance vs rq3_results.csv")
print("=" * 70)
print(f"{'model':20s} {'paper_chars':>12s} {'paper_tokens':>13s} {'csv_chars':>12s} {'csv_tokens':>12s} {'status':>10s}")
print("-" * 90)

try:
    with open(f"../../{stats_dir}/rq3_results.csv") as f:
        for row in csv.DictReader(f):
            m = row.get("model", "")
            if m in PAPER_ADVANCE:
                csv_chars = float(row.get("advance_chars", 0))
                csv_tokens = float(row.get("advance_tokens", 0))
                paper_chars, paper_tokens = PAPER_ADVANCE[m]
                diff_chars = abs(csv_chars - paper_chars)
                diff_tokens = abs(csv_tokens - paper_tokens)
                status = "✅" if diff_chars < 100 and diff_tokens < 100 else "⚠️"
                print(f"{m:20s} {paper_chars:>12.0f} {paper_tokens:>13.0f} {csv_chars:>12.0f} {csv_tokens:>12.0f} {status:>10s}")
except FileNotFoundError:
    print("⚠️ rq3_results.csv 不存在於 b2-stats/")
