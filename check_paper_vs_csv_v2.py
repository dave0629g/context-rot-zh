#!/usr/bin/env python3
"""論文數字 vs b2-stats/ CSV 校正版（處理欄位名與 CSV 格式問題）"""
import csv
import os

stats_dir = "b2-stats"

# ============================================================
# 檢查 1：表 4-3 衰減係數 vs rq1_results.csv
# ============================================================
PAPER_DECAY = {
    "gemma3:4b":    -0.46,
    "gemma3:12b":   -0.12,
    "gemma3:27b":   -0.02,
    "llama3.1:8b":  -0.08,
    "llama3.3:70b": -0.12,
}

print("=" * 80)
print("檢查 1：表 4-3 衰減係數 (decay_rate) vs rq1_results.csv")
print("=" * 80)

# 先讀 rq1_results.csv 結構
print("rq1_results.csv 結構掃描:")
try:
    with open(f"../../{stats_dir}/rq1_results.csv") as f:
        for i, line in enumerate(f):
            if i < 12:
                print(f"  line {i+1}: {line.rstrip()[:120]}")
            else:
                break
except FileNotFoundError:
    print("⚠️ rq1_results.csv 不存在")
    
print()

# 嘗試直接比對：尋找含 model 名稱的行
print(f"{'model':18s} {'paper':>10s} {'csv_value':>12s} {'diff':>10s} {'status':>8s}")
print("-" * 80)

try:
    with open(f"../../{stats_dir}/rq1_results.csv") as f:
        rows = list(f)
    
    for m, paper_val in PAPER_DECAY.items():
        # 在 rq1_results.csv 找含該模型名的行
        for row in rows:
            if m in row and "decay" not in row.lower():
                # 提取數字
                parts = row.strip().split(",")
                # 找小數值在合理 decay 範圍內 (-1 to 0)
                for p in parts:
                    try:
                        val = float(p.strip())
                        if -2 < val < 0.01 and abs(val) > 0.001:
                            # 候選 decay 值
                            diff = abs(val - paper_val)
                            status = "✅" if diff < 0.05 else "⚠️"
                            print(f"{m:18s} {paper_val:>10.4f} {val:>12.4f} {diff:>10.4f} {status:>8s}")
                            break
                    except ValueError:
                        continue
                break
except FileNotFoundError:
    pass

# ============================================================
# 檢查 3：表 4-19 advance vs rq3_results.csv
# ============================================================
PAPER_ADVANCE = {
    "gemma3:4b":    (0, 0),
    "gemma3:12b":   (0, 0),
    "gemma3:27b":   (0, 0),
    "llama3.1:8b":  (35000, 31844),
    "llama3.3:70b": (0, 0),
}

print("\n" + "=" * 80)
print("檢查 3：表 4-19 advance vs rq3_results.csv")
print("=" * 80)

# 先掃結構
print("rq3_results.csv 結構掃描:")
try:
    with open(f"../../{stats_dir}/rq3_results.csv") as f:
        for i, line in enumerate(f):
            if i < 12:
                print(f"  line {i+1}: {line.rstrip()[:120]}")
            else:
                break
except FileNotFoundError:
    print("⚠️ rq3_results.csv 不存在")

print()
print(f"{'model':18s} {'paper_chars':>12s} {'paper_tokens':>13s} {'csv_full_row':>50s}")
print("-" * 100)

try:
    with open(f"{stats_dir}/rq3_results.csv") as f:
        rows = list(f)
    
    for m, (paper_chars, paper_tokens) in PAPER_ADVANCE.items():
        for row in rows:
            if m in row:
                print(f"{m:18s} {paper_chars:>12.0f} {paper_tokens:>13.0f}  | {row.strip()[:80]}")
                break
except FileNotFoundError:
    pass


# ============================================================
# 檢查 4：表 4-18 LRT 數字 vs rq2_results.csv
# ============================================================
print("\n" + "=" * 80)
print("檢查 4：表 4-18 LRT χ²(1) = 137.69, ΔR² = .030 vs rq2_results.csv")
print("=" * 80)

print("搜尋 rq2_results.csv 中含 'LRT', 'chi', '137' 之行:")
try:
    with open(f"../../{stats_dir}/rq2_results.csv") as f:
        for i, line in enumerate(f):
            if any(kw in line.lower() for kw in ["lrt", "chi", "137", "δ", "delta", "log_lik", "loglik", "nested", "m1", "m2"]):
                print(f"  line {i+1}: {line.rstrip()[:140]}")
except FileNotFoundError:
    print("⚠️ rq2_results.csv 不存在")


# ============================================================
# 檢查 5：fertility (Gemma .742, LLaMA .910) vs rq3_results.csv 或 summary
# ============================================================
print("\n" + "=" * 80)
print("檢查 5：fertility (Gemma .742, LLaMA .910)")
print("=" * 80)

# 搜尋包含 fertility 之行
for fname in ["rq1_results.csv", "rq2_results.csv", "rq3_results.csv"]:
    fpath = f"../../{stats_dir}/{fname}"
    if not os.path.exists(fpath):
        continue
    print(f"\n[{fname}]")
    with open(fpath) as f:
        for i, line in enumerate(f):
            if "fertility" in line.lower() or ".742" in line or ".910" in line or "0.742" in line or "0.910" in line:
                print(f"  line {i+1}: {line.rstrip()[:140]}")
