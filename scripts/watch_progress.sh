#!/bin/bash
# 實驗進度監控腳本
# 用法: watch -n 5 bash scripts/watch_progress.sh

python3 - <<'PYEOF'
import json, os
from datetime import datetime

def elapsed_str(start_str):
    if not start_str or start_str == "—":
        return "—"
    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    s = int((datetime.now() - start).total_seconds())
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def last_experiment(jsonl_path):
    try:
        with open(jsonl_path) as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            return "—", "—", "—"
        r = json.loads(lines[-1])
        exp_id  = str(r["experiment_id"])
        length  = f"{r['context_length_chars']:,}"
        elapsed = f"{r['elapsed_seconds']:.1f}s"
        return exp_id, length, elapsed
    except:
        return "—", "—", "—"

def count_done(jsonl_path):
    try:
        with open(jsonl_path) as f:
            return sum(1 for l in f if l.strip())
    except:
        return 0

# ── 實驗清單（依序更新狀態和開始時間）──────────────────────────
EXPERIMENTS = [
    {"no": 1, "model": "gemma3:4b",    "size": "4B",  "start": "2026-03-26 08:40:10"},
    {"no": 2, "model": "llama3.1:8b",  "size": "8B",  "start": "2026-03-26 12:16:07"},
    {"no": 3, "model": "qwen3:8b",     "size": "8B",  "start": "2026-03-26 22:08:18"},
    {"no": 4, "model": "qwen3.5:35b",  "size": "35B", "start": "2026-03-28 09:37:24"},
    {"no": 5, "model": "gemma3:27b",   "size": "27B", "start": ""},
    {"no": 6, "model": "llama3.3:70b", "size": "70B", "start": ""},
]
TOTAL = 2200

# ── 判斷各模型狀態 ──────────────────────────────────────────────
for exp in EXPERIMENTS:
    path = f"results/{exp['model']}_results.jsonl"
    done = count_done(path)
    exp["done"] = done
    if done >= TOTAL:
        exp["status"] = "✅ 完成"
    elif done > 0 or (exp["start"] and exp["start"] != ""):
        exp["status"] = "🔄 進行中"
    else:
        exp["status"] = "⏳ 待跑"
    exp_id, length, exp_elapsed = last_experiment(path)
    exp["last_id"] = exp_id
    exp["last_len"] = length
    exp["last_t"] = exp_elapsed

# ── 輸出表格 ────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"  更新時間：{now}")
print()
print(f"  {'#':<3}  {'模型':<14}  {'大小':<5}  {'狀態':<10}  {'開始時間':<19}  {'已執行':>8}  {'進度':>12}  最後完成實驗")
print(f"  {'─'*3}  {'─'*14}  {'─'*5}  {'─'*10}  {'─'*19}  {'─'*8}  {'─'*12}  {'─'*28}")
for exp in EXPERIMENTS:
    start  = exp["start"] or "—"
    dur    = elapsed_str(exp["start"]) if exp["start"] else "—"
    done   = exp["done"]
    prog   = f"{done}/{TOTAL}" if done > 0 else "—"
    last   = f"id={exp['last_id']} len={exp['last_len']} ({exp['last_t']})" if exp["last_id"] != "—" else "—"
    print(f"  {exp['no']:<3}  {exp['model']:<14}  {exp['size']:<5}  {exp['status']:<10}  {start:<19}  {dur:>8}  {prog:>12}  {last}")
print()
PYEOF
