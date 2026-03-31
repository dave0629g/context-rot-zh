#!/bin/bash
# 實驗進度監控腳本
# 用法: watch --color -n 5 bash scripts/watch_progress.sh

python3 - <<'PYEOF'
import json, os, subprocess, unicodedata
from datetime import datetime

# ANSI 顏色（dark theme）
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
DIM    = "\033[90m"
RESET  = "\033[0m"

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

def scan_variant(jsonl_path, variant):
    done = 0
    skipped = 0
    total_sec = 0.0
    last = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    if r.get("variant") != variant: continue
                    if r.get("skipped"):
                        skipped += 1
                        continue
                    done += 1
                    total_sec += r.get("elapsed_seconds", 0)
                    last = r
                except: pass
    except:
        pass
    # 最後寫入時間（用檔案 mtime）
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(jsonl_path)).strftime("%H:%M:%S")
    except:
        mtime = "—"

    if last:
        return done, skipped, total_sec, str(last["experiment_id"]), f"{last['context_length_chars']:,}", f"{last['elapsed_seconds']:.1f}s", mtime
    return done, skipped, 0, "—", "—", "—", "—"

def fmt_time(sec):
    if sec <= 0: return "—"
    s = int(sec)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def is_running(model, variant_key):
    try:
        lines = subprocess.check_output(["ps", "aux"], text=True).splitlines()
        for line in lines:
            if model not in line: continue
            if variant_key == "simplified_q":
                if "06_hypothesis2" in line: return True
            elif variant_key == "traditional":
                if "03_run_experiment" in line and "--variant traditional" in line: return True
                if "03_run_experiment" in line and "--variant" not in line: return True
            elif variant_key == "simplified":
                if "03_run_experiment" in line and "--variant simplified" in line: return True
        return False
    except:
        return False

TOTAL = 1320

MODELS = [
    {"model": "gemma3:4b",    "size": "4B"},
    {"model": "llama3.1:8b",  "size": "8B"},
    {"model": "qwen3:8b",     "size": "8B"},
    {"model": "qwen3.5:35b",  "size": "35B"},
    {"model": "gemma3:27b",   "size": "27B"},
    {"model": "llama3.3:70b", "size": "70B"},
]

VARIANT_KEYS = [
    ("繁問繁答", "traditional"),
    ("簡問簡答", "simplified_q"),
    ("繁問簡答", "simplified"),
]

# ── 掃描結果 ─────────────────────────────────────────────────────
for m in MODELS:
    path    = f"results/{m['model']}_results.jsonl"
    path_sq = f"results/h2_{m['model']}_results.jsonl"
    m["paths"] = {"traditional": path, "simplified_q": path_sq, "simplified": path}
    m["data"] = {}
    for label, vk in VARIANT_KEYS:
        p = path_sq if vk == "simplified_q" else path
        d, s, t, lid, llen, lt, mtime = scan_variant(p, vk)
        m["data"][vk] = {
            "done": d, "skip": s, "sec": t,
            "last": (lid, llen, lt),
            "mtime": mtime,
            "running": is_running(m["model"], vk),
        }

def fmt(done, skip):
    total = done + skip
    if total == 0: return "—"
    if skip > 0: return f"{done}+{skip}s/{TOTAL}"
    return f"{done}/{TOTAL}"

def color_for(d):
    processed = d["done"] + d["skip"]
    if processed >= TOTAL: return GREEN
    if d["done"] == 0 and d["skip"] == 0: return DIM
    if d["running"]: return RED
    return YELLOW

# ── 輸出 ─────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"  更新時間：{now}")
print()

C = [16, 5, 16, 8, 16, 8, 16, 8]
headers = ["模型", "大小", "繁問繁答", "GPU時間", "簡問簡答", "GPU時間", "繁問簡答", "GPU時間"]
HDR = "  " + "  ".join(wljust(h, c) for h, c in zip(headers, C))
SEP = "  " + "  ".join("─" * c for c in C)
print(HDR)
print(SEP)

for m in MODELS:
    parts = [wljust(m["model"], C[0]), wljust(m["size"], C[1])]
    for i, (label, vk) in enumerate(VARIANT_KEYS):
        d = m["data"][vk]
        col = color_for(d)
        val = fmt(d["done"], d["skip"])
        sec = fmt_time(d["sec"])
        parts.append(f"{col}{wrjust(val, C[2+i*2])}{RESET}")
        parts.append(f"{col}{wrjust(sec, C[3+i*2])}{RESET}")
    print("  " + "  ".join(parts))

# ── 進行中 / 暫停 ───────────────────────────────────────────────
active = []
for m in MODELS:
    for label, vk in VARIANT_KEYS:
        d = m["data"][vk]
        processed = d["done"] + d["skip"]
        if 0 < processed < TOTAL:
            lid, llen, lt = d["last"]
            skip_info = f"+{d['skip']}s" if d["skip"] > 0 else ""
            tag = f"{RED}▶{RESET}" if d["running"] else f"{YELLOW}⏸{RESET}"
            time_str = f"@ {d['mtime']}" if d["mtime"] != "—" else ""
            active.append(f"    {tag} {m['model']} {label}: {d['done']}{skip_info}/{TOTAL}  last id={lid} len={llen} ({lt}) {time_str}")

if active:
    print()
    print("  ▌ 進行中 / 暫停")
    for a in active:
        print(a)

print()
PYEOF
