#!/bin/bash
# 實驗進度監控腳本
# 用法: watch -n 5 bash scripts/watch_progress.sh

python3 - <<'PYEOF'
import json, os, unicodedata
from datetime import datetime

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
    """回傳 (done, total_seconds, last_id, last_len, last_elapsed)
    結果檔案僅 1-2MB，全檔掃描只需毫秒"""
    done = 0
    total_sec = 0.0
    last = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    if r.get("variant") != variant: continue
                    if r.get("skipped"): continue
                    done += 1
                    total_sec += r.get("elapsed_seconds", 0)
                    last = r
                except: pass
    except:
        pass
    if last:
        return done, total_sec, str(last["experiment_id"]), f"{last['context_length_chars']:,}", f"{last['elapsed_seconds']:.1f}s"
    return done, 0, "—", "—", "—"

def fmt_time(sec):
    if sec <= 0: return "—"
    s = int(sec)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

TOTAL = 1320  # 12 長度 × 11 位置 × 10 trials

MODELS = [
    {"model": "gemma3:4b",    "size": "4B"},
    {"model": "llama3.1:8b",  "size": "8B"},
    {"model": "qwen3:8b",     "size": "8B"},
    {"model": "qwen3.5:35b",  "size": "35B"},
    {"model": "gemma3:27b",   "size": "27B"},
    {"model": "llama3.3:70b", "size": "70B"},
]

# ── 掃描結果 ─────────────────────────────────────────────────────
for m in MODELS:
    path    = f"results/{m['model']}_results.jsonl"
    path_sq = f"results/h2_{m['model']}_results.jsonl"

    td, tt, tl_id, tl_len, tl_t = scan_variant(path, "traditional")
    sd, st, sl_id, sl_len, sl_t = scan_variant(path_sq, "simplified_q")
    xd, xt, xl_id, xl_len, xl_t = scan_variant(path, "simplified")

    m["trad_done"], m["trad_sec"] = td, tt
    m["sq_done"],   m["sq_sec"]   = sd, st
    m["simp_done"], m["simp_sec"] = xd, xt
    m["trad_last"] = (tl_id, tl_len, tl_t)
    m["sq_last"]   = (sl_id, sl_len, sl_t)
    m["simp_last"] = (xl_id, xl_len, xl_t)

def fmt(n): return f"{n}/{TOTAL}" if n > 0 else "—"

# ── 輸出 ─────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"  更新時間：{now}")
print()

C = [16, 5, 12, 8, 12, 8, 12, 8]
headers = ["模型", "大小", "繁問繁答", "GPU時間", "簡問簡答", "GPU時間", "繁問簡答", "GPU時間"]
HDR = "  " + "  ".join(wljust(h, c) for h, c in zip(headers, C))
SEP = "  " + "  ".join("─" * c for c in C)
print(HDR)
print(SEP)

for m in MODELS:
    cols = [m["model"], m["size"],
            fmt(m["trad_done"]), fmt_time(m["trad_sec"]),
            fmt(m["sq_done"]),   fmt_time(m["sq_sec"]),
            fmt(m["simp_done"]), fmt_time(m["simp_sec"])]
    print("  " + "  ".join(wrjust(v, c) if i >= 2 else wljust(v, c)
                            for i, (v, c) in enumerate(zip(cols, C))))

# ── 進行中 / 暫停 ───────────────────────────────────────────────
active = []
for m in MODELS:
    for label, done, last in [
        ("繁問繁答", m["trad_done"], m["trad_last"]),
        ("簡問簡答", m["sq_done"],   m["sq_last"]),
        ("繁問簡答", m["simp_done"], m["simp_last"]),
    ]:
        if 0 < done < TOTAL:
            lid, llen, lt = last
            active.append(f"    {m['model']} {label}: {done}/{TOTAL}  last id={lid} len={llen} ({lt})")

if active:
    print()
    print("  ▌ 進行中 / 暫停")
    for a in active:
        print(a)

print()
PYEOF
