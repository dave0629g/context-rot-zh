#!/bin/bash
# 實驗進度監控腳本
# 用法: watch -n 5 bash scripts/watch_progress.sh

python3 - <<'PYEOF'
import json, os
from datetime import datetime

def elapsed_str(start_str, done, total, jsonl_path):
    if not start_str or start_str == "—":
        return "—"
    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    if done >= total:
        try:
            end_ts = os.path.getmtime(jsonl_path)
            end = datetime.fromtimestamp(end_ts)
        except:
            end = datetime.now()
    else:
        end = datetime.now()
    s = int((end - start).total_seconds())
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def count_variant(jsonl_path, variant):
    try:
        n = 0
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    if r.get("variant") == variant and not r.get("skipped"):
                        n += 1
                except: pass
        return n
    except:
        return 0

def last_experiment(jsonl_path, variant=None):
    try:
        with open(jsonl_path) as f:
            lines = [l for l in f if l.strip()]
        candidates = []
        for l in reversed(lines):
            try:
                r = json.loads(l)
                if variant is None or r.get("variant") == variant:
                    candidates.append(r)
                    break
            except: pass
        if not candidates:
            return "—", "—", "—"
        r = candidates[0]
        return str(r["experiment_id"]), f"{r['context_length_chars']:,}", f"{r['elapsed_seconds']:.1f}s"
    except:
        return "—", "—", "—"

TOTAL = 1100  # 每個 variant 各 1100 筆

# ── 模型清單 ─────────────────────────────────────────────────────
MODELS = [
    {"model": "gemma3:4b",    "size": "4B",
     "trad_start": "2026-03-26 08:40:10",
     "sq_start":   "2026-03-29 11:05:00",
     "simp_start":  "2026-03-26 08:40:10"},
    {"model": "llama3.1:8b",  "size": "8B",
     "trad_start": "2026-03-26 12:16:07",
     "sq_start":   "",
     "simp_start":  "2026-03-26 12:16:07"},
    {"model": "qwen3:8b",     "size": "8B",
     "trad_start": "2026-03-29 09:43:00",
     "sq_start":   "",
     "simp_start":  "2026-03-29 09:43:00"},
    {"model": "qwen3.5:35b",  "size": "35B",
     "trad_start": "", "sq_start": "", "simp_start": ""},
    {"model": "gemma3:27b",   "size": "27B",
     "trad_start": "", "sq_start": "", "simp_start": ""},
    {"model": "llama3.3:70b", "size": "70B",
     "trad_start": "", "sq_start": "", "simp_start": ""},
]

# ── 計算進度 ─────────────────────────────────────────────────────
for m in MODELS:
    path      = f"results/{m['model']}_results.jsonl"
    path_sq   = f"results/h2_{m['model']}_results.jsonl"
    m["path"]      = path
    m["path_sq"]   = path_sq
    m["trad_done"] = count_variant(path, "traditional")
    m["sq_done"]   = count_variant(path_sq, "simplified_q")
    m["simp_done"] = count_variant(path, "simplified")

def status(done, total, start):
    if done >= total: return "✅ 完成"
    if done > 0 or start: return "🔄 進行中"
    return "⏳ 待跑"

def fmt(n): return f"{n}/{TOTAL}" if n > 0 else "—"

# ── 輸出 ─────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"  更新時間：{now}")
print()

HDR = (f"  {'模型':<14}  {'大小':<5}  "
       f"{'繁問繁答':>10}  {'時長':>8}  "
       f"{'簡問簡答':>10}  {'時長':>8}  "
       f"{'繁問簡答':>10}  {'時長':>8}")
SEP = (f"  {'─'*14}  {'─'*5}  "
       f"{'─'*10}  {'─'*8}  "
       f"{'─'*10}  {'─'*8}  "
       f"{'─'*10}  {'─'*8}")
print(HDR)
print(SEP)

for m in MODELS:
    trad_dur = elapsed_str(m["trad_start"], m["trad_done"], TOTAL, m["path"]) if m["trad_start"] else "—"
    sq_dur   = elapsed_str(m["sq_start"],   m["sq_done"],   TOTAL, m["path_sq"]) if m["sq_start"] else "—"
    simp_dur = elapsed_str(m["simp_start"], m["simp_done"], TOTAL, m["path"]) if m["simp_start"] else "—"

    print(f"  {m['model']:<14}  {m['size']:<5}  "
          f"{fmt(m['trad_done']):>10}  {trad_dur:>8}  "
          f"{fmt(m['sq_done']):>10}  {sq_dur:>8}  "
          f"{fmt(m['simp_done']):>10}  {simp_dur:>8}")

# ── 正在執行摘要 ─────────────────────────────────────────────────
print()
print("  ▌ 最後完成")
for m in MODELS:
    for variant, path, label in [
        ("traditional", m["path"],    "繁問繁答"),
        ("simplified_q", m["path_sq"], "簡問簡答"),
        ("simplified",  m["path"],    "繁問簡答"),
    ]:
        done = m["trad_done"] if variant == "traditional" else (m["sq_done"] if variant == "simplified_q" else m["simp_done"])
        if 0 < done < TOTAL:
            eid, elen, et = last_experiment(path, variant)
            print(f"    {m['model']} {label}: {done}/{TOTAL}  last id={eid} len={elen} ({et})")
print()
PYEOF
