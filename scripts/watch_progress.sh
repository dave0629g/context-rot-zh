#!/bin/bash
# 實驗進度監控腳本
# 用法: watch -n 5 bash scripts/watch_progress.sh

python3 - <<'PYEOF'
import json, os, unicodedata
from datetime import datetime

def wlen(s):
    """計算字串的終端顯示寬度（中文字算 2）"""
    w = 0
    for c in s:
        ea = unicodedata.east_asian_width(c)
        w += 2 if ea in ('W', 'F') else 1
    return w

def wljust(s, width):
    """左對齊，依顯示寬度補空格"""
    return s + ' ' * max(0, width - wlen(s))

def wrjust(s, width):
    """右對齊，依顯示寬度補空格"""
    return ' ' * max(0, width - wlen(s)) + s

def elapsed_str(start_str, done, total, jsonl_path):
    if not start_str or start_str == "—":
        return "—"
    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    if done == 0:
        # 尚未有任何結果：用 now（真正剛開始跑）
        end = datetime.now()
    else:
        # 有結果（不論完成或暫停）：用檔案最後修改時間，不再隨 now 累計
        try:
            end = datetime.fromtimestamp(os.path.getmtime(jsonl_path))
        except:
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
     "sq_start":   "2026-03-29 14:28:30",
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

C = [16, 5, 10, 8, 10, 8, 10, 8]  # 各欄顯示寬度
headers = ["模型", "大小", "繁問繁答", "時長", "簡問簡答", "時長", "繁問簡答", "時長"]
HDR = "  " + "  ".join(wljust(h, c) for h, c in zip(headers, C))
SEP = "  " + "  ".join("─" * c for c in C)
print(HDR)
print(SEP)

for m in MODELS:
    trad_dur = elapsed_str(m["trad_start"], m["trad_done"], TOTAL, m["path"]) if m["trad_start"] else "—"
    sq_dur   = elapsed_str(m["sq_start"],   m["sq_done"],   TOTAL, m["path_sq"]) if m["sq_start"] else "—"
    simp_dur = elapsed_str(m["simp_start"], m["simp_done"], TOTAL, m["path"]) if m["simp_start"] else "—"

    cols = [m["model"], m["size"],
            fmt(m["trad_done"]), trad_dur,
            fmt(m["sq_done"]),   sq_dur,
            fmt(m["simp_done"]), simp_dur]
    print("  " + "  ".join(wrjust(v, c) if i >= 2 else wljust(v, c)
                            for i, (v, c) in enumerate(zip(cols, C))))

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
