"""
Step 7: 匯出 GitHub Pages 靜態資料

將所有實驗結果聚合為 docs/data.json，供 GitHub Pages 靜態展示使用。
執行後 commit docs/data.json 即可更新靜態頁面。

用法:
  python scripts/07_export_web.py
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_analyze = import_module("04_analyze")
reevaluate = _analyze.reevaluate
compute_rot_coefficient = _analyze.compute_rot_coefficient
compute_breakpoint = _analyze.compute_breakpoint
compute_token_overhead_by_length = _analyze.compute_token_overhead_by_length
compute_long_context_accuracy = _analyze.compute_long_context_accuracy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

MODEL_META = {
    "gemma3:1b":   dict(label="Gemma 3 1B",   family="Gemma 3",      color="#A8C0FF", params="1B",  ctx=131072),
    "gemma3:4b":   dict(label="Gemma 3 4B",   family="Gemma 3",      color="#6690E0", params="4B",  ctx=131072),
    "gemma3:12b":  dict(label="Gemma 3 12B",  family="Gemma 3",      color="#3A62B8", params="12B", ctx=131072),
    "gemma3:27b":  dict(label="Gemma 3 27B",  family="Gemma 3",      color="#1A3A8C", params="27B", ctx=131072),
    "gemma4:e2b":  dict(label="Gemma 4 E2B",  family="Gemma 4 Edge", color="#FFB060", params="E2B", ctx=131072),
    "gemma4:e4b":  dict(label="Gemma 4 E4B",  family="Gemma 4 Edge", color="#D07010", params="E4B", ctx=131072),
    "gemma4:26b":  dict(label="Gemma 4 26B",  family="Gemma 4",      color="#E06040", params="26B", ctx=262144),
    "gemma4:31b":  dict(label="Gemma 4 31B",  family="Gemma 4",      color="#8B2500", params="31B", ctx=262144),
    "llama3.1:8b": dict(label="Llama 3.1 8B", family="Llama",        color="#E07070", params="8B",  ctx=131072),
    "llama3.3:70b":dict(label="Llama 3.3 70B",family="Llama",        color="#8B0000", params="70B", ctx=131072),
    "qwen3:8b":    dict(label="Qwen3 8B",     family="Qwen3",        color="#2E8B4A", params="8B",  ctx=40960),
    "qwen3.5:2b":  dict(label="Qwen3.5 2B",  family="Qwen3.5",       color="#C09AE8", params="2B",  ctx=262144),
    "qwen3.5:4b":  dict(label="Qwen3.5 4B",  family="Qwen3.5",       color="#A070D0", params="4B",  ctx=262144),
    "qwen3.5:9b":  dict(label="Qwen3.5 9B",  family="Qwen3.5",       color="#7840B8", params="9B",  ctx=262144),
    "qwen3.5:27b": dict(label="Qwen3.5 27B", family="Qwen3.5",       color="#5010A0", params="27B", ctx=262144),
    "qwen3.5:35b": dict(label="Qwen3.5 35B", family="Qwen3.5",       color="#2E0060", params="35B", ctx=262144),
}

# (file_prefix, variant_key_in_jsonl) → 顯示名稱
VARIANT_SOURCES = [
    ("",    "traditional",  "繁問繁答"),
    ("",    "simplified",   "繁問簡答"),
    ("h2_", "simplified_q", "簡問簡答"),
]


def load_model_records(model: str) -> dict:
    """載入模型所有 variant 的記錄，回傳 {variant_label: [records]}"""
    data = defaultdict(list)
    for prefix, vkey, label in VARIANT_SOURCES:
        fname = f"{prefix}{model}_results.jsonl"
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("skipped") or r.get("variant") != vkey:
                        continue
                    r["_correct"] = reevaluate(r)
                    data[label].append(r)
                except Exception:
                    pass
    return dict(data)


def aggregate_variant(records: list) -> dict:
    """將一個 variant 的記錄聚合為統計數據"""
    def acc_by(key_fn):
        d = defaultdict(lambda: [0, 0])
        for r in records:
            k = key_fn(r)
            d[k][1] += 1
            d[k][0] += int(r["_correct"])
        return {str(k): round(c / t * 100, 1) for k, (c, t) in d.items() if t > 0}

    # 熱力圖：{str(length): {str(position): pct}}
    hm = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        l, p = str(r["context_length_chars"]), str(r["needle_position"])
        hm[l][p][1] += 1
        hm[l][p][0] += int(r["_correct"])
    heatmap = {
        l: {p: round(c / t * 100, 1) for p, (c, t) in pos.items() if t > 0}
        for l, pos in hm.items()
    }

    return {
        "by_length":   acc_by(lambda r: r["context_length_chars"]),
        "by_position": acc_by(lambda r: r["needle_position"]),
        "by_needle":   acc_by(lambda r: r["needle_id"]),
        "heatmap":     heatmap,
    }


def compute_token_overhead(trad: list, simp: list) -> float | None:
    """計算繁體比簡體多用的 token 百分比"""
    t_map = {r["experiment_id"]: r.get("token_count_prompt", 0) for r in trad}
    s_map = {r["experiment_id"]: r.get("token_count_prompt", 0) for r in simp}
    pairs = [
        (t_map[e], s_map[e])
        for e in set(t_map) & set(s_map)
        if t_map[e] > 0 and s_map[e] > 0
    ]
    if not pairs:
        return None
    avg = sum(t / s for t, s in pairs) / len(pairs)
    return round((avg - 1) * 100, 2)


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "models": {},
        "data": {},
    }

    # 掃描所有非 h2_ 的結果檔
    model_ids = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_results.jsonl") and not fname.startswith("h2_"):
            model = fname.replace("_results.jsonl", "")
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.getsize(fpath) > 0:
                model_ids.append(model)

    total_records = 0
    for model in model_ids:
        print(f"處理 {model}...", end=" ")
        model_data = load_model_records(model)
        if not model_data:
            print("（無資料，跳過）")
            continue

        meta = MODEL_META.get(model, {})
        rec_counts = {v: len(r) for v, r in model_data.items()}
        total = sum(rec_counts.values())
        total_records += total
        print(f"{total} 筆 | variants: {list(model_data.keys())}")

        output["models"][model] = {
            **{k: meta.get(k) for k in ("label", "family", "color", "params", "ctx")},
            "available_variants": list(model_data.keys()),
            "record_counts": rec_counts,
        }

        output["data"][model] = {
            v: aggregate_variant(recs) for v, recs in model_data.items()
        }

        # Tokenizer overhead（繁問繁答 vs 繁問簡答）
        overhead = compute_token_overhead(
            model_data.get("繁問繁答", []),
            model_data.get("繁問簡答", []),
        )
        output["data"][model]["token_overhead_pct"] = overhead

        # ── 深度分析指標 ──
        # 將記錄轉回 04_analyze 格式（需要 variant + evaluation.is_correct）
        LABEL_TO_VARIANT = {"繁問繁答": "traditional", "繁問簡答": "simplified", "簡問簡答": "simplified_q"}
        flat_records = []
        for label, recs in model_data.items():
            vkey = LABEL_TO_VARIANT.get(label)
            if not vkey:
                continue
            for r in recs:
                r["variant"] = vkey
                r["evaluation"] = {"is_correct": r["_correct"]}
                flat_records.append(r)

        rot = compute_rot_coefficient(flat_records)
        bps = compute_breakpoint(flat_records)
        long_acc = compute_long_context_accuracy(flat_records)
        overhead_by_len = compute_token_overhead_by_length(flat_records)

        VARIANT_TO_LABEL = {v: k for k, v in LABEL_TO_VARIANT.items()}
        output["data"][model]["rot_coefficient"] = {
            VARIANT_TO_LABEL.get(v, v): {
                "slope": rc["slope_pp_per_1k_chars"],
                "r_squared": rc["r_squared"],
                "baseline_pct": rc["baseline_pct"],
            }
            for v, rc in rot.items()
        }
        output["data"][model]["breakpoints"] = {
            VARIANT_TO_LABEL.get(v, v): {
                "breakpoint_drop": bp["breakpoint_drop"],
                "breakpoint_below_80": bp["breakpoint_below_80"],
                "baseline_pct": bp["baseline_pct"],
                "acc_at_max_length": bp["acc_at_max_length"],
                "total_drop_pp": bp["total_drop_pp"],
            }
            for v, bp in bps.items()
        }
        output["data"][model]["long_context_accuracy"] = {
            VARIANT_TO_LABEL.get(v, v): round(a * 100, 2) if a else None
            for v, a in long_acc.items()
        }
        if overhead_by_len:
            output["data"][model]["token_overhead_by_length"] = {
                str(l): d["overhead_pct"] for l, d in overhead_by_len.items()
            }

    # ── 跨模型比較摘要 ──
    cross_model = []
    for model_id in model_ids:
        d = output["data"].get(model_id)
        if not d:
            continue
        meta = MODEL_META.get(model_id, {})
        rot_trad = (d.get("rot_coefficient") or {}).get("繁問繁答", {})
        rot_simp = (d.get("rot_coefficient") or {}).get("簡問簡答", {})
        bp_trad = (d.get("breakpoints") or {}).get("繁問繁答", {})
        bp_simp = (d.get("breakpoints") or {}).get("簡問簡答", {})
        cross_model.append({
            "model": model_id,
            "label": meta.get("label", model_id),
            "family": meta.get("family", ""),
            "params": meta.get("params", ""),
            "rot_slope_trad": rot_trad.get("slope"),
            "rot_slope_simp": rot_simp.get("slope"),
            "breakpoint_trad": bp_trad.get("breakpoint_drop"),
            "breakpoint_simp": bp_simp.get("breakpoint_drop"),
            "total_drop_trad": bp_trad.get("total_drop_pp"),
            "total_drop_simp": bp_simp.get("total_drop_pp"),
            "token_overhead_pct": d.get("token_overhead_pct"),
        })
    output["cross_model_summary"] = cross_model

    out_path = os.path.join(DOCS_DIR, "data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n✓ 已輸出：{out_path}")
    print(f"  模型數：{len(output['models'])}，總記錄數：{total_records:,}，檔案大小：{size_kb:.1f} KB")


if __name__ == "__main__":
    main()
