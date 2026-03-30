"""
假設二驗證：問題語言是否影響簡體 context 下的準確率？

原始實驗設計的簡體 variant：
  - Context：簡體中文
  - Question：繁體中文（未轉換，設計上的疏漏）

本腳本新增 simplified_q variant：
  - Context：簡體中文（與原始相同）
  - Question：簡體中文（OpenCC t2s 轉換）

比較兩者準確率，若 simplified_q > simplified，
表示跨字形搜尋（繁體問題 + 簡體 context）確實造成損耗。

用法:
  python scripts/06_hypothesis2_simp_question.py --model gemma3:4b
  python scripts/06_hypothesis2_simp_question.py --model llama3.1:8b
  python scripts/06_hypothesis2_simp_question.py --model gemma3:4b --compare

輸出:
  results/h2_{model}_results.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

# 共用常數
HAYSTACKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "haystacks", "experiments.jsonl"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OLLAMA_BASE = "http://localhost:11434"
VARIANT = "simplified_q"  # 本腳本的 variant 名稱


# ── OpenCC ────────────────────────────────────────────────────────────────────

def get_converter():
    try:
        import opencc
        return opencc.OpenCC("t2s")
    except ImportError:
        print("錯誤：請先安裝 opencc-python-reimplemented")
        print("  pip install opencc-python-reimplemented")
        sys.exit(1)


# ── Ollama helpers（與 03_run_experiment.py 相同邏輯）─────────────────────────

def get_model_context_length(model: str) -> int:
    url = f"{OLLAMA_BASE}/api/show"
    payload = json.dumps({"name": model}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for key, val in data.get("model_info", {}).items():
            if "context_length" in key:
                return int(val)
    except Exception:
        pass
    return 131072


def is_thinking_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("qwen3") or m.startswith("deepseek-r1")


def estimate_tokens(text: str) -> int:
    import unicodedata
    chinese = sum(1 for c in text if unicodedata.category(c) == "Lo")
    others = len(text) - chinese
    return int(chinese * 0.6 + others * 0.25)


def ollama_generate(model: str, prompt: str, num_ctx: int) -> dict:
    url = f"{OLLAMA_BASE}/api/generate"
    payload_dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
            "num_ctx": num_ctx,
        },
    }

    if is_thinking_model(model):
        payload_dict["think"] = False

    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {
        "response": data.get("response", ""),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": data.get("eval_count", 0),
        "elapsed_seconds": time.time() - start,
    }


def build_prompt(context: str, question: str) -> str:
    return f"""請根據以下文本回答問題。只根據文本中的資訊作答，用簡短的一句話回答。

文本：
{context}

問題：{question}

回答："""


def evaluate_answer(response: str, expected: str, converter) -> dict:
    response_clean = response.strip().lower()
    expected_clean = expected.strip().lower()
    expected_simp = converter.convert(expected_clean)

    exact_match = expected_clean in response_clean or expected_simp in response_clean
    response_numbers = set(re.findall(r"[\d.]+", response_clean))
    expected_numbers = set(re.findall(r"[\d.]+", expected_clean))
    number_match = bool(expected_numbers and expected_numbers.issubset(response_numbers))

    expected_chars = set(re.sub(r"[^\u4e00-\u9fff\w]", "", expected_clean))
    response_chars = set(re.sub(r"[^\u4e00-\u9fff\w]", "", response_clean))
    char_overlap = len(expected_chars & response_chars) / max(len(expected_chars), 1)

    return {
        "exact_match": exact_match,
        "number_match": number_match,
        "char_overlap": round(char_overlap, 3),
        "is_correct": exact_match or number_match,
    }


# ── 主要執行邏輯 ──────────────────────────────────────────────────────────────

def run_experiment(args, converter):
    model = args.model
    output_path = os.path.join(RESULTS_DIR, f"h2_{model}_results.jsonl")
    model_max_ctx = get_model_context_length(model)

    print(f"模型: {model}")
    print(f"Variant: {VARIANT}（簡體 context + 簡體 question）")
    print(f"模型最大 context length: {model_max_ctx:,} tokens")

    # 載入實驗
    experiments = []
    with open(HAYSTACKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            experiments.append(json.loads(line))

    # 篩選指定長度
    if args.lengths:
        target_lengths = set(int(x) for x in args.lengths.split(","))
        experiments = [e for e in experiments if e["context_length_chars"] in target_lengths]
        print(f"篩選長度: {sorted(target_lengths)} → {len(experiments)} 筆")

    total = len(experiments)

    # Resume 支援
    completed = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add(r["experiment_id"])
        print(f"已完成: {len(completed)} 筆，從中斷處繼續")

    print(f"實驗總數: {total}")
    print()

    mode = "a" if args.resume else "w"
    out_file = open(output_path, mode, encoding="utf-8")

    done = correct = 0
    skipped_lengths = set()

    try:
        for experiment in experiments:
            if experiment["experiment_id"] in completed:
                continue

            exp_id = experiment["experiment_id"]
            length = experiment["context_length_chars"]
            pos = experiment["needle_position"]
            expected = experiment["expected_answer"]

            # 已知此長度會 SKIP，直接跳過
            if length in skipped_lengths:
                done += 1
                continue

            # simplified context + simplified question
            context = experiment["simplified"]["text"]
            question_simp = converter.convert(experiment["question"])

            prompt = build_prompt(context, question_simp)

            print(
                f"  [{done+1:4d}/{total}] id={exp_id} len={length:,} pos={pos}",
                end="", flush=True,
            )

            # 跳過超出 context window 的
            estimated = estimate_tokens(prompt)
            if estimated > model_max_ctx - 256:
                result = {
                    "experiment_id": exp_id,
                    "model": model,
                    "variant": VARIANT,
                    "context_length_chars": length,
                    "needle_position": pos,
                    "trial": experiment["trial"],
                    "needle_id": experiment["needle_id"],
                    "question_traditional": experiment["question"],
                    "question_simplified": question_simp,
                    "expected_answer": expected,
                    "model_response": None,
                    "token_count_prompt": None,
                    "token_count_output": None,
                    "elapsed_seconds": 0.0,
                    "skipped": True,
                    "skip_reason": "context_length_exceeded",
                    "model_max_ctx": model_max_ctx,
                    "evaluation": {"exact_match": False, "number_match": False,
                                   "char_overlap": 0.0, "is_correct": False},
                }
                skipped_lengths.add(length)
                print(f" tokens≈{estimated} SKIP(context_length_exceeded) 此長度後續全部跳過")
            else:
                gen = ollama_generate(model, prompt, model_max_ctx)
                eval_result = evaluate_answer(gen["response"], expected, converter)

                result = {
                    "experiment_id": exp_id,
                    "model": model,
                    "variant": VARIANT,
                    "context_length_chars": length,
                    "needle_position": pos,
                    "trial": experiment["trial"],
                    "needle_id": experiment["needle_id"],
                    "question_traditional": experiment["question"],
                    "question_simplified": question_simp,
                    "expected_answer": expected,
                    "model_response": gen["response"],
                    "token_count_prompt": gen["prompt_eval_count"],
                    "token_count_output": gen["eval_count"],
                    "elapsed_seconds": gen["elapsed_seconds"],
                    "skipped": False,
                    "skip_reason": None,
                    "model_max_ctx": model_max_ctx,
                    "evaluation": eval_result,
                }

                status = "✓" if eval_result["is_correct"] else "✗"
                print(f" tokens={gen['prompt_eval_count']} {status}")

                correct += int(eval_result["is_correct"])

            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            done += 1

    except KeyboardInterrupt:
        print("\n\n中斷。可用 --resume 繼續。")
    finally:
        out_file.close()

    if done > 0:
        print(f"\n=== 結果摘要 ===")
        non_skipped = done - sum(1 for _ in open(output_path) if json.loads(_).get("skipped"))
        print(f"  simplified_q 準確率: {correct}/{non_skipped} ({correct/max(non_skipped,1)*100:.1f}%)")
        print(f"  結果儲存至: {output_path}")


# ── 比較分析 ──────────────────────────────────────────────────────────────────

def compare(model: str):
    orig_path = os.path.join(RESULTS_DIR, f"{model}_results.jsonl")
    h2_path = os.path.join(RESULTS_DIR, f"h2_{model}_results.jsonl")

    if not os.path.exists(orig_path):
        print(f"找不到原始結果: {orig_path}")
        return
    if not os.path.exists(h2_path):
        print(f"找不到假設二結果: {h2_path}")
        return

    try:
        import opencc
        converter = opencc.OpenCC("t2s")
    except ImportError:
        converter = None

    # 載入原始 simplified 結果（重新 evaluate）
    orig_simp = {}
    with open(orig_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("skipped") or r["variant"] != "simplified":
                continue
            # 重新評估（確保 opencc 比對正確）
            if converter:
                resp = (r.get("model_response") or "").strip().lower()
                exp = r["expected_answer"].strip().lower()
                exp_simp = converter.convert(exp)
                nums_r = set(re.findall(r"[\d.]+", resp))
                nums_e = set(re.findall(r"[\d.]+", exp))
                is_correct = (exp in resp or exp_simp in resp or
                              bool(nums_e and nums_e.issubset(nums_r)))
            else:
                is_correct = r["evaluation"]["is_correct"]
            orig_simp[r["experiment_id"]] = is_correct

    # 載入 simplified_q 結果
    h2_simp = {}
    with open(h2_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("skipped"):
                continue
            h2_simp[r["experiment_id"]] = r["evaluation"]["is_correct"]

    # 只比較兩者都有的 experiment_id
    common = set(orig_simp) & set(h2_simp)
    if not common:
        print("沒有可比較的共同實驗")
        return

    # 整體比較
    orig_correct = sum(orig_simp[i] for i in common)
    h2_correct = sum(h2_simp[i] for i in common)
    n = len(common)

    print(f"\n{'═'*60}")
    print(f"  假設二驗證：{model}")
    print(f"  共 {n} 組可比較實驗")
    print(f"{'═'*60}")
    print(f"\n  {'Variant':<20} {'正確':>6} {'總數':>6} {'準確率':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*8}")
    print(f"  {'simplified（繁題+簡文）':<20} {orig_correct:>6} {n:>6} {orig_correct/n*100:>7.1f}%")
    print(f"  {'simplified_q（簡題+簡文）':<20} {h2_correct:>6} {n:>6} {h2_correct/n*100:>7.1f}%")
    diff = (h2_correct - orig_correct) / n * 100
    print(f"\n  差距（simplified_q − simplified）: {diff:+.1f} pp")
    if diff > 2:
        print(f"  ✅ 假設二成立：問題語言統一後準確率提升 {diff:.1f}pp，跨字形有損耗")
    elif diff < -2:
        print(f"  ❌ 假設二不成立：問題改為簡體反而下降 {abs(diff):.1f}pp")
    else:
        print(f"  ⚪ 結果不顯著（差距 ≤ 2pp），跨字形影響有限")

    # 按 context 長度分組
    from collections import defaultdict

    by_length_orig = defaultdict(lambda: [0, 0])
    by_length_h2   = defaultdict(lambda: [0, 0])

    with open(orig_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("skipped") or r["variant"] != "simplified": continue
            if r["experiment_id"] not in common: continue
            length = r["context_length_chars"]
            by_length_orig[length][1] += 1
            by_length_orig[length][0] += int(orig_simp[r["experiment_id"]])

    with open(h2_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("skipped"): continue
            if r["experiment_id"] not in common: continue
            length = r["context_length_chars"]
            by_length_h2[length][1] += 1
            by_length_h2[length][0] += int(h2_simp[r["experiment_id"]])

    print(f"\n  {'長度':>10}  {'簡體(繁題)':>12}  {'簡體(簡題)':>12}  {'差距':>8}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}")
    for length in sorted(set(by_length_orig) | set(by_length_h2)):
        c1, t1 = by_length_orig[length]
        c2, t2 = by_length_h2[length]
        if t1 == 0 or t2 == 0: continue
        a1, a2 = c1/t1*100, c2/t2*100
        print(f"  {length:>10,}  {a1:>11.1f}%  {a2:>11.1f}%  {a2-a1:>+7.1f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="假設二驗證：問題語言對簡體準確率的影響")
    parser.add_argument("--model", required=True, help="Ollama 模型名稱")
    parser.add_argument("--resume", action="store_true", help="從中斷處繼續")
    parser.add_argument("--lengths", type=str, default=None,
                        help="只跑指定長度（逗號分隔，例如 100000,130000）")
    parser.add_argument("--compare", action="store_true",
                        help="不執行實驗，只做比較分析（需已有兩份結果）")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="最多執行幾組（測試用）")
    args = parser.parse_args()

    converter = get_converter()

    if args.compare:
        compare(args.model)
    else:
        run_experiment(args, converter)


if __name__ == "__main__":
    main()
