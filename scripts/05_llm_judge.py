"""
Step 5: 用 LLM 重新評估實驗結果

讀取現有的模型回答，交給指定的 judge 模型判斷是否正確。
判斷時不區分繁簡體，以語意為準。

用法:
  # 用本地 Ollama 模型（例如 128B）
  python scripts/05_llm_judge.py --model qwen3:8b --judge ollama:qwen2.5:72b

  # 用 OpenAI API
  python scripts/05_llm_judge.py --model qwen3:8b --judge openai:gpt-4.1

  # 從中斷處繼續
  python scripts/05_llm_judge.py --model qwen3:8b --judge ollama:qwen2.5:72b --resume

輸出:
  results/evaluations/{judge_id}/{model}_eval.jsonl
  每行格式: {"experiment_id": 0, "variant": "traditional", "is_correct": true, "judge_response": "正確"}

之後用 04_analyze.py 時可加 --eval-file 指定此評估檔：
  python scripts/04_analyze.py --model qwen3:8b --eval-file results/evaluations/.../qwen3:8b_eval.jsonl
"""

import argparse
import json
import os
import time
import urllib.request

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluations")
OLLAMA_BASE = "http://localhost:11434"

JUDGE_PROMPT = """\
你是一個客觀的評估助手。請判斷模型回答是否正確。

問題：{question}
標準答案：{expected_answer}
模型回答：{model_response}

判斷標準：
1. 回答中是否包含與標準答案相同意思的資訊
2. 繁體與簡體視為相同（例如「隻」=「只」、「點」=「点」）
3. 數字、金額、日期等事實必須一致
4. 允許模型用不同句式表達，只要核心事實正確即可

請只回答「正確」或「錯誤」，不要加任何解釋。"""


# ── backends ──────────────────────────────────────────────────────────────────

def judge_ollama(model: str, prompt: str) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def judge_openai(model: str, prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def parse_verdict(response: str) -> bool:
    """把 judge 回應轉成 bool（寬鬆解析）"""
    r = response.strip()
    if "正確" in r and "錯誤" not in r:
        return True
    if "錯誤" in r and "正確" not in r:
        return False
    if r.lower() in ("yes", "correct", "true", "1"):
        return True
    if r.lower() in ("no", "incorrect", "false", "0"):
        return False
    # 無法解析，視為錯誤並記錄
    return False


# ── per-model logic ───────────────────────────────────────────────────────────

def judge_one_model(model: str, judge_id: str, backend: str, judge_model: str,
                    openai_key: str, resume: bool):
    results_path = os.path.join(RESULTS_DIR, f"{model}_results.jsonl")
    if not os.path.exists(results_path):
        print(f"  找不到結果檔，跳過：{results_path}")
        return

    out_dir = os.path.join(EVAL_DIR, judge_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model}_eval.jsonl")

    # 載入已完成的評估（resume 用）
    completed = set()
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add((r["experiment_id"], r["variant"]))
        print(f"  已完成 {len(completed)} 筆，從中斷處繼續")

    # 載入全部原始結果
    all_results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    remaining = [r for r in all_results
                 if (r["experiment_id"], r["variant"]) not in completed]
    print(f"  總筆數: {len(all_results)}  待評估: {len(remaining)}")

    mode = "a" if resume else "w"
    done = 0
    errors = 0

    with open(out_path, mode, encoding="utf-8") as out_f:
        for r in remaining:
            prompt = JUDGE_PROMPT.format(
                question=r["question"],
                expected_answer=r["expected_answer"],
                model_response=r["model_response"],
            )

            label = "繁" if r["variant"] == "traditional" else "簡"
            print(
                f"  [{done+1:4d}/{len(remaining)}] "
                f"id={r['experiment_id']} {label} "
                f"len={r['context_length_chars']:,} pos={r['needle_position']}",
                end="", flush=True,
            )

            try:
                if backend == "ollama":
                    judge_resp = judge_ollama(judge_model, prompt)
                else:
                    judge_resp = judge_openai(judge_model, prompt, openai_key)

                is_correct = parse_verdict(judge_resp)
                status = "✓" if is_correct else "✗"
                print(f"  {judge_resp[:8]!r} → {status}")

                out_f.write(json.dumps({
                    "experiment_id": r["experiment_id"],
                    "variant": r["variant"],
                    "is_correct": is_correct,
                    "judge_response": judge_resp,
                    "question": r["question"],
                    "expected_answer": r["expected_answer"],
                    "model_response": r["model_response"][:200],
                }, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"  錯誤: {e}")
                errors += 1

            done += 1

    print(f"  完成。錯誤: {errors}  →  {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="用 LLM 重新評估實驗結果")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="被評估的模型名稱（對應 results/{model}_results.jsonl）")
    group.add_argument("--all", action="store_true", help="評估所有 results/*_results.jsonl")
    parser.add_argument("--judge", required=True,
                        help="judge 模型，格式：ollama:<name> 或 openai:<name>")
    parser.add_argument("--resume", action="store_true", help="從中斷處繼續")
    parser.add_argument("--openai-api-key", default=None,
                        help="OpenAI API key（也可用環境變數 OPENAI_API_KEY）")
    args = parser.parse_args()

    # 解析 judge 後端
    if ":" not in args.judge:
        print("--judge 格式錯誤，請用 ollama:<model> 或 openai:<model>")
        return
    backend, judge_model = args.judge.split(":", 1)
    if backend not in ("ollama", "openai"):
        print(f"不支援的後端：{backend}")
        return

    openai_key = None
    if backend == "openai":
        openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("使用 OpenAI 需要提供 API key（--openai-api-key 或 OPENAI_API_KEY 環境變數）")
            return

    # 決定要處理哪些模型
    if args.all:
        models = sorted(
            f.replace("_results.jsonl", "")
            for f in os.listdir(RESULTS_DIR)
            if f.endswith("_results.jsonl")
        )
        if not models:
            print(f"在 {RESULTS_DIR} 找不到任何 *_results.jsonl")
            return
    else:
        models = [args.model]

    judge_id = args.judge.replace(":", "_").replace("/", "-")
    print(f"Judge: {args.judge}  模型數: {len(models)}")
    print(f"輸出目錄: {os.path.join(EVAL_DIR, judge_id)}\n")

    eval_files = []
    for model in models:
        print(f"── {model} ──")
        out_path = judge_one_model(
            model=model,
            judge_id=judge_id,
            backend=backend,
            judge_model=judge_model,
            openai_key=openai_key,
            resume=args.resume,
        )
        if out_path:
            eval_files.append((model, out_path))

    print(f"\n{'═' * 50}")
    print("全部完成。重新分析指令：")
    for model, path in eval_files:
        print(f"  python scripts/04_analyze.py --model {model} --eval-file {path}")


if __name__ == "__main__":
    main()
