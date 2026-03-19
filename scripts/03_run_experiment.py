"""
Step 3: 執行 Context Rot 實驗

透過 Ollama API 對模型進行 needle-in-a-haystack 測試
分別用繁體和簡體版本，記錄每次的結果

用法:
  python scripts/03_run_experiment.py --model qwen3
  python scripts/03_run_experiment.py --model llama3.1
  python scripts/03_run_experiment.py --model gemma3

前置條件:
  1. Ollama 正在運行: ollama serve
  2. 模型已下載: ollama pull qwen3
  3. 已執行 02_build_haystacks.py 產生實驗資料

輸出: results/{model}_results.jsonl
"""

import argparse
import json
import os
import time
import urllib.request

HAYSTACKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "haystacks", "experiments.jsonl"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OLLAMA_BASE = "http://localhost:11434"


def ollama_tokenize(model: str, text: str) -> int:
    """用 Ollama API 取得精確的 token 數"""
    url = f"{OLLAMA_BASE}/api/tokenize"
    payload = json.dumps({"model": model, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        return len(data.get("tokens", []))
    except Exception:
        return -1  # tokenize API 可能不支援，回傳 -1


def ollama_generate(model: str, prompt: str, temperature: float = 0.0) -> dict:
    """呼叫 Ollama API 生成回應"""
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,         # 限制輸出長度
            "num_ctx": 65536,           # 確保 context window 夠大
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    start_time = time.time()
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - start_time

    return {
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count", 0),       # 輸出 token 數
        "prompt_eval_count": data.get("prompt_eval_count", 0),  # 輸入 token 數
        "total_duration_ns": data.get("total_duration", 0),
        "elapsed_seconds": elapsed,
    }


def build_prompt(context: str, question: str) -> str:
    """
    建構 prompt

    設計原則：
    - 簡潔明確的指令
    - 不要過度引導模型
    - 繁簡體使用相同的 prompt 結構
    """
    return f"""請根據以下文本回答問題。只根據文本中的資訊作答，用簡短的一句話回答。

文本：
{context}

問題：{question}

回答："""


def evaluate_answer(response: str, expected: str) -> dict:
    """
    評估模型回答是否正確

    策略：檢查 expected answer 是否出現在回答中
    （寬鬆匹配，因為模型可能用不同方式表達同一事實）
    """
    response_clean = response.strip().lower()
    expected_clean = expected.strip().lower()

    # 完全包含
    exact_match = expected_clean in response_clean

    # 數字匹配（提取數字比較）
    import re
    response_numbers = set(re.findall(r"[\d.]+", response_clean))
    expected_numbers = set(re.findall(r"[\d.]+", expected_clean))
    number_match = bool(expected_numbers and expected_numbers.issubset(response_numbers))

    # 關鍵詞匹配（期望答案的主要詞彙是否出現）
    # 去掉標點符號後切字檢查
    expected_chars = set(re.sub(r"[^\u4e00-\u9fff\w]", "", expected_clean))
    response_chars = set(re.sub(r"[^\u4e00-\u9fff\w]", "", response_clean))
    char_overlap = len(expected_chars & response_chars) / max(len(expected_chars), 1)

    return {
        "exact_match": exact_match,
        "number_match": number_match,
        "char_overlap": round(char_overlap, 3),
        "is_correct": exact_match or number_match,  # 主要判定標準
    }


def run_single_experiment(model: str, experiment: dict, variant: str) -> dict:
    """
    對單一實驗組合執行測試

    variant: "traditional" 或 "simplified"
    """
    data = experiment[variant]
    context = data["text"]
    question = experiment["question"]
    expected = experiment["expected_answer"]

    # 取得精確 token 數
    token_count = ollama_tokenize(model, context)

    # 建構 prompt 並送出
    prompt = build_prompt(context, question)
    result = ollama_generate(model, prompt)

    # 評估答案
    evaluation = evaluate_answer(result["response"], expected)

    return {
        "experiment_id": experiment["experiment_id"],
        "model": model,
        "variant": variant,
        "context_length_chars": experiment["context_length_chars"],
        "needle_position": experiment["needle_position"],
        "trial": experiment["trial"],
        "needle_id": experiment["needle_id"],
        "question": question,
        "expected_answer": expected,
        "model_response": result["response"],
        "token_count_actual": token_count,
        "token_count_prompt": result["prompt_eval_count"],
        "token_count_output": result["eval_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser(description="執行 Context Rot 實驗")
    parser.add_argument("--model", required=True, help="Ollama 模型名稱")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="最多執行幾組（用於測試）")
    parser.add_argument("--resume", action="store_true",
                        help="從上次中斷處繼續")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{args.model}_results.jsonl")

    # 載入實驗資料
    experiments = []
    with open(HAYSTACKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            experiments.append(json.loads(line))

    print(f"模型: {args.model}")
    print(f"實驗總數: {len(experiments)} × 2 (繁/簡) = {len(experiments) * 2}")

    # 如果 resume，找出已完成的實驗
    completed = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["experiment_id"], r["variant"])
                completed.add(key)
        print(f"已完成: {len(completed)} 筆，從中斷處繼續")

    # 開啟輸出檔（append 模式）
    mode = "a" if args.resume else "w"
    out_file = open(output_path, mode, encoding="utf-8")

    total = len(experiments) * 2
    if args.max_experiments:
        total = min(total, args.max_experiments)

    done = 0
    correct_trad = 0
    correct_simp = 0
    total_trad = 0
    total_simp = 0

    try:
        for experiment in experiments:
            for variant in ["traditional", "simplified"]:
                if done >= total:
                    break

                key = (experiment["experiment_id"], variant)
                if key in completed:
                    continue

                # 進度顯示
                exp_id = experiment["experiment_id"]
                length = experiment["context_length_chars"]
                pos = experiment["needle_position"]
                label = "繁" if variant == "traditional" else "簡"
                print(
                    f"  [{done+1:4d}/{total}] "
                    f"id={exp_id} {label} "
                    f"len={length:,} pos={pos}",
                    end="",
                    flush=True,
                )

                try:
                    result = run_single_experiment(
                        args.model, experiment, variant
                    )

                    # 記錄結果
                    out_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    out_file.flush()

                    # 統計
                    is_correct = result["evaluation"]["is_correct"]
                    token_info = f"tokens={result['token_count_actual']}"
                    status = "✓" if is_correct else "✗"
                    print(f" {token_info} {status}")

                    if variant == "traditional":
                        total_trad += 1
                        correct_trad += int(is_correct)
                    else:
                        total_simp += 1
                        correct_simp += int(is_correct)

                except Exception as e:
                    print(f" 錯誤: {e}")

                done += 1

            if done >= total:
                break

    except KeyboardInterrupt:
        print("\n\n中斷。已儲存目前的結果。")
        print("可用 --resume 從中斷處繼續。")

    finally:
        out_file.close()

    # 最終統計
    print("\n=== 結果摘要 ===")
    if total_trad > 0:
        print(f"  繁體: {correct_trad}/{total_trad} "
              f"({correct_trad/total_trad*100:.1f}%)")
    if total_simp > 0:
        print(f"  簡體: {correct_simp}/{total_simp} "
              f"({correct_simp/total_simp*100:.1f}%)")
    print(f"\n結果儲存至: {output_path}")


if __name__ == "__main__":
    main()
