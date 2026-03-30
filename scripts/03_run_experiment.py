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


def get_model_context_length(model: str) -> int:
    """從 Ollama /api/show 取得模型的最大 context length"""
    url = f"{OLLAMA_BASE}/api/show"
    payload = json.dumps({"name": model}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        info = data.get("model_info", {})
        # 各模型架構的 key 名稱不同，例如 llama.context_length、qwen3.context_length
        for key, val in info.items():
            if "context_length" in key:
                return int(val)
    except Exception:
        pass
    return 131072  # 查不到時保守預設


def estimate_tokens(text: str) -> int:
    """粗估 token 數：中文字約 0.6 tokens/char，其餘約 0.25 tokens/char"""
    import unicodedata
    chinese = sum(1 for c in text if unicodedata.category(c) == "Lo")
    others = len(text) - chinese
    return int(chinese * 0.6 + others * 0.25)


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


def is_thinking_model(model: str) -> bool:
    """
    判斷是否為 thinking 模型（會輸出 <think>...</think> 推理過程）

    目前已知的 thinking 模型：
      - qwen3.*（所有 qwen3 系列，包括 qwen3:8b, qwen3.5:35b 等）
      - deepseek-r1.*

    對這些模型透過 Ollama API 的 "think": false 參數關閉推理模式，
    確保 response 只包含答案，不含推理過程。
    """
    model_lower = model.lower()
    return model_lower.startswith("qwen3") or model_lower.startswith("deepseek-r1")


def ollama_generate(model: str, prompt: str, num_ctx: int, temperature: float = 0.0) -> dict:
    """呼叫 Ollama API 生成回應"""
    url = f"{OLLAMA_BASE}/api/generate"
    payload_dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
            "num_ctx": num_ctx,
        },
    }

    # Thinking 模型：透過 API 參數關閉推理，不污染 prompt
    if is_thinking_model(model):
        payload_dict["think"] = False

    payload = json.dumps(payload_dict).encode("utf-8")

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


def evaluate_answer(response: str, expected: str, variant: str = "traditional") -> dict:
    """
    評估模型回答是否正確

    策略：檢查 expected answer 是否出現在回答中
    （寬鬆匹配，因為模型可能用不同方式表達同一事實）

    simplified variant 會額外把 expected 轉成簡體後比對，
    避免繁體 expected 對上簡體回答時誤判為錯。
    注意：這是粗估用的快速判斷，最終評估請用 05_llm_judge.py。
    """
    response_clean = response.strip().lower()
    expected_clean = expected.strip().lower()

    # 簡體 variant：把 expected 也轉成簡體再比對
    expected_simp_clean = expected_clean
    if variant == "simplified":
        try:
            import opencc
            expected_simp_clean = opencc.OpenCC("t2s").convert(expected_clean)
        except ImportError:
            pass  # opencc 未安裝時退回原始比對

    # 完全包含（繁體或簡體版本任一中即可）
    exact_match = expected_clean in response_clean or expected_simp_clean in response_clean

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


def run_single_experiment(
    model: str, experiment: dict, variant: str, model_max_ctx: int
) -> dict:
    """
    對單一實驗組合執行測試

    variant: "traditional" 或 "simplified"
    model_max_ctx: 模型最大 context length（由 get_model_context_length 取得）

    若預估 prompt token 數超過 model_max_ctx，直接回傳 skipped 記錄，
    不送給模型，避免截斷導致實驗結果失真。
    """
    data = experiment[variant]
    context = data["text"]
    question = experiment["question"]
    expected = experiment["expected_answer"]

    # 建構 prompt
    prompt = build_prompt(context, question)

    # 預估 token 數，預留 256 tokens 給模型輸出
    estimated = estimate_tokens(prompt)
    if estimated > model_max_ctx - 256:
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
            "model_response": None,
            "token_count_actual": estimated,
            "token_count_prompt": None,
            "token_count_output": None,
            "elapsed_seconds": 0.0,
            "skipped": True,
            "skip_reason": "context_length_exceeded",
            "model_max_ctx": model_max_ctx,
            "evaluation": {
                "exact_match": False,
                "number_match": False,
                "char_overlap": 0.0,
                "is_correct": False,
            },
        }

    # 送出給模型
    result = ollama_generate(model, prompt, num_ctx=model_max_ctx)

    # 取得精確 token 數：優先用 /api/tokenize，不支援時 fallback 到 prompt_eval_count
    token_count = ollama_tokenize(model, context)
    if token_count == -1:
        token_count = result["prompt_eval_count"]

    # 評估答案
    evaluation = evaluate_answer(result["response"], expected, variant)

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
        "skipped": False,
        "skip_reason": None,
        "model_max_ctx": model_max_ctx,
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser(description="執行 Context Rot 實驗")
    parser.add_argument("--model", required=True, help="Ollama 模型名稱")
    parser.add_argument("--variant", choices=["traditional", "simplified", "both"],
                        default="both",
                        help="只跑指定 variant（預設 both）")
    parser.add_argument("--lengths", type=str, default=None,
                        help="只跑指定長度（逗號分隔，例如 100000,130000）")
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

    # 篩選指定長度
    if args.lengths:
        target_lengths = set(int(x) for x in args.lengths.split(","))
        experiments = [e for e in experiments if e["context_length_chars"] in target_lengths]
        print(f"篩選長度: {sorted(target_lengths)} → {len(experiments)} 筆")

    # 取得模型 context length
    model_max_ctx = get_model_context_length(args.model)

    # 決定要跑哪些 variant
    if args.variant == "both":
        variants_to_run = ["traditional", "simplified"]
    else:
        variants_to_run = [args.variant]

    print(f"模型: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"模型最大 context length: {model_max_ctx:,} tokens")
    print(f"實驗總數: {len(experiments)} × {len(variants_to_run)} = {len(experiments) * len(variants_to_run)}")

    # 如果 resume，找出已完成的實驗
    completed = set()
    if (args.resume or args.variant != "both" or args.lengths) and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    key = (r["experiment_id"], r["variant"])
                    completed.add(key)
                except json.JSONDecodeError:
                    pass  # 跳過損壞行
        if completed:
            print(f"已完成: {len(completed)} 筆，跳過繼續")

    # 開啟輸出檔（variant 指定時強制用 append，避免覆蓋另一個 variant 的資料）
    mode = "a" if (args.resume or args.variant != "both" or args.lengths) else "w"
    out_file = open(output_path, mode, encoding="utf-8")

    total = len(experiments) * len(variants_to_run)
    if args.max_experiments:
        total = min(total, args.max_experiments)

    done = 0
    correct_trad = 0
    correct_simp = 0
    total_trad = 0
    total_simp = 0

    skipped_lengths = set()  # 已知超過 context window 的長度，直接跳過

    try:
        for experiment in experiments:
            for variant in variants_to_run:
                if done >= total:
                    break

                key = (experiment["experiment_id"], variant)
                if key in completed:
                    continue

                exp_id = experiment["experiment_id"]
                length = experiment["context_length_chars"]
                pos = experiment["needle_position"]
                label = "繁" if variant == "traditional" else "簡"

                # 已知此長度會 SKIP，直接跳過不逐筆處理
                if length in skipped_lengths:
                    done += 1
                    continue

                # 進度顯示
                print(
                    f"  [{done+1:4d}/{total}] "
                    f"id={exp_id} {label} "
                    f"len={length:,} pos={pos}",
                    end="",
                    flush=True,
                )

                try:
                    result = run_single_experiment(
                        args.model, experiment, variant, model_max_ctx
                    )

                    # 記錄結果
                    out_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    out_file.flush()

                    # 統計
                    if result.get("skipped"):
                        skipped_lengths.add(length)
                        print(f" tokens≈{result['token_count_actual']} SKIP(context_length_exceeded) 此長度後續全部跳過")
                    else:
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
