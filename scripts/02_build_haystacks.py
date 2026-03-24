"""
Step 2: 建構實驗用的 Haystack

從維基百科原始語料建構：
  1. 繁體版 haystack
  2. 簡體版 haystack（OpenCC 轉換）
  3. 在指定位置插入 needle
  4. 記錄每個 haystack 在各 tokenizer 下的 token 數

用法: python scripts/02_build_haystacks.py

輸出: data/haystacks/ 目錄下的 JSON 檔案
"""

import json
import os
import random
import re

# --- 繁簡轉換 ---
# 使用 opencc-python-reimplemented（純 Python，無需 C 編譯）
try:
    from opencc import OpenCC
    converter_tw2sp = OpenCC("t2s")  # 繁體轉簡體
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    print("警告: 未安裝 opencc-python-reimplemented")
    print("請執行: pip install opencc-python-reimplemented")


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wiki_raw_v2", "zh")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "haystacks")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "wiki_articles_v2.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wiki_articles() -> list[str]:
    """載入所有維基百科文章，回傳純文字清單"""
    articles = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if len(text) > 500:
            articles.append(text)
    return articles


def clean_text(text: str) -> str:
    """清理文本：移除多餘空白、章節標題記號等"""
    # 移除連續空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 移除 == 標題標記 ==（維基百科格式殘留）
    text = re.sub(r"={2,}.*?={2,}", "", text)
    # 移除多餘空白
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def to_simplified(text: str) -> str:
    """繁體轉簡體"""
    if not HAS_OPENCC:
        raise RuntimeError("需要安裝 opencc-python-reimplemented")
    return converter_tw2sp.convert(text)


def build_haystack_text(articles: list[str], target_chars: int, rng: random.Random) -> str:
    """
    從多篇文章中拼接出指定字元數的 haystack

    策略：隨機選取文章段落，拼接到目標長度
    """
    # 把所有文章拆成段落
    paragraphs = []
    for article in articles:
        cleaned = clean_text(article)
        for para in cleaned.split("\n\n"):
            para = para.strip()
            if len(para) > 50:  # 過濾太短的段落
                paragraphs.append(para)

    if not paragraphs:
        raise ValueError("沒有可用的段落")

    # 隨機打亂段落順序
    rng.shuffle(paragraphs)

    # 拼接到目標長度
    result = []
    current_length = 0

    for para in paragraphs:
        if current_length >= target_chars:
            break
        result.append(para)
        current_length += len(para)

    # 如果段落不夠，重複使用（打亂後再用一輪）
    while current_length < target_chars:
        rng.shuffle(paragraphs)
        for para in paragraphs:
            if current_length >= target_chars:
                break
            result.append(para)
            current_length += len(para)

    # 拼接並截斷到精確長度
    full_text = "\n\n".join(result)
    return full_text[:target_chars]


def insert_needle(haystack: str, needle: str, position: float) -> str:
    """
    在 haystack 的指定相對位置插入 needle

    position: 0.0~1.0 的浮點數，表示插入位置的百分比
    插入點選在最近的段落邊界（\n\n），避免破壞句子
    """
    target_pos = int(len(haystack) * position)

    # 找最近的段落邊界
    newline_positions = [m.start() for m in re.finditer(r"\n\n", haystack)]

    if not newline_positions:
        # 沒有段落邊界，直接插入
        insert_pos = target_pos
    else:
        # 找最接近 target_pos 的段落邊界
        insert_pos = min(newline_positions, key=lambda x: abs(x - target_pos))

    # 插入 needle（前後加空行以區隔）
    result = haystack[:insert_pos] + "\n\n" + needle + "\n\n" + haystack[insert_pos:]
    return result


def count_tokens_by_char(text: str) -> dict:
    """
    計算各種 token 統計（不依賴外部 tokenizer）
    回傳字元數和估算的 token 數
    """
    stats = {
        "char_count": len(text),
        "chinese_char_count": sum(1 for c in text if "\u4e00" <= c <= "\u9fff"),
        "ascii_char_count": sum(1 for c in text if c.isascii()),
    }

    # 粗估 token 數（繁體中文約 1.5~1.8 字元/token）
    # 精確 token 數留給 Step 3 在各模型上實際測量
    stats["estimated_tokens_zh"] = int(
        stats["chinese_char_count"] / 1.5 + stats["ascii_char_count"] / 4
    )

    return stats


def main():
    if not HAS_OPENCC:
        print("錯誤: 請先安裝 opencc-python-reimplemented")
        print("  pip install opencc-python-reimplemented")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    config = load_config()
    params = config["experiment_params"]
    needles = config["needles"]["general"]  # v2: needles 是 dict，取 general 清單

    context_lengths = params["context_lengths_chars"]
    needle_positions = params["needle_positions"]
    seed = params["random_seed"]

    # 載入文章
    articles = load_wiki_articles()
    if not articles:
        print("錯誤: data/wiki_raw_v2/zh/ 目錄下沒有文章")
        print("請先執行: python scripts/01_fetch_wiki_v2.py")
        return

    print(f"載入 {len(articles)} 篇文章")
    print(f"Context 長度: {context_lengths}")
    print(f"Needle 位置: {needle_positions}")
    print()

    rng = random.Random(seed)
    all_experiments = []
    experiment_id = 0

    for length in context_lengths:
        for pos in needle_positions:
            # 每個 (長度, 位置) 組合，用不同 seed 產生多個 haystack
            for trial in range(params["trials_per_combination"]):
                # 為每個 trial 建立獨立的 rng
                trial_rng = random.Random(seed + experiment_id)

                # 選一個 needle（輪流使用）
                needle_info = needles[trial % len(needles)]

                # 建構繁體 haystack
                haystack_traditional = build_haystack_text(
                    articles, length, trial_rng
                )

                # 插入繁體 needle
                needle_traditional = needle_info["fact"]
                full_traditional = insert_needle(
                    haystack_traditional, needle_traditional, pos
                )

                # 轉換為簡體版本
                full_simplified = to_simplified(full_traditional)

                # 統計
                stats_trad = count_tokens_by_char(full_traditional)
                stats_simp = count_tokens_by_char(full_simplified)

                experiment = {
                    "experiment_id": experiment_id,
                    "context_length_chars": length,
                    "needle_position": pos,
                    "trial": trial,
                    "needle_id": needle_info["id"],
                    "question": needle_info["question"],
                    "expected_answer": needle_info["expected"],
                    "traditional": {
                        "text": full_traditional,
                        "needle": needle_traditional,
                        "stats": stats_trad,
                    },
                    "simplified": {
                        "text": full_simplified,
                        "needle": to_simplified(needle_traditional),
                        "stats": stats_simp,
                    },
                }

                all_experiments.append(experiment)
                experiment_id += 1

    # 儲存
    output_path = os.path.join(OUT_DIR, "experiments.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for exp in all_experiments:
            f.write(json.dumps(exp, ensure_ascii=False) + "\n")

    print(f"已產生 {len(all_experiments)} 個實驗組合")
    print(f"儲存至: {output_path}")

    # 摘要統計
    print("\n=== 實驗矩陣摘要 ===")
    print(f"  Context 長度:   {len(context_lengths)} 級")
    print(f"  Needle 位置:    {len(needle_positions)} 個")
    print(f"  每組合重複:     {params['trials_per_combination']} 次")
    print(f"  字形版本:       2 (繁體/簡體)")
    print(f"  總實驗數:       {len(all_experiments)} × 2 = {len(all_experiments) * 2}")

    # 顯示一個範例
    sample = all_experiments[len(all_experiments) // 2]
    print(f"\n=== 範例 (experiment_id={sample['experiment_id']}) ===")
    print(f"  長度: {sample['context_length_chars']} 字元")
    print(f"  Needle 位置: {sample['needle_position']}")
    print(f"  問題: {sample['question']}")
    print(f"  繁體字元數: {sample['traditional']['stats']['char_count']}")
    print(f"  簡體字元數: {sample['simplified']['stats']['char_count']}")
    print(f"  繁體中文字數: {sample['traditional']['stats']['chinese_char_count']}")
    print(f"  簡體中文字數: {sample['simplified']['stats']['chinese_char_count']}")


if __name__ == "__main__":
    main()
