"""
快速測試：用最少的資料驗證整個 pipeline 是否能跑通

用法: python scripts/05_quick_test.py --model qwen3

這個腳本會：
  1. 用一段硬編碼的繁體中文文本（不需要下載維基百科）
  2. 轉換成簡體
  3. 插入 needle
  4. 對模型測試一次繁體、一次簡體
  5. 比較 token 數和回答結果
"""

import argparse
import json
import urllib.request

OLLAMA_BASE = "http://localhost:11434"

# --- 測試用的硬編碼語料 ---
SAMPLE_TEXT = """台灣位於亞洲東部，太平洋西北側，北迴歸線橫跨其南部。台灣全島面積約為三萬六千平方公里，
南北長約三百九十五公里，東西最寬處約一百四十四公里。台灣本島因地殼運動而隆起，地形以山地為主，
山脈縱貫全島。台灣的最高峰為玉山，海拔三千九百五十二公尺，是東北亞的第一高峰。台灣的氣候屬於
亞熱帶至熱帶季風氣候，全年溫暖濕潤。北部年均溫約攝氏二十二度，南部年均溫約攝氏二十五度。
台灣的降雨量豐沛，年均降雨量約為二千五百毫米。颱風季節主要在每年七月至十月之間。

台灣的生態環境相當豐富，擁有超過四千種維管束植物，其中約有四分之一為台灣特有種。台灣的動物
多樣性也非常高，已記錄的哺乳類動物約有七十餘種，鳥類超過六百種，兩棲類約有三十餘種，
爬蟲類約有九十餘種。台灣黑熊是台灣最具代表性的野生動物之一，目前族群數量約為二百至六百隻。
櫻花鉤吻鮭是台灣的國寶魚，僅分布在大甲溪上游的七家灣溪等少數溪流中。

台灣的經濟發展始於一九六零年代的出口導向工業化，隨後在一九八零年代轉型為高科技產業。
目前台灣是全球最大的半導體代工製造基地，台灣積體電路製造公司是全球市值最高的半導體企業之一。
台灣也是全球重要的電子零組件供應地，在筆記型電腦、伺服器、網通設備等領域具有重要地位。
台灣的農業以稻米、水果和茶葉為主要產品，其中台灣高山茶享譽國際。"""

NEEDLE_FACT = "根據最新的環境調查報告，阿里山地區的千年檜木群中發現了一株樹齡超過三千八百年的紅檜，是目前已知台灣最古老的樹木。"
QUESTION = "台灣最古老的樹木樹齡是多少？"
EXPECTED = "三千八百年"


def ollama_tokenize(model: str, text: str) -> list[int]:
    """取得 token 列表"""
    url = f"{OLLAMA_BASE}/api/tokenize"
    payload = json.dumps({"model": model, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("tokens", [])
    except Exception as e:
        print(f"  tokenize API 錯誤: {e}")
        return []


def ollama_generate(model: str, prompt: str) -> str:
    """生成回應"""
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "")


def to_simplified(text: str) -> str:
    """繁體轉簡體"""
    try:
        from opencc import OpenCC
        cc = OpenCC("t2s")
        return cc.convert(text)
    except ImportError:
        print("警告: 未安裝 opencc，使用原始文本")
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", help="模型名稱")
    args = parser.parse_args()
    model = args.model

    print(f"模型: {model}")
    print(f"{'═' * 50}")

    # 準備繁體和簡體版本
    # 在文本中間插入 needle
    mid = len(SAMPLE_TEXT) // 2
    text_trad = SAMPLE_TEXT[:mid] + "\n\n" + NEEDLE_FACT + "\n\n" + SAMPLE_TEXT[mid:]
    text_simp = to_simplified(text_trad)

    print(f"\n📝 文本統計")
    print(f"  繁體字元數: {len(text_trad):,}")
    print(f"  簡體字元數: {len(text_simp):,}")

    # Token 數比較
    print(f"\n🔢 Token 數比較")
    tokens_trad = ollama_tokenize(model, text_trad)
    tokens_simp = ollama_tokenize(model, text_simp)

    if tokens_trad and tokens_simp:
        print(f"  繁體 tokens: {len(tokens_trad)}")
        print(f"  簡體 tokens: {len(tokens_simp)}")
        ratio = len(tokens_trad) / len(tokens_simp) if tokens_simp else 0
        print(f"  繁/簡比值:   {ratio:.4f}")
        print(f"  → 繁體比簡體多 {(ratio - 1) * 100:.1f}% 的 tokens")
    else:
        print("  （tokenize API 不可用，跳過）")

    # 測試繁體
    print(f"\n🔍 測試繁體版本")
    prompt_trad = f"請根據以下文本回答問題。只根據文本中的資訊作答，用簡短的一句話回答。\n\n文本：\n{text_trad}\n\n問題：{QUESTION}\n\n回答："
    response_trad = ollama_generate(model, prompt_trad)
    hit_trad = EXPECTED in response_trad
    print(f"  回答: {response_trad.strip()[:100]}")
    print(f"  包含正確答案: {'✓' if hit_trad else '✗'}")

    # 測試簡體
    print(f"\n🔍 測試簡體版本")
    question_simp = to_simplified(QUESTION)
    expected_simp = to_simplified(EXPECTED)
    prompt_simp = f"请根据以下文本回答问题。只根据文本中的信息作答，用简短的一句话回答。\n\n文本：\n{text_simp}\n\n问题：{question_simp}\n\n回答："
    response_simp = ollama_generate(model, prompt_simp)
    hit_simp = expected_simp in response_simp or EXPECTED in response_simp
    print(f"  回答: {response_simp.strip()[:100]}")
    print(f"  包含正確答案: {'✓' if hit_simp else '✗'}")

    # 總結
    print(f"\n{'═' * 50}")
    print(f"📊 快速測試結果")
    print(f"  繁體 tokens / 簡體 tokens = {len(tokens_trad)} / {len(tokens_simp)}")
    print(f"  繁體正確: {'✓' if hit_trad else '✗'}")
    print(f"  簡體正確: {'✓' if hit_simp else '✗'}")

    if tokens_trad and tokens_simp:
        print(f"\n  如果此結果合理，可以開始完整實驗：")
        print(f"    python scripts/01_fetch_wiki.py")
        print(f"    python scripts/02_build_haystacks.py")
        print(f"    python scripts/03_run_experiment.py --model {model}")
        print(f"    python scripts/04_analyze.py --model {model}")


if __name__ == "__main__":
    main()
