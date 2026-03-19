"""
Step 1 (v2): 下載維基百科條目並標記方向

根據 wiki_articles_v2.json 的設定：
  - 下載所有繁體中文條目
  - 下載有 en_title 的英文對照版本（方向三需要）
  - 計算每篇文章的語言特性統計（部首密度、繁簡差異字密度等）
  - 輸出帶有完整 metadata 的語料庫

用法: python scripts/01_fetch_wiki_v2.py

輸出:
  data/wiki_raw_v2/zh/{title}.txt       繁體中文版
  data/wiki_raw_v2/en/{title}.txt       英文版（有 en_title 的條目）
  data/wiki_raw_v2/corpus_metadata.json 語料庫 metadata
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "wiki_articles_v2.json")
ZH_DIR = os.path.join(BASE_DIR, "data", "wiki_raw_v2", "zh")
EN_DIR = os.path.join(BASE_DIR, "data", "wiki_raw_v2", "en")
META_PATH = os.path.join(BASE_DIR, "data", "wiki_raw_v2", "corpus_metadata.json")

# 每篇文章的最低字元數門檻（低於此視為內容不足）
MIN_CHARS = 3000

# 各方向所需的最低文章數
DIRECTION_MIN_COUNTS = {
    "D1_fragmentation":    8,
    "D2_no_boundary":      7,
    "D3_semantic_density": 8,
    "D4_positional_bias":  7,
    "D5_glyph_interference": 10,
    "baseline":            5,
}


# --- 語言特性分析工具 ---

# 常見繁簡差異字（繁體版本）
# 這些字在簡體中有不同寫法，會導致 tokenizer 切法不同
TRAD_SIMP_DIFF_CHARS = set(
    "機學國發關體點數運計網絡訓練積電製處術線備達開場經濟區連結陣應義務業質員環認識"
    "頭從復歷車軍門間雲龍風飛馬魚鳥黃齒團圖園壓報執塊夢獎導島歲師帶廠廣歸彈態懷"
    "戰擊權殘氣決沒準滅滿漢災為熱營獨獻環產當畫異療發確種節範紀級組終網練總義習聲"
    "與舉華號術複觀訊許論議護變讓賣農達選鄰鐵鑑關電離雜難類項飛驗體黨齊齡龍"
)

# 部首分類（簡化版，用 Unicode 範圍近似）
def classify_radical(char: str) -> str:
    """根據常見字歸類部首（簡化啟發式方法）"""
    water = set("海洋河湖江溪流波浪潮湧淺深沿港灣泊沉浮漂溫溶液泉瀑渠池沼澤淡清濁漲洪淹溢渡灌溉氾濕潤滲滴涓淋漫濃淵")
    metal = set("鐵鋼銅鋁鋅錫鉛鎳鉻鎢鑄鍛鍊鍍釘鏈錠鑽銀鈣鉀鈉鋰鈷錳銻鉍鉬鉭鈮釕銥鉑鈀銠鑭鈰鋯鉿鈦釩鈧金針鑰鍋鋸鏡銘")
    wood  = set("林森樹木枝根棵株桿板柱棚架柵欄杉松柏楓櫻橡榕桃梅柳桂橘柿椰棕樺榆槐檀梧桐楠榴椿楊榭樁橋柩棺棍桶框")
    earth = set("地坡塊堆埋填基壁壤塘坑坪垣堤壩塔墊墓墳塑堡壘垃圾坊坎均坦坐堅執培域堂場塵境增壓墨壯")
    fire  = set("火焰燃燒熔煙熱烈炸爆灰灼烘炙煮烤蒸熏炒燉燜煎燙焚煥燦熄熠煌燿烙焊熬煉煤燈爐灶灸烯烴烷")
    flesh = set("臟腑腸胃肝脾腎肺腦膜腺肌膚脂膝臂腿腰胸腹背肩肘膀胎胚胞脈腫瘤脹膨臉脖腳踝")

    if char in water: return "水"
    if char in metal: return "金"
    if char in wood:  return "木"
    if char in earth: return "土"
    if char in fire:  return "火"
    if char in flesh: return "肉"
    return "other"


def analyze_text_properties(text: str) -> dict:
    """分析一段文本的語言特性"""
    total_chars = len(text)
    if total_chars == 0:
        return {}

    chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    chinese_count = len(chinese_chars)

    trad_diff_count = sum(1 for c in chinese_chars if c in TRAD_SIMP_DIFF_CHARS)

    radical_counts = {}
    for c in chinese_chars:
        r = classify_radical(c)
        radical_counts[r] = radical_counts.get(r, 0) + 1

    sentences = re.split(r"[。？！?!]", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)

    return {
        "total_chars": total_chars,
        "chinese_chars": chinese_count,
        "chinese_ratio": round(chinese_count / total_chars, 4),
        "trad_diff_char_count": trad_diff_count,
        "trad_diff_char_density": round(trad_diff_count / max(chinese_count, 1), 4),
        "radical_distribution": radical_counts,
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_len, 1),
    }


# --- 維基百科 API ---

def fetch_wiki(title: str, lang: str = "zh", variant: str = "zh-tw",
               retries: int = 3) -> tuple[str | None, str]:
    """
    從維基百科 API 取得純文字。

    回傳 (text, status)：
      text   : 文章純文字，失敗時為 None
      status : "ok" | "not_found" | "error:<msg>"
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "exsectionformat": "plain",
        "redirects": "1",          # 自動跟隨重定向（如 台灣歷史 → 臺灣歷史）
        "format": "json",
    }
    if variant:
        params["variant"] = variant

    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "ContextRotZH/2.0 (Academic Research; github.com/context-rot-zh)"
    })

    data = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break  # 成功則跳出重試迴圈
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f" [重試 {attempt+1}/{retries-1}，等待 {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                return None, f"error:{e}"

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None, "not_found"
        return page_data.get("extract", ""), "ok"
    return None, "not_found"


def main():
    os.makedirs(ZH_DIR, exist_ok=True)
    os.makedirs(EN_DIR, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    articles = [a for a in config["articles"] if "title" in a]

    print(f"準備下載 {len(articles)} 篇條目（最低門檻：{MIN_CHARS:,} 字元）\n")

    corpus_metadata = []
    zh_success = 0
    en_success = 0
    failures = []  # 記錄失敗條目，供最後彙整

    for i, article in enumerate(articles):
        title = article["title"]
        directions = article.get("directions", [])
        en_title = article.get("en_title")
        safe_title = re.sub(r"[/\\:*?\"<>|]", "_", title)

        meta = {
            "index": i,
            "title": title,
            "directions": directions,
            "en_title": en_title,
        }

        # --- 下載繁體中文 ---
        zh_path = os.path.join(ZH_DIR, f"{i:02d}_{safe_title}.txt")
        if os.path.exists(zh_path):
            with open(zh_path, "r", encoding="utf-8") as f:
                zh_text = f.read()
            char_count = len(zh_text)
            status_str = f"{char_count:,} 字元（快取）"
            if char_count < MIN_CHARS:
                # 快取檔案字元數不足，刪除並重新下載
                os.remove(zh_path)
                zh_text = ""
                status_str = f"快取不足 ({char_count:,} 字元)，重新下載..."
            print(f"  [{i+1:2d}/{len(articles)}] {title}  {status_str}")
        else:
            zh_text = ""

        if not zh_text:
            print(f"  [{i+1:2d}/{len(articles)}] 下載中文: {title}", end="", flush=True)
            text, status = fetch_wiki(title, lang="zh", variant="zh-tw")

            if status == "not_found":
                print(f" → 條目不存在 ✗")
                failures.append({"title": title, "lang": "zh", "reason": "not_found"})
            elif status.startswith("error"):
                print(f" → 網路錯誤: {status[6:]} ✗")
                failures.append({"title": title, "lang": "zh", "reason": status})
            elif not text or len(text) < MIN_CHARS:
                actual = len(text) if text else 0
                print(f" → 內容不足 ({actual:,} 字元，需 ≥ {MIN_CHARS:,}) ✗")
                failures.append({"title": title, "lang": "zh",
                                  "reason": f"too_short:{actual}"})
            else:
                with open(zh_path, "w", encoding="utf-8") as f:
                    f.write(text)
                zh_text = text
                print(f" → {len(zh_text):,} 字元 ✓")

            time.sleep(1)

        if zh_text:
            zh_success += 1
            meta["zh_file"] = os.path.basename(zh_path)
            meta["zh_properties"] = analyze_text_properties(zh_text)

        # --- 下載英文版（如果有 en_title）---
        if en_title:
            en_safe = re.sub(r"[/\\:*?\"<>|]", "_", en_title)
            en_path = os.path.join(EN_DIR, f"{i:02d}_{en_safe}.txt")
            if os.path.exists(en_path):
                print(f"           英文版已存在: {en_title}")
                en_success += 1
                meta["en_file"] = os.path.basename(en_path)
            else:
                print(f"           下載英文: {en_title}", end="", flush=True)
                en_text, status = fetch_wiki(en_title, lang="en", variant=None)

                if status == "not_found":
                    print(f" → not found ✗")
                    failures.append({"title": en_title, "lang": "en", "reason": "not_found"})
                elif status.startswith("error"):
                    print(f" → error: {status[6:]} ✗")
                    failures.append({"title": en_title, "lang": "en", "reason": status})
                elif not en_text or len(en_text) < MIN_CHARS:
                    actual = len(en_text) if en_text else 0
                    print(f" → insufficient ({actual:,} chars, need ≥ {MIN_CHARS:,}) ✗")
                    failures.append({"title": en_title, "lang": "en",
                                      "reason": f"too_short:{actual}"})
                else:
                    with open(en_path, "w", encoding="utf-8") as f:
                        f.write(en_text)
                    print(f" → {len(en_text):,} chars ✓")
                    en_success += 1
                    meta["en_file"] = os.path.basename(en_path)

                time.sleep(1)

        corpus_metadata.append(meta)

    # 儲存 metadata
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "download_date": time.strftime("%Y-%m-%d"),
            "min_chars_threshold": MIN_CHARS,
            "zh_articles": zh_success,
            "en_articles": en_success,
            "articles": corpus_metadata,
        }, f, ensure_ascii=False, indent=2)

    # --- 統計報告 ---
    print(f"\n{'═' * 60}")
    print(f"  下載完成")
    print(f"  中文: {zh_success}/{len(articles)} 篇")
    print(f"  英文: {en_success}/{sum(1 for a in articles if a.get('en_title'))} 篇")
    print(f"  Metadata: {META_PATH}")
    print(f"{'═' * 60}")

    # 各方向的語料覆蓋率
    direction_counts = {}
    for meta in corpus_metadata:
        if "zh_properties" not in meta:
            continue
        for d in meta.get("directions", []):
            if d not in direction_counts:
                direction_counts[d] = {"count": 0, "total_chars": 0}
            direction_counts[d]["count"] += 1
            direction_counts[d]["total_chars"] += meta["zh_properties"]["total_chars"]

    print(f"\n📊 各方向語料覆蓋驗證")
    all_ok = True
    for d, req_count in DIRECTION_MIN_COUNTS.items():
        actual = direction_counts.get(d, {}).get("count", 0)
        total_chars = direction_counts.get(d, {}).get("total_chars", 0)
        ok = actual >= req_count
        icon = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {icon} {d:25s}: {actual:2d}/{req_count} 篇  ({total_chars:>10,} 字元)")
    if not all_ok:
        print("\n  ⚠️  部分方向語料不足，請補充條目後重新執行")

    # 失敗彙整
    if failures:
        print(f"\n⚠️  失敗條目（共 {len(failures)} 筆）")
        for f_item in failures:
            print(f"  [{f_item['lang']}] {f_item['title']}  →  {f_item['reason']}")

    # 繁簡差異字密度排行
    print(f"\n📊 繁簡差異字密度 Top 10")
    ranked = sorted(
        [m for m in corpus_metadata if "zh_properties" in m],
        key=lambda m: m["zh_properties"]["trad_diff_char_density"],
        reverse=True,
    )
    for m in ranked[:10]:
        density = m["zh_properties"]["trad_diff_char_density"]
        print(f"  {density:.4f}  {m['title']}")

    # 部首密度報告
    print(f"\n📊 特定部首密度 Top 5 (各部首)")
    for target_radical in ["水", "金", "木", "土", "火", "肉"]:
        print(f"\n  [{target_radical}部]")
        items = []
        for m in corpus_metadata:
            if "zh_properties" not in m:
                continue
            dist = m["zh_properties"]["radical_distribution"]
            chinese_total = m["zh_properties"]["chinese_chars"]
            if chinese_total == 0:
                continue
            count = dist.get(target_radical, 0)
            density = count / chinese_total
            items.append((density, count, m["title"]))
        for density, count, title in sorted(items, reverse=True)[:5]:
            print(f"    {density:.4f} ({count:4d}字)  {title}")


if __name__ == "__main__":
    main()
