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
    # 水部常見字
    water = set("海洋河湖江溪流波浪潮湧淺深沿港灣泊沉浮漂溫溶液泉瀑渠池沼澤淡清濁漲洪淹溢渡灌溉氾濕潤滲滴涓淋漫濃淵")
    # 金部常見字
    metal = set("鐵鋼銅鋁鋅錫鉛鎳鉻鎢鑄鍛鍊鍍釘鏈錠鑽銀鈣鉀鈉鋰鈷錳銻鉍鉬鉭鈮釕銥鉑鈀銠鑭鈰鋯鉿鈦釩鈧金針鑰鍋鋸鏡銘")
    # 木部常見字
    wood = set("林森樹木枝根棵株桿板柱棚架柵欄杉松柏楓櫻橡榕桃梅柳桂橘柿椰棕樺榆槐檀梧桐楠榴椿楊榭樁橋柩棺棍桶框")
    # 土部常見字
    earth = set("地坡塊堆埋填基壁壤塘坑坪垣堤壩塔墊墓墳塑堡壘垃圾坊坎均坦坐堅執培域堂場塵境增壓墨壯")
    # 火部常見字
    fire = set("火焰燃燒熔煙熱烈炸爆灰灼烘炙煮烤蒸熏炒燉燜煎燙焚煥燦熄熠煌燿烙焊熬煉煤燈爐灶灸烯烴烷")
    # 肉部常見字（月旁）
    flesh = set("臟腑腸胃肝脾腎肺腦膜腺肌膚脂膝臂腿腰胸腹背肩肘膀胎胚胞脈腫瘤脹膨臉脖腳踝")

    if char in water: return "水"
    if char in metal: return "金"
    if char in wood: return "木"
    if char in earth: return "土"
    if char in fire: return "火"
    if char in flesh: return "肉"
    return "other"


def analyze_text_properties(text: str) -> dict:
    """分析一段文本的語言特性"""
    total_chars = len(text)
    if total_chars == 0:
        return {}

    # 中文字元
    chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    chinese_count = len(chinese_chars)

    # 繁簡差異字密度
    trad_diff_count = sum(1 for c in chinese_chars if c in TRAD_SIMP_DIFF_CHARS)

    # 部首分布
    radical_counts = {}
    for c in chinese_chars:
        r = classify_radical(c)
        radical_counts[r] = radical_counts.get(r, 0) + 1

    # 平均句長（用句號、問號、驚嘆號分句）
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

def fetch_wiki(title: str, lang: str = "zh", variant: str = "zh-tw") -> str | None:
    """從維基百科 API 取得純文字"""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "exsectionformat": "plain",
        "format": "json",
    }
    if variant:
        params["variant"] = variant

    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "ContextRotZH/2.0 (Academic Research; github.com/context-rot-zh)"
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f" 錯誤: {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None
        return page_data.get("extract", "")
    return None


def main():
    os.makedirs(ZH_DIR, exist_ok=True)
    os.makedirs(EN_DIR, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    articles = [a for a in config["articles"] if "title" in a]

    print(f"準備下載 {len(articles)} 篇條目\n")

    corpus_metadata = []
    zh_success = 0
    en_success = 0

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
            print(f"  [{i+1:2d}/{len(articles)}] 已存在: {title}")
        else:
            print(f"  [{i+1:2d}/{len(articles)}] 下載中文: {title}", end="", flush=True)
            zh_text = fetch_wiki(title, lang="zh", variant="zh-tw")
            if zh_text and len(zh_text) > 300:
                with open(zh_path, "w", encoding="utf-8") as f:
                    f.write(zh_text)
                print(f" → {len(zh_text):,} 字元 ✓")
            else:
                zh_text = ""
                print(f" → 內容不足 ✗")
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
                en_text = fetch_wiki(en_title, lang="en", variant=None)
                if en_text and len(en_text) > 300:
                    with open(en_path, "w", encoding="utf-8") as f:
                        f.write(en_text)
                    print(f" → {len(en_text):,} chars ✓")
                    en_success += 1
                    meta["en_file"] = os.path.basename(en_path)
                else:
                    print(f" → insufficient ✗")
                time.sleep(1)

        corpus_metadata.append(meta)

    # 儲存 metadata
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "download_date": time.strftime("%Y-%m-%d"),
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
    print(f"\n📊 各方向語料覆蓋")
    direction_counts = {}
    for meta in corpus_metadata:
        if "zh_properties" not in meta:
            continue
        for d in meta.get("directions", []):
            if d not in direction_counts:
                direction_counts[d] = {"count": 0, "total_chars": 0}
            direction_counts[d]["count"] += 1
            direction_counts[d]["total_chars"] += meta["zh_properties"]["total_chars"]

    for d, stats in sorted(direction_counts.items()):
        print(f"  {d:25s}: {stats['count']:2d} 篇, {stats['total_chars']:>10,} 字元")

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
