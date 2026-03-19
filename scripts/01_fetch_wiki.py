"""
Step 1: 從維基百科下載繁體中文條目

用法: python scripts/01_fetch_wiki.py

輸出: data/wiki_raw/ 目錄下的純文字檔案
"""

import json
import os
import time
import urllib.request
import urllib.parse

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wiki_raw")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "wiki_articles.json")


def fetch_wiki_article(title: str, lang: str = "zh") -> str:
    """透過 Wikipedia API 取得條目純文字內容（繁體中文）"""
    
    # 使用 action=query 取得純文字
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",    # 純文字，不要 HTML
        "exsectionformat": "plain",
        "format": "json",
        "variant": "zh-tw",       # 指定繁體中文
    }
    
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "ContextRotExperiment/1.0 (Academic Research)"
    })
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None  # 條目不存在
        return page_data.get("extract", "")
    
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    articles = config["articles"]
    success_count = 0
    fail_count = 0
    
    print(f"準備下載 {len(articles)} 篇維基百科條目...\n")
    
    for i, article in enumerate(articles):
        title = article["title"]
        category = article["category"]
        
        # 檔名：用序號+標題，避免特殊字元問題
        safe_title = title.replace("/", "_").replace(" ", "_")
        filename = f"{i:02d}_{safe_title}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        # 如果已下載過就跳過
        if os.path.exists(filepath):
            print(f"  [{i+1:2d}/{len(articles)}] 已存在，跳過: {title}")
            success_count += 1
            continue
        
        print(f"  [{i+1:2d}/{len(articles)}] 下載中: {title} ({category})", end="")
        
        try:
            text = fetch_wiki_article(title)
            
            if text and len(text) > 500:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                char_count = len(text)
                print(f" → {char_count:,} 字元 ✓")
                success_count += 1
            else:
                print(f" → 內容不足，跳過 ✗")
                fail_count += 1
                
        except Exception as e:
            print(f" → 錯誤: {e} ✗")
            fail_count += 1
        
        # 避免請求過快
        time.sleep(1)
    
    print(f"\n完成: 成功 {success_count}, 失敗 {fail_count}")
    print(f"檔案位置: {DATA_DIR}")
    
    # 輸出語料統計
    print("\n=== 語料統計 ===")
    total_chars = 0
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            char_count = len(text)
            total_chars += char_count
            print(f"  {filename}: {char_count:,} 字元")
    print(f"\n  總計: {total_chars:,} 字元")


if __name__ == "__main__":
    main()
