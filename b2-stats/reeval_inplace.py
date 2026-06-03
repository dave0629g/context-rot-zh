"""
把 results/*_results.jsonl 與 results/h2_*_results.jsonl 用 reevaluate() 重寫：
  - 只更新 evaluation.is_correct
  - 加入 evaluation.reevaluated = True 旗標
  - 保留 exact_match / number_match / char_overlap 等原值
  - 寫入前完整備份到 results.backup_pre_reeval/
  - 原子寫入（先寫 .tmp 再 rename）
  - 已有 reevaluated=True 的記錄會跳過重算（冪等）
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import importlib.util

REPO_ROOT = Path(__file__).parent / "context-rot-zh"
RESULTS = REPO_ROOT / "results"
BACKUP = REPO_ROOT / "results.backup_pre_reeval"

# 載入 reevaluate
_spec = importlib.util.spec_from_file_location("_analyze",
                                                REPO_ROOT / "scripts" / "04_analyze.py")
_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyze)
reevaluate = _analyze.reevaluate


def list_jsonl_files() -> list[Path]:
    return sorted(RESULTS.glob("*_results.jsonl"))


def backup_once() -> None:
    if BACKUP.exists():
        print(f"備份目錄已存在，跳過備份：{BACKUP}")
        return
    BACKUP.mkdir(parents=True)
    n = 0
    for p in list_jsonl_files():
        shutil.copy2(p, BACKUP / p.name)
        n += 1
    print(f"已備份 {n} 個檔案到 {BACKUP}")


def reeval_file(path: Path) -> dict:
    """逐行重算並原子寫回。回傳統計。"""
    n_total = 0
    n_changed = 0
    n_already = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(path, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write(line + "\n")
                continue
            r = json.loads(line)
            n_total += 1
            if r.get("skipped"):
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            ev = r.setdefault("evaluation", {})
            if ev.get("reevaluated") is True:
                n_already += 1
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            old = bool(ev.get("is_correct", False))
            new = bool(reevaluate(r))
            ev["is_correct"] = new
            ev["reevaluated"] = True
            if new != old:
                n_changed += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 原子取代
    tmp.replace(path)
    return {"file": path.name, "n_total": n_total, "n_changed": n_changed,
            "n_already_reeval": n_already}


def main():
    files = list_jsonl_files()
    print(f"目標：{len(files)} 個 JSONL 檔")
    backup_once()
    stats = []
    for p in files:
        s = reeval_file(p)
        stats.append(s)
        print(f"  {s['file']:40s}  total={s['n_total']:>5}  "
              f"changed={s['n_changed']:>5}  already={s['n_already_reeval']:>5}")
    total = sum(s["n_total"] for s in stats)
    changed = sum(s["n_changed"] for s in stats)
    already = sum(s["n_already_reeval"] for s in stats)
    print(f"\n總計：files={len(files)}  records={total}  changed={changed}  "
          f"already_reeval={already}")


if __name__ == "__main__":
    main()
