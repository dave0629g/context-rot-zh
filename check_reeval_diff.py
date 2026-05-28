#!/usr/bin/env python3
"""比對 results vs results.backup_pre_reeval 的 is_correct 差異"""
import json
import os
from glob import glob

results_dir = "results"
backup_dir = "results.backup_pre_reeval"

print(f"比對 {results_dir}/ vs {backup_dir}/\n")
print(f"{'檔案':50s} {'raw_acc':>10s} {'reeval_acc':>12s} {'changed':>10s} {'changed_%':>10s}")
print("-" * 100)

for new_path in sorted(glob(f"{results_dir}/*.jsonl")):
    fname = os.path.basename(new_path)
    old_path = f"{backup_dir}/{fname}"
    if not os.path.exists(old_path):
        print(f"{fname:50s} ⚠️ backup 不存在")
        continue
    
    new_recs = {}
    with open(new_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                eid = rec.get("experiment_id")
                if eid is not None:
                    new_recs[eid] = rec.get("evaluation", {}).get("is_correct")
            except json.JSONDecodeError:
                continue
    
    old_recs = {}
    with open(old_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                eid = rec.get("experiment_id")
                if eid is not None:
                    old_recs[eid] = rec.get("evaluation", {}).get("is_correct")
            except json.JSONDecodeError:
                continue
    
    # 交集
    common_ids = set(new_recs.keys()) & set(old_recs.keys())
    if not common_ids:
        print(f"{fname:50s} ⚠️ 無共同 experiment_id")
        continue
    
    raw_correct = sum(1 for eid in common_ids if old_recs[eid] is True)
    reeval_correct = sum(1 for eid in common_ids if new_recs[eid] is True)
    changed = sum(1 for eid in common_ids if old_recs[eid] != new_recs[eid])
    
    n = len(common_ids)
    raw_acc = raw_correct / n if n else 0
    reeval_acc = reeval_correct / n if n else 0
    changed_pct = changed / n * 100 if n else 0
    
    print(f"{fname:50s} {raw_acc:>10.4f} {reeval_acc:>12.4f} {changed:>10d} {changed_pct:>10.2f}%")
