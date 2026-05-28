#!/usr/bin/env python3
"""驗證 results/*.jsonl 是否全部已 reeval"""
import json
import os
from collections import Counter
from glob import glob

results_dir = "results"
print(f"檢查 {results_dir}/ 內所有 jsonl\n")

for path in sorted(glob(f"{results_dir}/*.jsonl")):
    fname = os.path.basename(path)
    reeval_flags = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                total += 1
                flag = rec.get("evaluation", {}).get("reevaluated", None)
                reeval_flags[str(flag)] += 1
            except json.JSONDecodeError:
                continue
    
    flag_str = ", ".join(f"{k}={v}" for k, v in reeval_flags.items())
    print(f"{fname:50s} total={total:>5} | reevaluated: {flag_str}")
