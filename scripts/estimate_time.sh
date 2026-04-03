#!/bin/bash
# 實驗時間估算（含家族分組）
# 用法: watch --color -n 30 bash scripts/estimate_time.sh

cd "$(dirname "$0")/.." || exit 1
python3 scripts/estimate_time.py
