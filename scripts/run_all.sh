#!/bin/bash
# 自動依序執行所有實驗，每個完成後自動 commit
# 用法: nohup bash scripts/run_all.sh > /tmp/run_all.log 2>&1 &
#
# 特性：
#   - 每個實驗完成後自動 git add + commit
#   - 單個實驗失敗不影響後續（繼續執行）
#   - 所有輸出記錄到 log
#   - 可隨時 kill 暫停，之後重跑會自動 --resume 跳過已完成的

cd "$(dirname "$0")/.." || exit 1

LOG="/tmp/run_all.log"

run_exp() {
    local desc="$1"
    shift
    local cmd="$*"

    echo "" | tee -a "$LOG"
    echo "══════════════════════════════════════════" | tee -a "$LOG"
    echo "  $desc" | tee -a "$LOG"
    echo "  開始: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "  指令: $cmd" | tee -a "$LOG"
    echo "══════════════════════════════════════════" | tee -a "$LOG"

    eval "$cmd" 2>&1 | tee -a "$LOG"
    local exit_code=${PIPESTATUS[0]}

    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  結束: $end_time (exit code: $exit_code)" | tee -a "$LOG"

    # 自動 commit（不論成功失敗都 commit 當前進度）
    git add results/ 2>/dev/null
    if git diff --cached --quiet 2>/dev/null; then
        echo "  （無新結果需要 commit）" | tee -a "$LOG"
    else
        git commit -m "$desc 完成 ($end_time)

dave0629@gmail.com" 2>&1 | tee -a "$LOG"
    fi

    if [ $exit_code -ne 0 ]; then
        echo "  ⚠️ 非正常結束，繼續下一個實驗" | tee -a "$LOG"
    fi
}

unload_model() {
    local model="$1"
    echo "  卸載模型: $model" | tee -a "$LOG"
    curl -s http://localhost:11434/api/generate \
        -d "{\"model\": \"$model\", \"keep_alive\": 0}" > /dev/null 2>&1 || true
}

echo "開始全部實驗排程: $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG"
echo "PID: $$" | tee -a "$LOG"

# 模型順序：gemma4（小→大）→ qwen3.5（小→大）→ 其他
# 長度順序：Phase A 先跑所有模型 100K，最後跑所有模型 130K

# ═══════════════════════════════════════════════════════════
# Phase A: 繁問繁答（以模型為單位，100K 優先，再跑 500-65K）
# ═══════════════════════════════════════════════════════════

run_exp "gemma4:e2b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma4:e2b --variant traditional --lengths 100000 --resume
run_exp "gemma4:e2b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma4:e2b --variant traditional --resume
unload_model "gemma4:e2b"

run_exp "gemma4:e4b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma4:e4b --variant traditional --lengths 100000 --resume
run_exp "gemma4:e4b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma4:e4b --variant traditional --resume
unload_model "gemma4:e4b"

run_exp "gemma4:26b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma4:26b --variant traditional --lengths 100000 --resume
run_exp "gemma4:26b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma4:26b --variant traditional --resume
unload_model "gemma4:26b"

run_exp "gemma4:31b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma4:31b --variant traditional --lengths 100000 --resume
run_exp "gemma4:31b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma4:31b --variant traditional --resume
unload_model "gemma4:31b"

run_exp "qwen3.5:2b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:2b --variant traditional --lengths 100000 --resume
run_exp "qwen3.5:2b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3.5:2b --variant traditional --resume
unload_model "qwen3.5:2b"

run_exp "qwen3.5:4b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:4b --variant traditional --lengths 100000 --resume
run_exp "qwen3.5:4b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3.5:4b --variant traditional --resume
unload_model "qwen3.5:4b"

run_exp "qwen3.5:9b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:9b --variant traditional --lengths 100000 --resume
run_exp "qwen3.5:9b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3.5:9b --variant traditional --resume
unload_model "qwen3.5:9b"

run_exp "qwen3.5:27b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:27b --variant traditional --lengths 100000 --resume
run_exp "qwen3.5:27b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3.5:27b --variant traditional --resume
unload_model "qwen3.5:27b"

run_exp "qwen3.5:35b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:35b --variant traditional --lengths 100000 --resume
run_exp "qwen3.5:35b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3.5:35b --variant traditional --resume
unload_model "qwen3.5:35b"

run_exp "gemma3:1b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma3:1b --variant traditional --lengths 100000 --resume
run_exp "gemma3:1b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma3:1b --variant traditional --resume
unload_model "gemma3:1b"

run_exp "gemma3:4b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma3:4b --variant traditional --lengths 100000 --resume
run_exp "gemma3:4b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma3:4b --variant traditional --resume
unload_model "gemma3:4b"

run_exp "gemma3:12b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma3:12b --variant traditional --lengths 100000 --resume
run_exp "gemma3:12b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma3:12b --variant traditional --resume
unload_model "gemma3:12b"

run_exp "gemma3:27b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model gemma3:27b --variant traditional --lengths 100000 --resume
run_exp "gemma3:27b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model gemma3:27b --variant traditional --resume
unload_model "gemma3:27b"

run_exp "llama3.1:8b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model llama3.1:8b --variant traditional --lengths 100000 --resume
run_exp "llama3.1:8b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model llama3.1:8b --variant traditional --resume
unload_model "llama3.1:8b"

run_exp "llama3.3:70b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model llama3.3:70b --variant traditional --lengths 100000 --resume
run_exp "llama3.3:70b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model llama3.3:70b --variant traditional --resume
unload_model "llama3.3:70b"

run_exp "qwen3:8b 繁問繁答 100K" \
    python3 scripts/03_run_experiment.py --model qwen3:8b --variant traditional --lengths 100000 --resume
run_exp "qwen3:8b 繁問繁答（全部）" \
    python3 scripts/03_run_experiment.py --model qwen3:8b --variant traditional --resume
unload_model "qwen3:8b"

# ═══════════════════════════════════════════════════════════
# Phase A-2: 繁問繁答 130K（全部長度跑完後，由小到大）
# ═══════════════════════════════════════════════════════════

run_exp "gemma4:e2b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma4:e2b --variant traditional --lengths 130000 --resume
unload_model "gemma4:e2b"

run_exp "gemma4:e4b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma4:e4b --variant traditional --lengths 130000 --resume
unload_model "gemma4:e4b"

run_exp "gemma4:26b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma4:26b --variant traditional --lengths 130000 --resume
unload_model "gemma4:26b"

run_exp "gemma4:31b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma4:31b --variant traditional --lengths 130000 --resume
unload_model "gemma4:31b"

run_exp "qwen3.5:2b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:2b --variant traditional --lengths 130000 --resume
unload_model "qwen3.5:2b"

run_exp "qwen3.5:4b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:4b --variant traditional --lengths 130000 --resume
unload_model "qwen3.5:4b"

run_exp "qwen3.5:9b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:9b --variant traditional --lengths 130000 --resume
unload_model "qwen3.5:9b"

run_exp "qwen3.5:27b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:27b --variant traditional --lengths 130000 --resume
unload_model "qwen3.5:27b"

run_exp "qwen3.5:35b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3.5:35b --variant traditional --lengths 130000 --resume
unload_model "qwen3.5:35b"

run_exp "gemma3:1b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma3:1b --variant traditional --lengths 130000 --resume
unload_model "gemma3:1b"

run_exp "gemma3:4b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma3:4b --variant traditional --lengths 130000 --resume
unload_model "gemma3:4b"

run_exp "gemma3:12b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma3:12b --variant traditional --lengths 130000 --resume
unload_model "gemma3:12b"

run_exp "gemma3:27b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model gemma3:27b --variant traditional --lengths 130000 --resume
unload_model "gemma3:27b"

run_exp "llama3.1:8b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model llama3.1:8b --variant traditional --lengths 130000 --resume
unload_model "llama3.1:8b"

run_exp "llama3.3:70b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model llama3.3:70b --variant traditional --lengths 130000 --resume
unload_model "llama3.3:70b"

run_exp "qwen3:8b 繁問繁答 130K" \
    python3 scripts/03_run_experiment.py --model qwen3:8b --variant traditional --lengths 130000 --resume
unload_model "qwen3:8b"

# ═══════════════════════════════════════════════════════════
# Phase C: 簡問簡答（由小到大）
# ═══════════════════════════════════════════════════════════

run_exp "gemma4:e2b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma4:e2b --resume
unload_model "gemma4:e2b"

run_exp "gemma4:e4b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma4:e4b --resume
unload_model "gemma4:e4b"

run_exp "gemma4:26b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma4:26b --resume
unload_model "gemma4:26b"

run_exp "gemma4:31b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma4:31b --resume
unload_model "gemma4:31b"

run_exp "qwen3.5:2b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3.5:2b --resume
unload_model "qwen3.5:2b"

run_exp "qwen3.5:4b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3.5:4b --resume
unload_model "qwen3.5:4b"

run_exp "qwen3.5:9b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3.5:9b --resume
unload_model "qwen3.5:9b"

run_exp "qwen3.5:27b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3.5:27b --resume
unload_model "qwen3.5:27b"

run_exp "qwen3.5:35b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3.5:35b --resume
unload_model "qwen3.5:35b"

run_exp "gemma3:1b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma3:1b --resume
unload_model "gemma3:1b"

run_exp "gemma3:4b 簡問簡答（補跑）" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma3:4b --resume
unload_model "gemma3:4b"

run_exp "gemma3:12b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma3:12b --resume
unload_model "gemma3:12b"

run_exp "gemma3:27b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model gemma3:27b --resume
unload_model "gemma3:27b"

run_exp "llama3.1:8b 簡問簡答（補跑）" \
    python3 scripts/06_hypothesis2_simp_question.py --model llama3.1:8b --resume
unload_model "llama3.1:8b"

run_exp "llama3.3:70b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model llama3.3:70b --resume
unload_model "llama3.3:70b"

run_exp "qwen3:8b 簡問簡答" \
    python3 scripts/06_hypothesis2_simp_question.py --model qwen3:8b --resume
unload_model "qwen3:8b"

# ═══════════════════════════════════════════════════════════
# Phase D: 繁問簡答（由小到大）
# ═══════════════════════════════════════════════════════════

run_exp "gemma4:e2b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma4:e2b --variant simplified --resume
unload_model "gemma4:e2b"

run_exp "gemma4:e4b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma4:e4b --variant simplified --resume
unload_model "gemma4:e4b"

run_exp "gemma4:26b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma4:26b --variant simplified --resume
unload_model "gemma4:26b"

run_exp "gemma4:31b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma4:31b --variant simplified --resume
unload_model "gemma4:31b"

run_exp "qwen3.5:2b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3.5:2b --variant simplified --resume
unload_model "qwen3.5:2b"

run_exp "qwen3.5:4b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3.5:4b --variant simplified --resume
unload_model "qwen3.5:4b"

run_exp "qwen3.5:9b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3.5:9b --variant simplified --resume
unload_model "qwen3.5:9b"

run_exp "qwen3.5:27b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3.5:27b --variant simplified --resume
unload_model "qwen3.5:27b"

run_exp "qwen3.5:35b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3.5:35b --variant simplified --resume
unload_model "qwen3.5:35b"

run_exp "gemma3:1b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma3:1b --variant simplified --resume
unload_model "gemma3:1b"

run_exp "gemma3:4b 繁問簡答（補跑）" \
    python3 scripts/03_run_experiment.py --model gemma3:4b --variant simplified --resume
unload_model "gemma3:4b"

run_exp "gemma3:12b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma3:12b --variant simplified --resume
unload_model "gemma3:12b"

run_exp "gemma3:27b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model gemma3:27b --variant simplified --resume
unload_model "gemma3:27b"

run_exp "llama3.1:8b 繁問簡答（補跑）" \
    python3 scripts/03_run_experiment.py --model llama3.1:8b --variant simplified --resume
unload_model "llama3.1:8b"

run_exp "llama3.3:70b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model llama3.3:70b --variant simplified --resume
unload_model "llama3.3:70b"

run_exp "qwen3:8b 繁問簡答" \
    python3 scripts/03_run_experiment.py --model qwen3:8b --variant simplified --resume
unload_model "qwen3:8b"

# ═══════════════════════════════════════════════════════════

echo "" | tee -a "$LOG"
echo "全部實驗完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
