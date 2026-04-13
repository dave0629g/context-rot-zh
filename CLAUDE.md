# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Needle-in-a-Haystack (NIAH) experiment testing whether Traditional Chinese tokenizer fragmentation impacts LLM long-context retrieval. Three variants are compared:

- **繁問繁答** (`traditional`): Traditional Chinese context + question
- **繁問簡答** (`simplified`): Simplified Chinese context + Traditional Chinese question
- **簡問簡答** (`simplified_q`): Simplified Chinese context + question (via `06_hypothesis2_simp_question.py`)

Experiments span 12 context lengths (500–130K chars), 11 needle positions, 5 needles, 10 repetitions = **1,320 trials per variant per model**.

## Common Commands

### Run Experiments

```bash
# Single model, single variant
python scripts/03_run_experiment.py --model gemma3:4b --variant traditional

# Hypothesis 2 variant (simplified question + context)
python scripts/06_hypothesis2_simp_question.py --model gemma3:4b

# Resume from interruption
python scripts/03_run_experiment.py --model gemma3:4b --resume

# Target specific context lengths
python scripts/03_run_experiment.py --model gemma3:4b --variant traditional --lengths 65000 100000 130000

# Automated batch runner (all models, all variants, with git commits)
nohup bash scripts/run_all.sh > /tmp/run_all.log 2>&1 &
```

### Monitor Progress

```bash
watch --color -n 5 bash scripts/watch_progress.sh
python scripts/estimate_time.py
tail -f /tmp/run_all.log
```

### Analysis & Visualization

```bash
python scripts/04_analyze.py --model gemma3:4b --reeval   # Re-evaluate single model
python scripts/04_analyze.py --all --reeval               # All models
python scripts/05_plot_results.py --models gemma3:4b llama3.1:8b
python scripts/07_export_web.py                           # Export to docs/data.json
```

### Interactive Dashboard

```bash
streamlit run app.py        # Local Streamlit at http://localhost:8501
```

## Architecture

### Data Pipeline

```
01_fetch_wiki_v2.py   → data/wiki_raw_v2/zh/*.txt
02_build_haystacks.py → data/haystacks/experiments.jsonl  (~102MB)
03_run_experiment.py  → results/{model}_results.jsonl
06_hypothesis2_*.py   → results/h2_{model}_results.jsonl
04_analyze.py         → results/analysis/{model}_analysis.json
05_plot_results.py    → results/plots/*.png
07_export_web.py      → docs/data.json
```

Steps 1–2 are one-time setup. Steps 3/6 are the long-running experiment phase.

### Result File Formats

**`results/{model}_results.jsonl`** — one JSON object per line:
```json
{
  "model": "gemma3:4b", "variant": "traditional",
  "experiment_id": "...", "context_length_chars": 100000,
  "needle_position_pct": 50, "model_answer": "...",
  "expected_answer": "...", "elapsed_seconds": 45.3,
  "tokens_trad": 15000, "tokens_simp": 14200, "skipped": false
}
```

Hypothesis 2 results use `h2_` prefix: `results/h2_{model}_results.jsonl`.

### Accuracy Evaluation (`04_analyze.py`)

Re-evaluation applies in order: substring match → Arabic numeral match → Chinese numeral normalization (e.g., 四百七十三億 → 47.3B). This corrects for original experiment logic gaps.

## Key Configuration

**`configs/wiki_articles_v2.json`** — needle definitions and expected answers (always in Traditional Chinese).

**Fixed experiment parameters:**
- Quantization: Q4_K_M for all models
- temperature: 0.0 (greedy)
- thinking/reasoning: disabled for qwen3/gemma4
- API timeout: 1200s (needed for 70B at 130K context)

## Known Limitations

- **qwen3:8b** context window is 40K, not 131K — experiments ≥65K are truncated
- **N04 needle** (black bear) has high distractor risk in haystack; absolute accuracy on N04 may be artificially inflated/deflated, but relative variant comparisons remain valid
- Ollama must be running (`ollama serve`) before any experiment scripts
