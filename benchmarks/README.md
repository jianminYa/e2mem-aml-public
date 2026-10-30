# Offline Evaluation Benchmarks (Development Set)

This directory contains summary metrics from local offline development benchmarks evaluated during the development of E²-Mem Evidence Anchor (v2.6.1).

## Scope & Dataset Origin
- **Nature**: Local development evaluation results across 248 benchmark questions curated from public research datasets (including LoCoMo, LongMemEval, PersonaMem, and CLBench).
- **Reader & Evaluator**: Evaluated locally using `gpt-4o-mini-2024-07-18` as the downstream answer generator and LLM judge.
- **Distinction**: These metrics reflect offline local research ablation experiments and are **not official Agent Memory Leaderboard (AML) competition results**.
- **Data Compliance**: Detailed question-by-question trace logs containing source dataset text are omitted in accordance with upstream research licenses. Aggregate performance metrics are retained in `evaluation_results/summary.json`.
