# Week2 Day11 — Evaluation & Ablation

## Goal

Day11 freezes ScholarPath's competition evaluation story after the retrieval, ranking, evidence, citation, and search-reasoning capabilities are implemented. The goal is not to introduce new retrieval algorithms, but to measure the frozen system consistently and provide one reusable source of truth for later reports, slides, demo narration, and defense preparation.

## Evaluation policy

1. Gold data is used for evaluation/analysis only.
2. The Day11 aggregator performs no network calls.
3. The Day11 aggregator performs no LLM calls.
4. Evidence, recommendation reasons, citation paths, and search reasoning do not modify the frozen Day8 E3 ranking.
5. Primary retrieval comparison uses strict identity matching.

## Frozen headline results

- B0 Top100 TP: 44
- B4 Top100 TP: 61
- Day4 Top100 TP: 105
- B4 → Day4 Top100 TP gain: +44
- Day8 baseline NDCG@10: 0.19291929976021374
- Day8 E3 NDCG@10: 0.2488522823567324
- Relative NDCG@10 gain: 28.99294299017245%
- Top100 Constraint Evidence Match Rate: 13.548223350253807%
- Top100 Paper Evidence Support Rate: 48.82%
- Top100 deterministic Reason Generation Rate: 100%
- Top100 Evidence-backed Reason Rate: 48.82%
- Frozen E3 Abstract Availability Rate: 0%
- Day9 RealScholarQuery_16 Top20 Citation Path case-study coverage: 19/20 = 95%

## Reporting boundaries

The 100% reason-generation number is an availability metric, not explanation accuracy. Evidence-backed Reason Rate is the stronger support metric.

The 95% citation-path value is a single-query Day9 case study and must not be reported as full 50-query benchmark citation coverage.

Evidence/Citation/Search Reasoning are explanation layers and must not be presented as if they improved the frozen E3 ranking.

## Unified outputs

Running:

```bat
python scripts\build_day11_evaluation_report.py
```

generates:

```text
outputs/week2_day11_evaluation/
├── evaluation_summary.json
├── retrieval_ranking_summary.json
├── explainability_summary.json
├── competition_metrics.json
├── retrieval_ablation.csv
├── ranking_ablation.csv
└── evaluation_report.md
```

`competition_metrics.json` is the compact source for PPT/report headline numbers. `evaluation_report.md` is the human-readable experiment report.

## Competition narrative

```text
Raw Retrieval
→ Candidate Fusion
→ Rescue Retrieval
→ Semantic Reranking
→ Evidence / Reason
→ Citation Path
→ Search Reasoning
```

Each stage is evaluated according to its actual role: retrieval stages use recall/TP/recovery metrics; ranking uses NDCG@10; evidence and reasons use structured support coverage; citation uses scoped traceability coverage; search reasoning is user-facing system explainability rather than private chain-of-thought.
