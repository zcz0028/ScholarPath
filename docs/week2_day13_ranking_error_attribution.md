# Week2 Day13-4 — Ranking Error Attribution

## Goal

Day13-3 identified 18 ranking-depth queries:

- 12 `deep_ranked_hit`
- 6 `mid_ranked_hit`

Day13-4 asks why known relevant papers are below the practical result
boundary in frozen Day8 E3.

This is an attribution stage, not a new reranker.

## Inputs

Primary:

```text
outputs/week2_day8_evidence_rerank/predictions_e3_selected.jsonl
outputs/week2_day13_f1_diagnostic/oracle_k_per_query.jsonl
```

Preferred scope artifact:

```text
outputs/week2_day13_f1_failure_decomposition/
    per_query_failure_decomposition.jsonl
```

If the Day13-3 per-query artifact is absent, the script falls back to Oracle-K
rows whose first strict hit is after rank 5.

## Features inspected

The analysis reads only signals already attached to the E3 candidates:

- Day8 final semantic score
- query / expanded-query / title / abstract overlap
- canonical and raw constraint coverage
- B3 semantic/final score
- B4 pre-rerank score
- B4 source count
- B4 best source rank
- Day8 evidence match rate / confidence

No feature is added to production.

## Two diagnostics

### Blocking-FP separation

For each known relevant paper, compare its feature value with false positives
ranked ahead of it under E3.

A feature win rate above 0.5 means that feature often prefers the relevant
paper even though E3 currently ranks the false positive above it.

### Feature-only local counterfactual

Inside the 18 ranking-depth queries only, temporarily sort candidates by one
existing feature and count matched-rank labels captured in Top5/Top20.

This is deliberately **analysis only**. It is not a leaderboard metric and
must never be reported as competition score.

Feature-only reranking is intentionally not recommended for production. The
counterfactual only detects potentially orthogonal signals.

## Decision rule

A final small calibration experiment is justified only when a non-E3 feature:

1. has blocker pairwise win rate >= 0.58; and
2. increases local matched-label capture at Top20.

Even then, the next experiment must be a *blended* protected calibration,
not replacement of E3.

## Boundaries

- API calls = 0
- LLM calls = 0
- E3 ranking modified = false
- production config modified = false
- gold / hit ranks used only for offline attribution

## Outputs

```text
outputs/week2_day13_ranking_error_attribution/
├── query_ranking_attribution.jsonl
├── feature_separation_summary.json
├── feature_counterfactual_summary.json
├── feature_attribution_table.csv
├── optimization_decision.json
└── day13_4_report.md
```

## Run

```bat
python scripts\build_day13_ranking_error_attribution.py
pytest -q tests\test_day13_ranking_error_attribution.py
pytest -q
```
