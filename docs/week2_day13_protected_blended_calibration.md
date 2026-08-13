# Week2 Day13-5 — Protected Blended Calibration Ablation

## Goal

Day13-4 found one existing retrieval-prior signal with enough local ranking
separation to justify a final protected experiment:

```text
b4_best_source_rank
```

Day13-5 tests one hypothesis only:

> Can a very small B4 source-rank prior improve frozen E3 ranking on held-out
> queries without damaging NDCG or retrieval coverage?

This is the final ranking Go / No-Go experiment for Day13.

## Formula

For every query:

```text
e3_norm = minmax(day8_final_score)
source_prior = 1 / log2(b4_best_source_rank + 1)

blended =
    (1 - beta) * e3_norm
    + beta * source_prior
```

Missing B4 source ranks receive prior `0`.

The default beta grid is intentionally tiny:

```text
0.00, 0.02, 0.04, 0.06, 0.08, 0.10
```

`beta=0` is the protected frozen-E3 baseline.

## Query-level CV

The 50 benchmark queries are split into five deterministic query-level folds.

For each fold:

1. 40 training queries select beta.
2. The objective is training Macro-F1@5.
3. Ties are broken by NDCG@10, then by smaller beta.
4. The selected beta is frozen.
5. The 10 validation queries are reranked using only E3/B4 features.
6. Validation gold is used only after inference for evaluation.

The final result is one 50-query OOF prediction set.

## Acceptance criteria

The calibration is accepted only if all of the following are true:

1. OOF Macro-F1@5 improves over frozen E3.
2. OOF NDCG@10 is not worse than E3 by more than the configured tolerance
   (default absolute tolerance: 0.005).
3. Top100 TP is unchanged.
4. Zero-recall query count is unchanged.
5. The Top100 candidate multiset for every query is unchanged.
6. API calls = 0.
7. LLM calls = 0.
8. Validation inference never reads gold.

If any criterion fails:

```text
reject_blended_calibration_keep_frozen_e3
```

No further ranking tuning should be done.

## Outputs

```text
outputs/week2_day13_protected_blended_calibration/
├── calibration_summary.json
├── protection_audit.json
├── optimization_decision.json
├── fold_results.json
├── training_grid.json
├── oof_blended_predictions.jsonl
└── day13_5_report.md
```

## Run

```bat
python scripts\run_day13_protected_blended_calibration.py
pytest -q tests\test_day13_protected_blended_calibration.py
pytest -q
```

## Boundary

This experiment does not modify the API, production ranking, Day8 artifacts,
retrieval candidate pool, or online system.
