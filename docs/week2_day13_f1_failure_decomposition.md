# Week2 Day13-3 — F1 Failure Decomposition & Optimization Decision

## Purpose

Day13-3 explains where the remaining internal F1 headroom comes from after:

- Day13-1 showed Best Fixed K=5 beats Top20 while Oracle-K remains much higher.
- Day13-2 showed the gold-free OOF adaptive cutoff does not beat the simple
  fixed K policy.

This stage does not add another ranking or retrieval model. It decomposes each
query into an interpretable failure family and makes a Go / No-Go decision for
further algorithm work.

## Failure taxonomy

The taxonomy is mutually exclusive:

- `retrieval_ceiling`: no strict gold hit anywhere in frozen E3 Top100.
- `deep_ranked_hit`: first strict hit is after Top20.
- `mid_ranked_hit`: first strict hit is after K=5 but within Top20.
- `cutoff_mismatch`: at least one hit is inside Top5, but Oracle-K materially
  improves query-level F1 with another result boundary.
- `already_good`: Fixed K=5 is within the configured tolerance of Oracle-K.

`retrieval_ceiling` is a retrieval problem. `deep_ranked_hit` and
`mid_ranked_hit` are primarily ranking-depth problems. `cutoff_mismatch` is a
selection-boundary problem. `already_good` should be frozen.

## Inputs

Required:

```text
outputs/week2_day13_f1_diagnostic/oracle_k_per_query.jsonl
```

Optional but strongly recommended:

```text
outputs/week2_day13_adaptive_cutoff/oof_cutoff_predictions.jsonl
outputs/week2_day13_adaptive_cutoff/oof_predictions.jsonl
data/processed/realscholarquery_gold.jsonl
```

The cutoff file enables cutoff-error analysis. The OOF predictions plus gold
are evaluated through ScholarPath's canonical strict evaluator so the
Day13-3 adaptive Macro-F1 exactly follows the Day13-2 deduplication contract.

## Boundaries

- API calls = 0
- LLM calls = 0
- ranking modified = false
- production configuration modified = false
- gold is used only for offline failure analysis
- failure labels are never inference features

## Outputs

```text
outputs/week2_day13_f1_failure_decomposition/
├── per_query_failure_decomposition.jsonl
├── failure_family_summary.json
├── failure_family_table.csv
├── priority_queries.jsonl
├── optimization_decision.json
└── day13_3_report.md
```

## Run

```bat
python scripts\build_day13_f1_failure_decomposition.py
pytest -q tests\test_day13_f1_failure_decomposition.py
pytest -q
```

The decision artifact is intentionally conservative. A high Oracle-K ceiling
does not justify further adaptive-selector tuning when Day13-2 OOF results do
not beat the fixed policy.
