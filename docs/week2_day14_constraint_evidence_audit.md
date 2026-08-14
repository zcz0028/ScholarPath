# Day14-1B — Constraint Quality + Evidence Sanity Audit

## Inputs

- `data/processed/realscholarquery_gold.jsonl`
- `outputs/week2_day8_evidence_rerank/predictions_e3_selected.jsonl`

## Outputs

`outputs/week2_day14_constraint_evidence_audit/`

- `audit_summary.json`
- `optimization_decision.json`
- `per_query_audit.jsonl`
- `paper_evidence_audit.csv`
- `constraint_quality_audit.csv`
- `day14_1b_report.md`

## Decision rules

`repair_constraint_decomposition_first` when suspicious constraints exceed 20% or more than 30% of queries contain at least one suspicious constraint.

`proceed_to_comprehensive_score_ablation` only when decomposition passes and TP-vs-FP evidence separation simultaneously satisfies: weighted-coverage delta >= 0.05, weighted-coverage ratio >= 1.20, and FP generic-entity-only rate exceeds TP by >= 0.05.

Otherwise: `repair_constraint_evidence_first`.

Boundary: API calls 0, LLM calls 0, gold only for strict offline audit, ranking/candidate/frontend unchanged.
