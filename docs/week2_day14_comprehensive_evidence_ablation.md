# Day14-1D — Comprehensive Evidence Score Protected Ablation

## Purpose
Test whether repaired query constraints provide held-out ranking value beyond frozen Day8 E3, without changing candidate identity or production ranking.

## Score
`evidence = weighted_coverage × matched_confidence × field_quality × specificity_factor − generic_entity_only_penalty`

`final = (1-beta) × query_minmax(E3) + beta × evidence`

Beta is selected only on training folds. `beta=0` is the frozen-E3 control and must preserve input order exactly.

## Protection gates
- candidate identity unchanged
- Top100 TP unchanged
- zero-recall query count unchanged
- OOF Macro-F1@5 strictly improves
- OOF NDCG@10 does not decrease (default tolerance 0)
- positive F1 delta appears in at least two validation folds
- no gold at validation inference
- no API/LLM calls

A failed gate means `reject_comprehensive_evidence_keep_frozen_e3`.
