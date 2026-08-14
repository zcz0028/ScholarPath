# Day14-1C — Constraint Decomposition Repair + Regression Audit

## Goal

Repair the specific constraint-fragmentation failure discovered by Day14-1B
without changing ranking, retrieval, the candidate pool, or the frontend.

The repair is deterministic and gold-free at inference.

## Changes

### `src/scholarpath/query/constraints.py`

Adds:

- broader prompt/boilerplate stripping;
- protected short academic terms (`RLHF`, `QAT`, `SFT`, `NER`, `RAG`, etc.);
- discourse-fragment cleanup;
- dangling-fragment rejection;
- deterministic extraction of complete capability / relation / output spans;
- quality filtering before `max_constraints` truncation.

### `src/scholarpath/rerank/constraint_evidence.py`

Only residual-recovery behavior is changed:

- prompt/discourse tokens become residual boundaries;
- incomplete residual spans are rejected;
- protected short academic terms remain allowed.

Normal title / abstract / concept evidence matching is unchanged.

## Safety boundary

- API calls = 0
- LLM calls = 0
- production ranking modified = false
- candidate pool modified = false
- Day8 E3 weights modified = false
- frontend modified = false
- gold is used only in the offline regression audit

## Run

```bat
pytest -q tests\test_day14_constraint_decomposition_repair.py

python scripts\build_day14_constraint_repair_regression.py

type outputs\week2_day14_constraint_repair_regression\optimization_decision.json
type outputs\week2_day14_constraint_repair_regression\repair_comparison.json

pytest -q
```

## Acceptance

The repair passes only if all hold:

1. suspicious constraint rate <= 20%
2. suspicious query rate <= 30%
3. TP mean weighted coverage > FP mean weighted coverage
4. TP/FP weighted coverage ratio >= 1.20
5. FP generic-entity-only rate > TP generic-entity-only rate

A pass does **not** modify production ranking. It only authorizes the next
protected comprehensive-score ablation.
