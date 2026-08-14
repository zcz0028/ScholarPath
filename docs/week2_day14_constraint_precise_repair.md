# Day14-1C-2 — Precise Constraint Repair

This follow-up avoids broad "short topic = bad" filtering.

It introduces:
- precise discourse/prompt residue stripping;
- high-precision hard-fragment rejection;
- conservative duplicate/containment suppression before max_constraints;
- a two-tier audit: hard fragments are PASS/FAIL, warnings are review-only;
- no changes to constraint_evidence.py or ranking weights.

Run:
1. python scripts\\apply_day14_1c2_patch.py
2. pytest -q tests\\test_day14_constraint_decomposition_repair.py
3. pytest -q tests\\test_day8_constraint_evidence.py
4. pytest -q
5. python scripts\\build_day14_constraint_repair_regression.py
6. type outputs\\week2_day14_constraint_repair_regression_v2\\optimization_decision.json
7. type outputs\\week2_day14_constraint_repair_regression_v2\\repair_v2_summary.json

Acceptance:
- hard fragment rate <= 10%
- hard-fragment query rate <= 20%
- TP weighted coverage > FP weighted coverage
- TP/FP weighted coverage ratio >= 1.20
- FP generic-entity-only rate > TP generic-entity-only rate
