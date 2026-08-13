# Day13-2 — Gold-free Adaptive Cutoff + Query-level CV

## Goal
Test whether query-specific output depth can beat the Day13-1 best global fixed K without changing Day8 E3 ranking.

## Leakage boundary
Gold is permitted only inside a training fold to construct per-query Oracle-K targets and to score held-out OOF predictions. Held-out Oracle-K is never supplied to the model. Inference features are derived only from the natural-language query and the frozen Day8 E3 ranked candidates/evidence.

## Small-sample design
The benchmark contains only 50 queries, so Day13-2 deliberately avoids a high-capacity classifier. It uses deterministic robust scaling plus distance-weighted k-nearest-neighbor regression over query/ranking confidence features, then snaps the estimate to the approved K grid. The final decision is based on query-level OOF Macro-F1, not in-sample fit.

## Decision rule
If OOF adaptive Macro-F1 exceeds Day13-1 best fixed-K Macro-F1, retain adaptive cutoff as a candidate production policy. Otherwise retain the fixed-K baseline and treat the adaptive experiment as a negative ablation result.

Internal strict F1 remains an iteration metric because the competition has not published its complete F1 implementation.
