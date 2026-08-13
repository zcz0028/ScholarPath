# Week2 Day12 — Failure Recovery and Robustness Analysis

## Goal

Day12 converts ScholarPath's existing evaluation artifacts into a
failure-aware system analysis.

The objective is not to improve benchmark numbers by introducing new
retrieval or ranking logic. Instead, Day12 answers four questions:

1. Where does the retrieval pipeline fail?
2. Which zero-recall queries were recovered by the Day4 rescue stage?
3. Which failures remain unresolved?
4. What robustness limitations must be disclosed alongside the current
   benchmark results?

## Existing failure taxonomy

Day12 does not introduce a new failure classifier.

It reuses:

`scholarpath.evaluation.pipeline_diagnostics.classify_failure`

The existing taxonomy includes:

- `success`
- `possible_matching_issue`
- `ranking_loss_top20`
- `ranking_loss_top50`
- `ranking_loss_top100`
- `fusion_pool_miss`
- `guard_drop`
- `selector_drop`
- `guard_aware_drop`
- `retrieval_miss`

The existing priority policy is also preserved.

## Recovery accounting

Day4 already records recovered and unrecovered zero-recall queries.

A recovered query is a query whose baseline TP was zero and whose
Day4 current TP is greater than zero.

An unrecovered query remains at zero TP after rescue.

Day12 aggregates these frozen artifacts into:

- baseline zero-recall query count;
- recovered query count;
- unrecovered query count;
- query recovery rate;
- recovered TP gain;
- remaining false negatives.

## Failure families

For presentation only, Day12 groups the existing low-level taxonomy into
higher-level families:

- retrieval;
- fusion;
- ranking;
- matching audit;
- post-ranking filtering;
- success.

The original `failure_type` remains the source of truth.

## Explainability robustness

The current evidence layer must be interpreted together with its data
limitations.

The Day11 frozen metrics report:

- constraint evidence match rate;
- paper evidence support rate;
- reason generation rate;
- evidence-backed reason rate;
- abstract availability rate.

Reason generation and evidence-backed explanation are intentionally
reported separately.

A generated reason is not automatically treated as an evidence-backed
reason.

## Citation robustness

Day9 citation-path coverage is currently a single-query case study.

It demonstrates citation-path feasibility but must not be reported as
full 50-query benchmark coverage.

## Ranking boundary

The frozen Day8 E3 configuration uses:

- alpha = 1.0
- beta = 0.0

Therefore the Day8 NDCG improvement must not be attributed to the
evidence or citation components.

Day12 does not modify the ranking.

## Reproducibility

Run:

```bash
python scripts/build_day12_failure_analysis.py