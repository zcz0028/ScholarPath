# Week2 Day12-3-2 — QE-R9 Real Retrieval Ablation

## Goal

QE-R9 tests whether lightweight query expansion produces a measurable
retrieval-quality gain on the residual retrieval failures discovered by
Day12.

The experiment is deliberately narrow:

- target only residual `retrieval_miss` queries;
- reuse the frozen Day3 query-plan schema;
- add one expanded OpenAlex retrieval plan per target query;
- start from the frozen Day4 merged candidate pool and predictions;
- reuse the existing Day4 fusion, semantic reranking, evaluation, cache,
  error handling, and cost accounting;
- do not overwrite Day4, Day8, Day9, Day10, or Day11 artifacts.

## Why residual-only

Day12 diagnosis showed that the main unresolved bottleneck is retrieval
coverage. Query expansion is therefore evaluated only where retrieval is
still failing.

This avoids spending additional API calls on queries that already have
successful retrieval results and reduces the risk of introducing noise into
successful cases.

## Baseline and experiment

Baseline:

```text
Frozen Day4 Rescue
```

Experiment:

```text
Frozen Day4 merged candidate pool
+ one QE OpenAlex plan on residual retrieval-miss queries
→ same Day4 fusion
→ same Day4 semantic reranking
→ strict Top20/50/100 evaluation
```

This is a retrieval-stage ablation. It does not modify the frozen Day8 E3
artifact.

If QE-R9 passes the adoption checks, a separate E3 integration experiment
should be performed before production adoption.

## Metrics

Primary quality metrics:

- TP@20 / TP@50 / TP@100
- Macro Precision
- Macro Recall
- Macro F1
- Micro F1

Efficiency metrics:

- additional API calls
- cache hits
- retries
- estimated cost
- latency
- API calls per added Top100 TP

## Adoption checks

QE-R9 is marked as a candidate for E3 integration only when all of the
following hold:

1. Top100 TP increases.
2. Macro Recall@100 increases.
3. Macro F1@100 increases.
4. Macro Precision@20 does not decrease by more than 0.002 absolute.

The checks are fixed before seeing the live experiment result.

## Usage

First perform a dry run:

```bat
python scripts\run_day12_query_expansion_retrieval_ablation.py --dry-run
```

Then run the actual experiment:

```bat
python scripts\run_day12_query_expansion_retrieval_ablation.py --continue-on-error
```

Use `--refresh-cache` only when an intentional fresh OpenAlex run is required.

## Reporting boundary

A positive QE-R9 result demonstrates retrieval-stage value on residual
failures. It must not be presented as a Day8 E3 improvement until the
expanded candidate pool is separately validated through the frozen E3
ranking stack.
