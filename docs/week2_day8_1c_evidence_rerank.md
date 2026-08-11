# Week 2 Day8-1C — Evidence-aware Rerank + NDCG@10

## Scope

Day8-1C is an offline ranking experiment. It does not modify `semantic_rerank.py`, the API, or the frontend.

### E0
Current B3 order. No Day8 score changes ranking.

### E1
`alpha * normalized_b3_final + beta * raw_constraint_coverage`

Raw planner constraints are intentionally not deduplicated. They are passed through the same Day8 evidence matcher so E1 vs E2 isolates canonicalization/deduplication.

### E2
`alpha * normalized_b3_final + beta * canonical_constraint_coverage`

### E3
`alpha * pure_semantic_base + beta * canonical_constraint_coverage`

The E3 base is isolated in `evidence_aware_rerank.py` and does not modify B3 business code.

`pure_semantic_base = 0.38 query_overlap + 0.30 expanded_query_overlap + 0.24 title_overlap + 0.08 abstract_signal`

It excludes:

- old B2 constraint score;
- old phrase/core constraint coverage;
- original-rank prior;
- title boost;
- core-constraint bonus;
- strong-identifier bonus;
- low-evidence penalty.

## Per-paper coverage

Coverage is calculated independently for every paper from that paper's `list[ConstraintEvidence]`.

Only `matched=True` evidence contributes. Weights always come from the corresponding constraint definition, never from evidence objects.

`coverage = matched_weight / total_weight`

Safety:

- no constraints -> `0.0`;
- total weight <= 0 -> `0.0`;
- duplicate evidence IDs cannot double count;
- unknown evidence IDs are ignored.

## Score scale

E1/E2 min-max normalize B3 scores inside each query candidate pool. If all B3 scores are equal, each normalized score is `0.5`.

E3 is already in `[0, 1]` because it is a convex combination of bounded overlap signals.

## Alpha grid

`alpha = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]`, `beta = 1 - alpha`.

## Metrics

Primary: binary `NDCG@10`.

Protection metrics:

- Top20 TP must not decrease vs E0;
- Top100 TP must not decrease vs E0;
- zero-recall query count must not increase vs E0.

Only protected alpha candidates participate in final selection. Within `0.005` NDCG, larger alpha (smaller beta) wins.

## Cross validation

Query-level deterministic 5-fold validation is used. Papers from the same query are never split across folds.

## Gold safety

Gold is used only by the offline evaluation script. Reranking and evidence construction never consume gold.

The included evaluator uses deterministic binary identity matching: canonical ID, DOI, OpenAlex ID, Semantic Scholar ID, arXiv ID, or exact normalized title. If the project's established `strict_v2` evaluator differs, keep `strict_v2` as the authoritative TP/FP/FN report and use this script's NDCG as the Day8 ranking ablation metric.

## Run

Use the same frozen B3/B5 candidate file you want to rerank, ideally Top100:

```cmd
python scripts\run_day8_evidence_rerank.py ^
  --predictions outputs\YOUR_FROZEN_BASELINE\predictions_top100.jsonl ^
  --gold data\processed\realscholarquery_gold.jsonl ^
  --output-dir outputs\week2_day8_evidence_rerank
```

Outputs:

```text
outputs/week2_day8_evidence_rerank/
  ablation_summary.json
  predictions_e0_baseline.jsonl
  predictions_e1_selected.jsonl
  predictions_e2_selected.jsonl
  predictions_e3_selected.jsonl
```

Do not connect the selected variant to Live/API until the offline ablation has been reviewed.
