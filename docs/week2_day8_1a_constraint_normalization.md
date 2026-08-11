# Week 2 Day 8-1A — Constraint Normalization & Deduplication

## Scope

Day8-1A only establishes a deterministic canonical-constraint layer.

It does **not**:
- modify the existing semantic reranker;
- change ranking scores;
- call a remote embedding service;
- call an LLM;
- generate recommendation reasons;
- generate citation paths.

## Rules

1. No raw-string `contains` matching.
2. Unicode NFKC normalization.
3. Case folding.
4. Hyphens/dashes become token boundaries.
5. Punctuation and repeated whitespace are normalized.
6. Only conservative singular normalization is used.
7. Established aliases such as LLM ↔ large language model are canonicalized.
8. Similarity-based merging requires the same `constraint_type`.
9. Partial concepts such as `multimodal` and `multimodal large language model` remain separate.
10. Merged weights use `max(weight)`, never `sum(weight)`.
11. No LLM / no network call.

## Manual check

```cmd
python scripts\inspect_day8_constraints.py "recent papers on multimodal large language models for scientific document understanding"
```

Inspect:
- raw constraints;
- canonical constraints;
- source_constraint_ids;
- aliases;
- weight.

Then:

```cmd
pytest -q
```

Day8-1B should only start after this layer passes regression tests.
