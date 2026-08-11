# Week 2 Day8-1A V2 — Boundary Cleanup + Residual Constraint Recovery

## Why V2 exists

Day8-1A V1 passed regression tests, but manual inspection exposed two issues:

1. `graph neural networks for` remained as a duplicate topic constraint.
2. `scientific document understanding` could be omitted from parsed constraints, producing an artificially high constraint coverage later.

V2 fixes only these two canonical-constraint issues. It does not change ranking.

## Boundary connector cleanup

Constraint phrases are normalized before comparison.

Examples:

- `Graph-Neural Networks` → `graph neural network`
- `graph neural networks for` → `graph neural network`
- `for molecular property prediction` → `molecular property prediction`

Only phrase boundaries are cleaned. Internal connectors are preserved:

- `bag of words` stays `bag of words`
- `in context learning` stays `in context learning`

Leading cleanup is deliberately conservative so meaningful method names are not damaged.

## Residual constraint recovery

After planner constraints are normalized and deduplicated:

1. normalize the original query;
2. mark token spans already covered by canonical constraints;
3. split remaining content on prompt/noise/connective tokens;
4. keep only meaningful contiguous residual spans;
5. add them as `topic` constraints with weight `1.0`;
6. run the whole set through the normal deduplicator again.

No hard-coded `scientific document understanding` pattern is used.

## Fixed acceptance queries

### Query A

`recent papers on multimodal large language models for scientific document understanding`

Expected final canonical constraints:

1. `large language model`
2. `multimodal`
3. `scientific document understanding`

Expected residual span:

`scientific document understanding`

### Query B

`graph neural networks for molecular property prediction`

Expected final canonical constraints:

1. `graph neural network`
2. `molecular property prediction`

Expected residual spans:

none.

## Safety rules retained

- no raw-string contains matching;
- Unicode NFKC + casefold + hyphen normalization;
- conservative singular normalization;
- deterministic alias map;
- same-type Jaccard merging for non-declared equivalents;
- merged weights use max, never sum;
- no LLM;
- no embedding API;
- no network call;
- no semantic-rerank modification.

## Commands

```cmd
pytest -q
```

Then:

```cmd
python scripts\inspect_day8_constraints.py "recent papers on multimodal large language models for scientific document understanding"
```

and:

```cmd
python scripts\inspect_day8_constraints.py "graph neural networks for molecular property prediction"
```

Only after these outputs are correct should Day8-1B Evidence Matcher begin.
