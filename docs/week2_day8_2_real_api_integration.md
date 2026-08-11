# Day8-2 — Real API Integration

## Scope

This patch integrates the accepted Day8 stack into the existing FastAPI search API.

Production path:

```text
Benchmark / Live candidates
    -> Day8 E3 pure-semantic rerank
    -> canonical constraint evidence
    -> deterministic recommendation reason
    -> PaperResult
```

The accepted Day8-1C production configuration is frozen:

```text
variant = E3
alpha = 1.0
beta = 0.0
```

Constraint evidence does not receive a linear ranking weight in production.

## Benchmark behavior

Benchmark mode always loads the frozen Day4 Top100 candidate pool, applies the
same E3 ranking used by the accepted offline experiment, and only then slices to
the requested Top20 / Top50 / Top100.

Benchmark mode still performs zero OpenAlex API calls.

Day5 citation paths remain explanation/discovery metadata only and do not change
the Day8 E3 ranking.

## Live behavior

Live mode keeps the existing:

```text
Query Planner -> OpenAlex retrieval -> deduplication
```

and replaces the old `semantic_first` production ranking with:

```text
Day8 E3 -> Constraint Evidence -> Recommendation Reason
```

No LLM is used for reason generation.

## PaperResult additions

The API now exposes:

```text
reason_tags
reason_text
constraint_evidence
matched_constraints
unmatched_constraints
matched_count
constraint_count
```

`score` is the Day8 `day8_final_score` when Day8 annotations exist.

Citation paths remain `null` when no real citation path exists.

## Evidence precedence

When Day8 evidence exists it becomes the API truth layer. Legacy B5/Guard
evidence is used only as fallback for records without Day8 annotations. Legacy
missing-constraint rows are never mixed into Day8 evidence.

## System feature flags

`GET /api/system` now reports:

```text
day8_e3_reranking
constraint_evidence
deterministic_recommendation_reason
```

## Verification

Run:

```cmd
pytest -q
```

Then start the backend with your existing FastAPI command and test both modes.

For Benchmark, select a real qid and verify:

```text
pipeline contains day8_e3_rerank
score is populated
constraint_evidence is populated or explicitly unmatched
reason_text is deterministic
matched_count / constraint_count are present
```

For Live, use:

```text
recent papers on multimodal large language models for scientific document understanding
```

and verify that the response contains the same Day8 fields.

## Build-side verification

The patch was validated against the uploaded current API files plus the accepted
Day8-1A/1B/1C/1D modules.

Targeted combined regression:

```text
87 passed
0 failed
```

This local reconstruction does not contain every test in the user's current
working tree. On the user's project, the acceptance condition remains `0 failed`.
