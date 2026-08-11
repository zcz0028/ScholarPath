# ScholarPath Day8-1A V2 Patch

This is a minimal corrective patch on top of Day8-1A V1.

## Files replaced

```text
src/scholarpath/rerank/constraint_evidence.py
tests/test_day8_constraint_evidence.py
scripts/inspect_day8_constraints.py
```

## New document

```text
docs/week2_day8_1a_v2_constraint_recovery.md
```

It does NOT overwrite:

```text
src/scholarpath/query/constraints.py
src/scholarpath/rerank/semantic_rerank.py
apps/api
apps/web
```

## Local verification performed while building this patch

Targeted Day8 tests:

```text
16 passed
```

Full regression on the available project snapshot:

```text
157 passed
```

That snapshot has fewer tests than the user's current project. Since the user's
current V1 run was `161 passed`, replacing the old 10 Day8-1A tests with the
new 16-test suite should normally make the local total higher, not exactly 157.

## Expected manual outputs

Query A final canonical count: `3`

```text
large language model
multimodal
scientific document understanding
```

Query B final canonical count: `2`

```text
graph neural network
molecular property prediction
```
