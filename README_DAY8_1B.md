# ScholarPath Day8-1B Constraint Evidence Matcher Patch

Apply this patch **after Day8-1A V2**.

## Replaced

```text
src/scholarpath/rerank/constraint_evidence.py
```

This replacement contains the complete Day8-1A V2 implementation plus Day8-1B.

## Added

```text
tests/test_day8_constraint_evidence_matcher.py
scripts/inspect_day8_evidence.py
docs/week2_day8_1b_constraint_evidence_matcher.md
```

## Not modified

```text
src/scholarpath/query/constraints.py
src/scholarpath/rerank/semantic_rerank.py
apps/api
apps/web
```

## Build verification

Targeted Day8-1B tests:

```text
24 passed
```

Full regression on the available ScholarPath snapshot, with the current uploaded
`constraints.py`, `semantic_rerank.py`, Day8-1A V2 tests, and Day8-1B tests:

```text
181 passed
```

Your local project already contains additional tests, so do not expect the same
total. The acceptance condition is `0 failed`.

## Commands

```cmd
pytest -q
python scripts\inspect_day8_evidence.py
```
