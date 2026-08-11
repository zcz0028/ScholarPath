# ScholarPath Day8-2 Real API Integration Patch

Apply this patch after the accepted Day8-1D checkpoint.

## Replaced

```text
apps/api/main.py
apps/api/schemas.py
apps/api/services/search_service.py
tests/test_day6_api.py
```

## Added

```text
docs/week2_day8_2_real_api_integration.md
```

## Not modified

```text
src/scholarpath/rerank/constraint_evidence.py
src/scholarpath/rerank/evidence_aware_rerank.py
src/scholarpath/rerank/recommendation_reason.py
src/scholarpath/rerank/semantic_rerank.py
frontend files
```

## Production ranking

```text
E3 / alpha=1.0 / beta=0.0
```

## Commands

```cmd
pytest -q
```

After tests pass, start the FastAPI server using the same command you used
before Day8-2, then inspect `/api/search` responses in Benchmark and Live mode.

Do not commit the patch until pytest and both API modes are verified locally.
