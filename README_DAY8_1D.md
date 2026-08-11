# ScholarPath Day8-1D Patch

This patch adds the deterministic Recommendation Reason Builder.

## Add

```text
src/scholarpath/rerank/recommendation_reason.py
tests/test_day8_recommendation_reason.py
scripts/inspect_day8_recommendation_reason.py
docs/week2_day8_1d_recommendation_reason.md
```

It does not overwrite Day8-1A/1B/1C business code, API code, or frontend code.

## Run

```cmd
pytest -q
python scripts\inspect_day8_recommendation_reason.py
```

Day8-1D remains offline in this patch. API/frontend integration is intentionally
deferred to the next integration step.
