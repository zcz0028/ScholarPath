# ScholarPath Day8-1C Patch

Apply after Day8-1B.

## Added

```text
src/scholarpath/rerank/evidence_aware_rerank.py
scripts/run_day8_evidence_rerank.py
tests/test_day8_evidence_aware_rerank.py
docs/week2_day8_1c_evidence_rerank.md
```

## Not modified

```text
src/scholarpath/rerank/semantic_rerank.py
src/scholarpath/rerank/constraint_evidence.py
apps/api
apps/web
```

## First verification

```cmd
pytest -q
```

Then run the offline benchmark command documented in `docs/week2_day8_1c_evidence_rerank.md`.

Important: use a frozen Top100 candidate/prediction file containing B3 scores. Do not point the script at Live API output.
