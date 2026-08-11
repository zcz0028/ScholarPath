# ScholarPath Day8-1C Phrase Match Performance Hotfix

Replace:

```text
src/scholarpath/rerank/constraint_evidence.py
```

Add:

```text
tests/test_day8_phrase_match_hotfix.py
docs/week2_day8_1c_phrase_match_hotfix.md
```

Then run:

```cmd
pytest -q
```

After 0 failures, rerun:

```cmd
python scripts\run_day8_evidence_rerank.py ^
  --predictions outputs\week2_day4_rescue\predictions_top100.jsonl ^
  --gold data\processed\realscholarquery_gold.jsonl ^
  --output-dir outputs\week2_day8_evidence_rerank
```
