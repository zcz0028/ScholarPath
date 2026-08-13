# Week2 Day13-1 — Global-K + Oracle-K F1 Diagnostic

This stage studies the Precision/Recall/F1 tradeoff on the frozen Day8 E3 ranking.

Boundary:
- API calls = 0
- LLM calls = 0
- ranking modified = false
- Day8 E3 artifact modified = false
- gold used only for offline evaluation/oracle analysis
- no production parameter changed

Global-K evaluates fixed K values:
1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,80,100

Oracle-K tries K=1..100 per query and chooses the query-level best F1.
Oracle-K is analysis-only and must not be used as production cutoff or reported as competition score.

Outputs:
outputs/week2_day13_f1_diagnostic/
- global_k_curve.csv
- global_k_summary.json
- oracle_k_per_query.jsonl
- oracle_k_summary.json
- day13_f1_diagnostic_report.md

Run:
python scripts\build_day13_f1_diagnostic.py
pytest -q tests\test_day13_f1_diagnostic.py
pytest -q
