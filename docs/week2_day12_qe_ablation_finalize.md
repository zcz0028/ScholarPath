# Week2 Day12-3-3 — Finalize QE-R9 Ablation

Day12-3-3 finalizes the residual-only Query Expansion experiment entirely
offline from already-generated QE retrieval predictions.

It does **not** rerun OpenAlex and does **not** modify the frozen Day8 E3
artifact.

Run:

```bat
python scripts\finalize_day12_qe_ablation.py
```

The finalizer reads frozen Day4 metrics plus QE-R9 Top20/50/100 predictions,
recomputes strict evaluation using:

```python
EvaluationConfig(
    mode="strict",
    recall_at=(20, 50, 100),
    deduplicate=True,
)
```

and writes:

```text
outputs/week2_day12_query_expansion_retrieval/
├── qe_vs_frozen_comparison.json
└── qe_ablation_report.md
```

Adoption requires all of the predeclared checks to pass:

1. Top100 TP gain > 0.
2. Macro Recall@100 gain > 0.
3. Macro F1@100 gain > 0.
4. Top20 Macro Precision decrease no worse than 0.002 absolute.

The manually verified Top100 QE-R9 result matched frozen Day4 exactly:

```text
TP@100 = 105
Macro Recall@100 = 0.16751382980510768
Macro F1 = 0.03420862536132911
```

Therefore the expected decision is:

```text
reject_qe_r9_keep_frozen_pipeline
```

The finalizer recomputes this from artifacts instead of hard-coding the result.
