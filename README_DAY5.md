# ScholarPath Week 2 Day 5 Patch

This patch adds the first controlled-citation-expansion stage:

- high-confidence seed gating;
- one-hop OpenAlex references and cited-by retrieval;
- global API budget and cache accounting;
- citation path logging;
- production gold-field rejection;
- unit tests and Day-5 manual review template.

## Merge

Copy `src`, `scripts`, `tests`, and `docs` into the project root and merge folders.

## Test

```cmd
pytest -q
```

## Dry run

```cmd
python scripts\run_day5_citation_expansion.py --dry-run
```

The dry run must report 11 target queries, at most 33 seeds, at most 60 estimated calls, one hop, and no gold use.

## Formal retrieval

```cmd
python scripts\run_day5_citation_expansion.py --continue-on-error
```

This first patch deliberately stops after producing auditable citation candidates. After seed quality is checked, the next Day-5 step merges them into the Day-4 pool, reranks them, applies Guard-aware Selector, and runs the ablation table.
