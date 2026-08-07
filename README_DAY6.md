# ScholarPath Week 2 Day 6 Competition API Patch

This patch adds the competition-oriented FastAPI application layer.

## Merge

Copy these folders/files into the ScholarPath project root and merge folders:

```text
apps/
tests/
docs/
README_DAY6.md
requirements-day6.txt
```

## Install

```cmd
python -m pip install -r requirements-day6.txt
```

## Test

```cmd
pytest -q
```

The uploaded project currently has 141 tests. This patch adds 10 API tests, so the expected local total is approximately:

```text
151 passed
```

(The exact count is allowed to differ if your local branch has extra tests; all tests should pass.)

## Run

```cmd
python -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Competition design

- Benchmark mode serves evaluated Day-4 Top-K without consuming network budget.
- Live mode plans academic queries and retrieves OpenAlex results for unseen queries.
- Citation paths from Day 5 are exposed as auditable explanation/discovery traces.
- Gold answers are not exposed to the API layer.
