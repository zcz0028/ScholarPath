# ScholarPath Week 2 Day 4

## 1. 测试

```cmd
pytest -q
```

预期新增 10 个测试；在 Day 3 的 120 个测试基础上应显示：

```text
130 passed
```

## 2. 先进行 Dry Run

```cmd
python scripts\run_day4_rescue_retrieval.py --dry-run
```

预期：

```text
Target queries: 24
Planned API calls: 48
API calls executed: 0
```

## 3. 正式运行

确保当前 CMD 已设置 `OPENALEX_API_KEY`，然后执行：

```cmd
python scripts\run_day4_rescue_retrieval.py --continue-on-error
```

## 4. 查看结果

```cmd
code outputs\week2_day4_rescue\run_summary.json
code outputs\week2_day4_rescue\day4_rescue_review.md
code outputs\week2_day4_rescue\recovered_queries.jsonl
```
