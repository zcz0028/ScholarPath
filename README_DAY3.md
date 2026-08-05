# ScholarPath Week 2 Day 3 Patch

本补丁新增 Anchor-aware Query Planner，不覆盖原项目已有文件。

## 合并方式

把补丁中的 `src`、`scripts`、`tests`、`docs` 和 `README_DAY3.md` 合并到 ScholarPath 项目根目录。

合并后运行：

```cmd
pytest -q
python scripts\build_anchor_query_plans.py
```

预期结果：

```text
120 passed
Query count: 24
Planned query count: 48
Queries with two plans: 24
Queries with zero plans: 0
Estimated Day-4 API calls: 48
API calls executed: 0
```

详细说明见：

```text
docs/week2_day3_query_planner.md
```
