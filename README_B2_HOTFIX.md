# ScholarPath 第7步B2热修复补丁

修复两个测试失败：

1. `textual_constraint_coverage` 现在允许简单复数匹配，例如 `law` 可以匹配标题里的 `laws`。
2. `keyword_fallback` 会提前加入子查询列表，避免和 `constraint_core` 文本重复时被去重逻辑删除。

覆盖文件：

```text
src/scholarpath/query/constraints.py
src/scholarpath/retrieval/constraint_aggregation.py
```

覆盖后重新运行：

```powershell
pytest -q
```
