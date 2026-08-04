# B2约束拆解增强热修复 v2

修复内容：

1. 将 `pre training / pretraining` 这类训练阶段约束提前到模型别名前面；
2. 当已经识别到 `large language model` 时，去掉冗余的 `language model`；
3. 避免 `max_constraints=6` 时 `pre training` 被挤掉。

覆盖文件：

```text
src/scholarpath/query/constraints.py
tests/test_constraint_decomposer.py
```

覆盖后运行：

```powershell
pytest -q
python scripts/run_b2_openalex.py --limit-queries 1 --output-dir outputs/b2_openalex_smoke_v3
Get-Content outputs\b2_openalex_smoke_v3\query_decompositions.jsonl
```
