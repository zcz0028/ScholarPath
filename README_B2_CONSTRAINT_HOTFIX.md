# B2约束拆解增强热修复

本补丁修复 B2 在比较型查询上的约束拆解问题，例如：

```text
Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
```

增强后会显式识别：

```text
large language model
pre training / pretraining
smaller dataset
bigger dataset
better models
smaller dataset better than bigger dataset
```

覆盖文件：

```text
src/scholarpath/query/constraints.py
tests/test_constraint_decomposer.py
```

覆盖后运行：

```powershell
pytest -q
python scripts/run_b2_openalex.py --limit-queries 1 --output-dir outputs/b2_openalex_smoke_v2
Get-Content outputs\b2_openalex_smoke_v2\query_decompositions.jsonl
```
