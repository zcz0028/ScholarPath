# ScholarPath Day8-1A Patch

新增：

```text
src/scholarpath/rerank/constraint_evidence.py
tests/test_day8_constraint_evidence.py
scripts/inspect_day8_constraints.py
docs/week2_day8_1a_constraint_normalization.md
```

本补丁不覆盖 `constraints.py` 和 `semantic_rerank.py`。

## 使用

在项目根目录解压覆盖，然后：

```cmd
pytest -q
```

单独查看复杂 Query：

```cmd
python scripts\inspect_day8_constraints.py "recent papers on multimodal large language models for scientific document understanding"
```

确认 Day8-1A 后，再进入 Day8-1B Evidence Matcher。
