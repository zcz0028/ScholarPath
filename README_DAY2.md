# ScholarPath Week 2 Day 2 Patch

将本目录中的 `src/`、`scripts/`、`tests/`、`docs/` 合并到 ScholarPath 项目根目录。

运行：

```cmd
pytest -q
python scripts\audit_paper_identity.py
```

预期在当前第一周基线上得到：

```text
110 passed
Legacy strict TP: 61
Strict v2 TP: 61
TP delta: +0
Legacy duplicates removed: 2
V2 duplicates removed: 2
Possible version matches: 2
API calls: 0
```

说明：

- `strict` 保持第一周评测行为不变；
- `strict_v2` 增加 DOI、URL、arXiv 与 OpenAlex 身份推断；
- 近似标题只进入审计，不直接计为 TP；
- Day 2 的目标是排除伪 FN 和误去重，而不是人为提高指标。
