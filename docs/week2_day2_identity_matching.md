# ScholarPath Week 2 Day 2：论文身份解析、匹配与去重

## 目标

Day 2 不修改召回算法，而是先保证评测与候选融合中的“同一篇论文”能够稳定识别。重点处理：

- DOI 字符串与 DOI URL；
- `10.48550/arXiv.*` 与 arXiv ID 的映射；
- arXiv `abs`、`pdf`、`export.arxiv.org` 和 ar5iv URL；
- OpenAlex URL 与 `W...` ID；
- arXiv 版本号；
- 预印本与正式出版版本；
- 标题、作者和年份辅助去重；
- 近似标题仅进入审计，不直接计为 TP。

## 双模式设计

- `strict`：完全保留第一周行为，保证历史实验可复现；
- `strict_v2`：启用身份推断和版本感知去重，用于 Day 2 对照实验。

在人工确认新增匹配与新增去重均可靠之前，不覆盖第一周正式基线。

## 运行

```cmd
pytest -q
python scripts\audit_paper_identity.py
```

也可以评测 `strict_v2`：

```cmd
python scripts\run_evaluation.py --gold data\processed\realscholarquery_gold.jsonl --predictions outputs\b4_fusion_semantic\predictions_top100.jsonl --output-dir outputs\week2_day2_strict_v2_eval --mode strict_v2 --strict-qids
```

## 输出

`outputs/week2_day2_identity_audit/`：

- `identity_audit_summary.json`
- `per_query_identity_audit.jsonl`
- `new_strict_v2_matches.jsonl`
- `possible_version_matches.jsonl`
- `dedup_changes.jsonl`
- `identifier_coverage.csv`
- `day2_identity_review.md`

## 验收

- 原有 `strict` 结果保持不变；
- 所有测试通过；
- API 调用数为 0；
- 新增 TP 必须逐条人工确认；
- 新增去重必须确认没有误删不同论文；
- 近似标题候选不得直接计入正式 TP。
