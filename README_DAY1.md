# ScholarPath Week 2 Day 1 诊断补丁

将本目录中的文件按相同相对路径复制到 ScholarPath 项目根目录。

## 运行顺序

```bash
pytest -q
python scripts/analyze_pipeline_failures.py
```

在当前第一周项目包上验证结果：

```text
99 passed
Target stage: b4_top100
Target total TP: 61
Target zero-recall queries: 26
ranking_loss_top100: 2
retrieval_miss: 24
success: 24
API calls: 0
```

## 输出文件

运行后生成：

- `outputs/week2_day1_diagnostics/diagnostic_summary.json`
- `outputs/week2_day1_diagnostics/query_stage_diagnostics.jsonl`
- `outputs/week2_day1_diagnostics/zero_recall_taxonomy.jsonl`
- `outputs/week2_day1_diagnostics/stage_recall_matrix.csv`
- `outputs/week2_day1_diagnostics/stage_recall_matrix.md`
- `outputs/week2_day1_diagnostics/anchor_inventory.jsonl`
- `outputs/week2_day1_diagnostics/priority_queries.md`
- `outputs/week2_day1_diagnostics/possible_matching_issues.jsonl`
