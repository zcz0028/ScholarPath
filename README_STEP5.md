# ScholarPath 第1周第5步：B0结果归档与错误分析

## 1. 当前目的

第4步已经跑通B0原始查询直接检索基线。本步骤不引入新检索策略，只做实验归档与错误分析，回答：

1. B0哪一个Top-K最好；
2. 哪些查询完全没有召回；
3. 哪些查询召回了但排名靠后；
4. 哪些查询FP最多；
5. B1查询改写和B2多子查询应优先解决哪些问题。

## 2. 新增文件

```text
src/scholarpath/evaluation/error_analysis.py
scripts/analyze_b0_results.py
tests/test_b0_error_analysis.py
README_STEP5.md
```

## 3. 运行命令

```powershell
pytest -q
```

预期测试数至少大于等于40，且全部通过。

然后运行：

```powershell
python scripts/analyze_b0_results.py --b0-dir outputs/b0_openalex --output-dir outputs/b0_openalex_analysis --focus-topk 50 --mode strict
```

## 4. 输入文件

默认读取：

```text
outputs/b0_openalex/run_summary.json
outputs/b0_openalex/query_logs.jsonl
outputs/b0_openalex/evaluation/top20/strict/summary.json
outputs/b0_openalex/evaluation/top50/strict/summary.json
outputs/b0_openalex/evaluation/top100/strict/summary.json
outputs/b0_openalex/evaluation/top100/strict/per_query.jsonl
```

注意：使用`top100/strict/per_query.jsonl`是为了同时分析Recall@20、Recall@50和Recall@100的排名变化。

## 5. 输出文件

```text
outputs/b0_openalex_analysis/
├── analysis_summary.json
├── b0_result_table.csv
├── b0_result_table.md
├── query_diagnostics.jsonl
├── zero_recall_queries.jsonl
├── high_fp_queries.jsonl
├── late_recall_queries.jsonl
├── matched_rank_distribution.json
├── false_negatives_sample.jsonl
├── false_positives_sample.jsonl
└── b1_recommendations.md
```

## 6. 输出含义

### b0_result_table.md

整理Top20、Top50、Top100在strict和pasa_title下的P/R/F1、TP/FP/FN。

### zero_recall_queries.jsonl

列出Top100仍然没有召回任何标准答案的查询。这类查询是B1查询改写和B2多子查询的最高优先级。

### late_recall_queries.jsonl

列出Top20没有召回，但Top50或Top100才召回的查询。这类查询说明相关论文进入候选池但排名靠后，后续需要Reranker。

### high_fp_queries.jsonl

列出FP最多的查询。这类查询说明原始查询太宽泛，后续需要约束解析、元数据过滤和Selector。

### matched_rank_distribution.json

统计TP出现在什么排名段：

```text
1-10
11-20
21-50
51-100
```

如果大量TP出现在51-100，说明B0召回池有用，但排序很差。

### b1_recommendations.md

自动生成下一阶段建议：

- B1优先处理完全无召回查询；
- B2优先拆解复杂组合约束；
- B3再处理排序和Precision。

## 7. 验收标准

本步骤通过条件：

1. `pytest -q`全部通过；
2. 成功生成`analysis_summary.json`；
3. 成功生成`b0_result_table.md`；
4. `zero_recall_queries.jsonl`非空或明确为空；
5. `b1_recommendations.md`给出可执行建议；
6. 能明确回答B1优先改写哪些查询。

通过后进入第6步：

```text
B1：查询改写与关键词扩展
```
