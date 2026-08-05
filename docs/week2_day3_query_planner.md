# ScholarPath Week 2 Day 3：Anchor-aware Query Planner

## 1. Day 3 目标

Day 3 不调用 OpenAlex，也不修改正式候选集。核心目标是把 Day 1 识别出的 24 条 `retrieval_miss` 查询转化为可解释、可控预算的定向救援检索计划。

每条查询最多生成两条检索式，并记录：

- 原始查询；
- 自动识别的学术锚点；
- 领域或任务别名；
- 检索计划类型；
- 生成原因；
- 预计 API 调用数；
- 负约束和时间约束。

Gold 论文不进入 Planner。Planner 的输入只有 `qid` 和 `question`。

## 2. 新增文件

```text
src/scholarpath/query/academic_query_planner.py
scripts/build_anchor_query_plans.py
tests/test_academic_query_planner.py
docs/week2_day3_query_planner.md
README_DAY3.md
```

Day 3 不覆盖 Day 1、Day 2 的已有代码。

## 3. Planner 的主要策略

Planner 按优先级生成检索式：

1. 显式论文标题锚点；
2. Benchmark 与比较关系；
3. 数据集、方法、模型和任务组合；
4. 领域别名扩展；
5. 任务别名扩展；
6. 歧义拆分；
7. 清洗后的关键词兜底。

重点覆盖：

- `in-context learning` 与 induction heads；
- 长视频描述与 long-form/dense video captioning；
- trigger-free document-level event extraction；
- HumanEval、MBPP、CodeContests 难度比较；
- neural quantum states / variational Monte Carlo；
- quantized / low-bit LLM pretraining；
- identity-consistent video generation；
- frame selection / keyframe sampling / temporal token pruning；
- cryptographic private learning；
- robot planning benchmark；
- financial LLM agents；
- alpha factor discovery；
- VLM game-playing agents。

## 4. 运行方法

在项目根目录执行：

```cmd
pytest -q
python scripts\build_anchor_query_plans.py
```

预期测试结果：

```text
120 passed
```

预期 Planner 输出：

```text
Query count: 24
Planned query count: 48
Queries with two plans: 24
Queries with zero plans: 0
Estimated Day-4 API calls: 48
API calls executed: 0
```

## 5. 输出文件

```text
outputs/week2_day3_query_plans/
├── planner_summary.json
├── query_plans.jsonl
├── planner_coverage.csv
├── uncovered_queries.jsonl
└── day3_query_plan_review.md
```

### `planner_summary.json`

记录查询数、计划数、计划类型、锚点类型、预计 Day 4 API 调用和实际 Day 3 API 调用。

### `query_plans.jsonl`

每条查询一行，包含两条定向检索式、锚点、别名、过滤条件、生成原因和调用预算。Day 4 的救援召回直接读取此文件。

### `planner_coverage.csv`

用于 Excel 检查，每行包含 QID、锚点类型、计划类型和两条检索式。

### `uncovered_queries.jsonl`

正常应为空。若不为空，说明存在无法生成有效检索计划的查询，不能直接进入 Day 4。

### `day3_query_plan_review.md`

可读性最好的人工审核报告，应逐条检查 24 条查询的两条检索式。

## 6. 负约束处理

例如 `exclude survey` 不直接拼入 OpenAlex 搜索词，也不在召回阶段删除候选，而是保存为：

```json
{
  "negative_constraints": ["exclude_survey"]
}
```

Day 4 召回后，再由 Selector 或 Constraint Guard 处理，以避免过早过滤造成 Recall 损失。

## 7. Day 3 验收

```text
[ ] pytest 显示 120 passed
[ ] query_count = 24
[ ] planned_query_count = 48
[ ] queries_with_two_plans = 24
[ ] queries_with_zero_plans = 0
[ ] estimated_api_calls = 48
[ ] api_calls_executed = 0
[ ] uncovered_queries.jsonl 为空
[ ] 每条计划均包含 plan_type、reason 和 anchor_texts
[ ] Gold 论文未进入 Planner
[ ] 24 条检索计划已经人工检查
```

Day 3 通过后，Day 4 才执行 `query_plans.jsonl` 中冻结的检索式，并分别统计：新增候选、Gold 命中、恢复的 zero-recall 查询、FP、延迟和成本。
