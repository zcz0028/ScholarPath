# ScholarPath Week 2 Day 3 验收

## 1. 工作目标

针对 Day 1 识别出的 24 条真实 retrieval-miss 查询，构建基于学术锚点、任务别名、方法别名、Benchmark 和比较关系的定向查询规划器。

Planner 每条查询最多生成两条救援检索式，Day 3 只生成计划，不调用 OpenAlex。

## 2. 自动运行结果

- 查询数量：24
- 生成检索式数量：48
- 生成两条计划的查询：24
- 无计划查询：0
- Day 4 预计新增 API 调用：48
- Day 3 实际 API 调用：0
- Planner 是否读取 Gold 论文：否

## 3. 实现内容

- 显式论文标题锚点检索；
- Benchmark 与比较关系检索；
- 数据集、任务、方法和模型组合检索；
- 学术方法别名扩展；
- 任务别名扩展；
- 领域术语扩展；
- 歧义查询拆分；
- 清洗关键词兜底；
- 负约束与时间约束独立保存；
- 每条检索式记录生成原因、锚点和计划类型。

## 4. 人工审核结论

24 条 retrieval-miss 查询均生成了两条定向检索式。检索式基本保留了原查询的核心任务和方法约束，并补充了常用学术别名。

`exclude_survey` 等负约束未被用于召回阶段的提前过滤，而是保存在 filters 中，避免损失候选召回率。

所有计划均为确定性规则生成，可以在前端展示查询锚点、别名扩展、检索计划及生成原因。

## 5. Day 3 验收

- [x] pytest 显示 120 passed
- [x] query_count = 24
- [x] planned_query_count = 48
- [x] queries_with_two_plans = 24
- [x] queries_with_zero_plans = 0
- [x] estimated_api_calls = 48
- [x] api_calls_executed = 0
- [x] uncovered_queries.jsonl 为空
- [x] 每条计划均包含 plan_type、reason 和 anchor_texts
- [x] Gold 论文未进入 Planner
- [ ] 24 条检索计划已经人工检查