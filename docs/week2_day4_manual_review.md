# ScholarPath Week 2 Day 4 验收

## 1. 工作目标

针对 Day 1 确认的 24 条真实 `retrieval_miss` 查询，执行 Day 3 生成的 48 条学术锚点检索计划，并将新增候选与 B4 完整候选池融合和语义重排。

非目标查询保留原 B4 输出，从而避免已有命中因本轮实验发生退化。

## 2. 运行配置

- 目标查询数：24
- 每条查询最多计划数：2
- 计划总数：48
- 每个计划返回数：50
- API 调用上限：48
- Rescue 候选上限：100/查询
- 融合候选上限：300/查询
- 语义重排模式：`semantic_first`
- Gold 是否进入召回：否
- LLM Token：0

## 3. 自动运行结果

请根据 `outputs/week2_day4_rescue/run_summary.json` 填写：

- 实际 API 调用：
- Cache 命中：
- 失败计划数：
- 估算成本：
- B4 Top20 TP：
- Day 4 Top20 TP：
- B4 Top50 TP：
- Day 4 Top50 TP：
- B4 Top100 TP：61
- Day 4 Top100 TP：
- Top100 TP 增量：
- B4 zero-recall：26
- Day 4 zero-recall：
- 恢复查询数：

## 4. 恢复查询人工复核

请根据 `recovered_queries.jsonl` 填写：

| QID | 命中 Gold | 命中排名 | 对应计划 | 判定 |
|---|---|---:|---|---|
|  |  |  |  | 真实恢复/疑似版本匹配 |

## 5. 未恢复查询复核

请根据 `unrecovered_queries.jsonl` 和 `plan_results.jsonl` 分类：

| QID | 两条计划均未召回 | Gold 位于结果 50 名之后 | 计划表达不准确 | 适合引文扩展 | 下一步 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 6. Day 4 判定

- [ ] pytest 显示 130 passed
- [ ] Dry Run 显示 24 条查询、48 条计划、0 次实际调用
- [ ] 正式运行实际 API 调用不超过 48
- [ ] failed_plans.jsonl 已检查
- [ ] Top20 TP 不低于 37
- [ ] Top100 TP 不低于 61，不发生回退
- [ ] recovered_queries.jsonl 中的命中均已人工确认
- [ ] Gold 未进入召回、融合和重排
- [ ] LLM Token 为 0

### 目标达成情况

- [ ] Top100 TP 不低于 69
- [ ] zero-recall 不高于 20

若两项目标均达到，Day 4 判定为“达标”。若指标有真实提升但未达到目标，判定为“部分达标”，保留有效计划并在 Day 5 开展受控引文扩展和消融实验。
