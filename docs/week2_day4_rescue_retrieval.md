# ScholarPath Week 2 Day 4：Anchor-aware Rescue Retrieval

## 1. 目标

只针对 Day 1 判定为 `retrieval_miss` 的 24 条查询执行 Day 3 生成的 48 条检索计划。非目标查询直接保留 B4 结果，避免已有命中退化。

## 2. 数据流

```text
Day 3 query_plans.jsonl
        ↓
每条计划调用一次 OpenAlex（最多 48 次）
        ↓
按 DOI/arXiv/OpenAlex ID 去重并聚合两路结果
        ↓
与 B4 完整候选池融合
        ↓
仅重排 24 条 retrieval-miss 查询
        ↓
其余 26 条查询原样保留 B4 Top20/50/100
        ↓
strict / strict_v2 / pasa_title 评测
```

## 3. 关键控制

- 每条查询最多 2 条检索计划；
- 总 API 预算默认不超过 48 次；
- Gold 不进入召回、融合或重排；
- Gold 仅在检索结束后用于离线评测；
- 非目标查询的 B4 输出保持不变；
- 每个候选保留计划来源、排名、锚点和聚合得分。

## 4. 建议验收目标

- Top100 TP：由 61 提升至不低于 69；
- zero-recall：由 26 降至不高于 20；
- Top20 TP：不低于 37；
- 实际 API 调用：不超过 48；
- 新增 LLM Token：0。
