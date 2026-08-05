# ScholarPath Week 2 Day 1：阶段文件检查表

本文档固定第一周各阶段中用于第二周诊断的输入文件，并记录它们的实际 JSONL 结构。Day 1 的诊断脚本只读取这些离线文件，不调用 OpenAlex，也不修改任何第一周输出。

## 1. Gold 标准答案

- 阶段名：`gold`
- 文件：`data/processed/realscholarquery_gold.jsonl`
- 记录数：50
- 顶层字段：
  - `qid`
  - `question`
  - `published_time`
  - `gold_papers`
- 论文列表字段：`gold_papers`
- 论文主要字段：
  - `title`
  - `source`
  - `source_record_id`
  - `authors`
  - `year`
  - `publication_date`
  - `venue`
  - `doi`
  - `arxiv_id`
  - `openalex_id`
  - `semantic_scholar_id`
  - `url`
  - `abstract`
  - `citation_count`
  - `normalized_title`
  - `canonical_id`

## 2. B0：原始 OpenAlex 召回

- 阶段名：`b0_top100`
- 文件：`outputs/b0_openalex/predictions_top100.jsonl`
- 记录数：50
- 顶层字段：`qid`、`question`、`papers`
- 论文列表字段：`papers`
- 每条查询最多 100 篇论文
- 排名方式：按照 `papers` 数组顺序，数组下标 0 对应 Rank 1
- 论文对象中没有统一的显式 `rank` 字段

## 3. B1.2：Raw-anchor 查询改写召回

- 阶段名：`b1_2_top100`
- 文件：`outputs/b1_2_raw_anchor_q3_p40/predictions_top100.jsonl`
- 记录数：50
- 顶层字段：`qid`、`question`、`papers`
- 论文列表字段：`papers`
- 每条查询最多 100 篇论文
- 排名方式：按照 `papers` 数组顺序

## 4. B2：约束感知多子查询召回

- 阶段名：`b2_top100`
- 文件：`outputs/b2_openalex/predictions_top100.jsonl`
- 记录数：50
- 顶层字段：`qid`、`question`、`papers`
- 论文列表字段：`papers`
- 每条查询最多 100 篇论文
- 排名方式：按照 `papers` 数组顺序

## 5. B4：融合后的完整候选池

- 阶段名：`b4_pool`
- 文件：`outputs/b4_fusion_semantic/fused_candidates_before_rerank.jsonl`
- 记录数：50
- 顶层字段：`qid`、`question`、`papers`、`fusion`
- 论文列表字段：`papers`
- 每条查询通常保留 200 篇候选论文
- 排名方式：按照 `papers` 数组顺序
- `fusion` 字段包含融合策略、来源名称和配置
- 该文件用于区分“没有召回”与“进入候选池但排序低于 Top100”

## 6. B4：语义重排结果

| 阶段名 | 文件 | 记录数 | 排名方式 |
|---|---|---:|---|
| `b4_top20` | `outputs/b4_fusion_semantic/predictions_top20.jsonl` | 50 | `papers` 数组顺序 |
| `b4_top50` | `outputs/b4_fusion_semantic/predictions_top50.jsonl` | 50 | `papers` 数组顺序 |
| `b4_top100` | `outputs/b4_fusion_semantic/predictions_top100.jsonl` | 50 | `papers` 数组顺序 |

## 7. B5.1：主推荐版本 Constraint Guard

| 阶段名 | 文件 | 记录数 | 排名方式 |
|---|---|---:|---|
| `b5_1_top20` | `outputs/b5_1_guard_balanced_v4/predictions_top20.jsonl` | 50 | `papers` 数组顺序 |
| `b5_1_top50` | `outputs/b5_1_guard_balanced_v4/predictions_top50.jsonl` | 50 | `papers` 数组顺序 |
| `b5_1_top100` | `outputs/b5_1_guard_balanced_v4/predictions_top100.jsonl` | 50 | `papers` 数组顺序 |

## 8. B5：最高 F1 高精度精选版本

- 阶段名：`b5_precision`
- 文件：`outputs/b5_on_b4_precision/predictions_top100.jsonl`
- 记录数：50
- 顶层字段：`qid`、`question`、`papers`
- 论文列表字段：`papers`
- 注意：文件名虽然是 `predictions_top100.jsonl`，但每条查询的实际结果数是可变的，通常远少于 100；它是 Selector 精筛后的完整结果，不应强制要求每条有 100 篇
- 排名方式：按照 `papers` 数组顺序

## 9. B5.2：Guard-aware Selector 均衡版本

| 阶段名 | 文件 | 记录数 | 排名方式 |
|---|---|---:|---|
| `b5_2_top20` | `outputs/b5_2_guard_aware_balanced/predictions_top20.jsonl` | 50 | `papers` 数组顺序 |
| `b5_2_top50` | `outputs/b5_2_guard_aware_balanced/predictions_top50.jsonl` | 50 | `papers` 数组顺序 |
| `b5_2_top100` | `outputs/b5_2_guard_aware_balanced/predictions_top100.jsonl` | 50 | `papers` 数组顺序 |

## 10. 统一字段约定

除 Gold 外，当前各阶段文件均采用：

```json
{
  "qid": "RealScholarQuery_0",
  "question": "...",
  "papers": [
    {
      "title": "...",
      "authors": [],
      "year": 2024,
      "doi": null,
      "arxiv_id": null,
      "openalex_id": "W...",
      "semantic_scholar_id": null,
      "url": "...",
      "normalized_title": "...",
      "canonical_id": "..."
    }
  ]
}
```

诊断代码统一复用：

- `scholarpath.evaluation.io.load_gold`
- `scholarpath.evaluation.io.load_predictions`
- `scholarpath.evaluation.matching.match_papers`

因此，不在诊断模块中重新实现 JSONL 解析和正式匹配规则。

## 11. Day 1 完整性检查项

运行诊断前应确认：

- [ ] Gold 恰好包含 50 个唯一 qid；
- [ ] 每个阶段文件恰好包含 50 个唯一 qid；
- [ ] 各阶段 qid 集合与 Gold 完全一致；
- [ ] 顶层论文列表字段可被现有 `load_predictions` 读取；
- [ ] 候选论文至少能够解析 `title`；
- [ ] DOI、arXiv ID、OpenAlex ID、URL、authors、year 缺失时允许为空；
- [ ] 排名统一按 `papers` 数组顺序计算；
- [ ] B5 精选版本不检查每条查询的论文数是否达到 100；
- [ ] 全部诊断过程 API 调用数为 0；
- [ ] Gold 仅用于 `evaluation` 目录下的离线诊断，不进入检索和查询改写。
