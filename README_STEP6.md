# ScholarPath 第1周第6步：B1查询改写与关键词扩展

## 1. 当前目的

第5步错误分析显示，B0在Top100下仍有大量零召回查询，因此第6步进入B1：

```text
B1 = 原始查询 + 确定性查询改写 + 关键词扩展 + 多检索式合并
```

本步骤仍然不做：

- 多Agent；
- LLM改写；
- Reranker；
- Selector；
- 引文扩展；
- 前端和图片输入。

## 2. B1和B0的区别

B0：

```text
每条question只调用一次OpenAlex
```

B1：

```text
每条question生成最多5条检索式
→ 分别调用OpenAlex
→ 合并候选论文
→ 统一ID去重
→ 输出Top20/50/100
→ 自动评测
```

## 3. 新增文件

```text
src/scholarpath/query/
├── __init__.py
└── rewriter.py

src/scholarpath/retrieval/
└── aggregation.py

scripts/
└── run_b1_openalex.py

tests/
├── test_query_rewriter.py
└── test_candidate_aggregation.py

README_STEP6.md
```

## 4. B1改写策略

当前使用确定性规则，不调用LLM：

1. 去掉疑问句和口语前缀；
2. 清理标点和连字符；
3. 提取核心关键词；
4. 保留学术术语、模型名、数据集名、任务名；
5. 少量同义词扩展，例如：
   - `scaling law` → `scaling laws`
   - `image text` → `vision language`
   - `video text` → `video language`
   - `multi module` → `multimodal`
   - `retrieval augmented generation` → `RAG`

## 5. 运行测试

```powershell
pytest -q
```

预期：全部通过，测试数会比第5步更多。

## 6. 先跑1条查询

```powershell
python scripts/run_b1_openalex.py --limit-queries 1 --output-dir outputs/b1_openalex_smoke
```

预期看到：

```text
[OK] Queries: 1
[OK] Actual API calls: 1~5
[OK] Total tokens: 0
```

## 7. 跑完整B1

```powershell
python scripts/run_b1_openalex.py --output-dir outputs/b1_openalex
```

默认参数：

```text
每条查询最多5个检索式
每个检索式返回40篇
最终输出Top20、Top50、Top100
```

如果想省API，先跑前10条：

```powershell
python scripts/run_b1_openalex.py --limit-queries 10 --output-dir outputs/b1_openalex_10
```

## 8. 预期输出

```text
outputs/b1_openalex/
├── run_summary.json
├── query_logs.jsonl
├── query_variants.jsonl
├── raw_predictions_top100.jsonl
├── predictions_top20.jsonl
├── predictions_top50.jsonl
├── predictions_top100.jsonl
└── evaluation/
    ├── top20/
    │   ├── strict/
    │   └── pasa_title/
    ├── top50/
    └── top100/
```

## 9. 如何比较B0和B1

重点看：

```text
outputs/b0_openalex/run_summary.json
outputs/b1_openalex/run_summary.json
```

对比：

- Top50 strict Macro F1；
- Top50 strict Macro Recall；
- Top100 strict TP；
- zero recall query是否减少；
- API调用是否增加过多；
- Token是否仍为0。

## 10. 验收标准

B1通过条件：

1. `pytest -q`全部通过；
2. 50条查询全部成功；
3. `failed_queries=0`；
4. 生成Top20/50/100预测文件；
5. strict和pasa_title共6组评测均生成；
6. Token仍为0；
7. Top50或Top100 Recall相比B0不下降；
8. 最好Macro F1不低于B0。

若B1 Recall提升但Precision下降严重，下一步不继续扩大检索，而进入B2“多子查询召回与候选控制”。
