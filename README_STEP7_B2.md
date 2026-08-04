# ScholarPath 第7步：B2约束感知多子查询召回

## 1. 当前目的

B1阶段结论是：

```text
规则查询改写能够提升候选池召回，但不能直接提升最终Top50 F1。
```

因此B2不继续做普通同义词改写，而是做：

```text
复杂查询 → 约束拆解 → 多子查询召回 → 候选约束覆盖标注
```

核心目标不是立刻让Top50 F1暴涨，而是为后续Reranker和Selector准备更好的候选池与约束证据。

## 2. B2和B1的区别

B1：

```text
原始query
→ cleaned / keyword / expanded
→ 多检索式合并
```

B2：

```text
原始query
→ 拆出多个约束
→ 每个子查询对应明确约束组合
→ 检索并合并
→ 每篇候选论文记录覆盖了哪些约束
```

例如：

```text
Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?
```

B2会拆成类似：

```text
约束1：scaling law
约束2：multi module / multimodal model
约束3：video text
约束4：image text
```

然后生成子查询：

```text
scaling law multimodal model
scaling law video text
scaling law image text
scaling law multimodal model video text image text
```

每篇候选论文会被标注：

```json
{
  "b2_covered_constraints": ["c1_scaling_law", "c3_video_text"],
  "b2_coverage_count": 2,
  "b2_subquery_hits": [...]
}
```

## 3. 新增文件

```text
src/scholarpath/query/constraints.py
src/scholarpath/retrieval/constraint_aggregation.py
scripts/run_b2_openalex.py
tests/test_constraint_decomposer.py
tests/test_constraint_aggregation.py
README_STEP7_B2.md
```

并修改：

```text
src/scholarpath/query/__init__.py
```

## 4. 先跑测试

```powershell
pytest -q
```

预期：全部通过。

## 5. 先跑1条Smoke测试

```powershell
python scripts/run_b2_openalex.py --limit-queries 1 --output-dir outputs/b2_openalex_smoke
```

查看约束拆解：

```powershell
Get-Content outputs\b2_openalex_smoke\query_decompositions.jsonl
```

## 6. 跑完整B2

```powershell
python scripts/run_b2_openalex.py --output-dir outputs/b2_openalex
```

默认参数：

```text
每条查询最多6个子查询
每个子查询返回35篇
输出Top20/Top50/Top100
Token=0
```

## 7. 查看B2结果

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2':'outputs/b2_openalex'}; [print(name,'Top50 MacroF1=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn']) for name,path in runs.items()]"
```

查看成本和调用：

```powershell
python -c "import json; d=json.load(open('outputs/b2_openalex/run_summary.json',encoding='utf-8')); print(json.dumps({'queries':d['queries'],'retrieval':d['retrieval'],'tokens':d['tokens'],'cost':d['estimated_cost_usd']},ensure_ascii=False,indent=2))"
```

## 8. 做B2错误分析

```powershell
python scripts/analyze_b0_results.py --b0-dir outputs/b2_openalex --output-dir outputs/b2_openalex_analysis --focus-topk 50 --mode strict
```

再查看：

```powershell
python -c "import json; d=json.load(open('outputs/b2_openalex_analysis/analysis_summary.json',encoding='utf-8')); print(json.dumps(d['error_analysis'],ensure_ascii=False,indent=2))"
```

重点对比：

```text
zero_recall_queries
total_matches
rank_distribution
```

## 9. 验收标准

B2通过条件：

1. 50条查询全部成功；
2. Token仍为0；
3. 每条查询都有约束拆解记录；
4. 每篇候选论文带有b2约束覆盖信息；
5. Top100 TP 或 zero_recall_queries 相比B0/B1.2有改善；
6. 即使Top50 F1未超过B0，也要能证明候选池和约束证据为后续Reranker/Selector提供价值。

B0参考：

```text
Top50 strict Macro F1 = 0.02183
Top100 TP = 44
zero_recall_queries = 35
```

B1.2参考：

```text
Top50 strict Macro F1 = 0.02053
Top100 TP = 53
zero_recall_queries = 31
```
