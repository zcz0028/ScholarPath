# ScholarPath 第10步：B4.0多来源候选池融合与B3重排

## 1. 当前目的

前面已经得到：

```text
B0：原始OpenAlex稳定基线
B1.2：召回增强候选池
B2：约束感知检索候选池，虽然直接输出差，但能降低部分零召回
B3：当前最强排序器
B5：解释与高精度精筛模块
```

现在要做的是：

```text
不新增API、不新增Token
复用已有候选池
融合 B0 + B1.2 + B2
再交给 B3 轻量语义Reranker 排序
```

核心目标：

```text
提高Top100候选覆盖
降低zero_recall
让B3在更大的候选池里重新排序
争取超过当前B3主结果
```

## 2. 为什么不是继续调B1？

B1单独输出已经证明不稳定；但B1.2作为候选池有价值。

现在不是重新调B1，而是：

```text
前面候选池统一融合
后面强Reranker重新排序
```

这比单独改B1检索式更合理。

## 3. B4输入

默认读取：

```text
outputs/b0_openalex/predictions_top100.jsonl
outputs/b1_2_raw_anchor_q3_p40/predictions_top100.jsonl
outputs/b2_openalex/predictions_top100.jsonl
data/raw/RealScholarQuery/test.jsonl
data/processed/realscholarquery_gold.jsonl
```

## 4. B4输出

```text
outputs/b4_fusion_semantic/
├── predictions_top20.jsonl
├── predictions_top50.jsonl
├── predictions_top100.jsonl
├── raw_predictions_top100.jsonl
├── fused_candidates_before_rerank.jsonl
├── fusion_stats.jsonl
├── query_decompositions.jsonl
└── run_summary.json
```

每篇论文会带：

```text
b4_sources
b4_source_count
b4_source_ranks
b4_pre_rerank_score
b4_best_source_rank
b4_fusion_reason_tags
```

## 5. 新增文件

```text
src/scholarpath/fusion/
├── __init__.py
└── candidate_fusion.py

scripts/
└── run_b4_candidate_fusion.py

tests/
└── test_candidate_fusion.py

README_STEP10_B4.md
```

## 6. 先跑测试

```powershell
pytest -q
```

## 7. 跑B4默认融合

```powershell
python scripts/run_b4_candidate_fusion.py --output-dir outputs/b4_fusion_semantic
```

默认参数：

```text
融合B0+B1.2+B2
每条query融合后最多保留200个候选
使用B3 semantic_first重排
输出Top20/50/100
API=0
Token=0
Cost=0
```

## 8. 跑B4保守融合

如果默认融合FP较高，可以跑保守版，只融合B0+B1.2：

```powershell
python scripts/run_b4_candidate_fusion.py --source b0=outputs/b0_openalex --source b1_2=outputs/b1_2_raw_anchor_q3_p40 --output-dir outputs/b4_fusion_b0_b12_semantic
```

## 9. 跑B4不含B0版本

用于消融，检查B0稳定通道是否必要：

```powershell
python scripts/run_b4_candidate_fusion.py --source b1_2=outputs/b1_2_raw_anchor_q3_p40 --source b2=outputs/b2_openalex --output-dir outputs/b4_fusion_b12_b2_semantic
```

## 10. 查看Top20/Top50对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B3':'outputs/b3_semantic_first','B4':'outputs/b4_fusion_semantic','B4_b0_b12':'outputs/b4_fusion_b0_b12_semantic','B4_b12_b2':'outputs/b4_fusion_b12_b2_semantic'}; [print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts']) for name,path in runs.items()]"
```

## 11. 查看Top100对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2':'outputs/b2_openalex','B3':'outputs/b3_semantic_first','B4':'outputs/b4_fusion_semantic','B4_b0_b12':'outputs/b4_fusion_b0_b12_semantic','B4_b12_b2':'outputs/b4_fusion_b12_b2_semantic'}; [print(name,'Top100=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r['counts']) for name,path in runs.items()]"
```

## 12. 看融合统计

```powershell
Get-Content outputs\b4_fusion_semantic\fusion_stats.jsonl
```

也可以看总体摘要：

```powershell
python -c "import json; d=json.load(open('outputs/b4_fusion_semantic/run_summary.json',encoding='utf-8')); print(json.dumps({'queries':d['queries'],'fusion':d['fusion'],'retrieval':d['retrieval'],'tokens':d['tokens'],'cost':d['estimated_cost_usd']},ensure_ascii=False,indent=2))"
```

## 13. 做错误分析

```powershell
python scripts/analyze_b0_results.py --b0-dir outputs/b4_fusion_semantic --output-dir outputs/b4_fusion_semantic_analysis --focus-topk 50 --mode strict
```

查看：

```powershell
python -c "import json; d=json.load(open('outputs/b4_fusion_semantic_analysis/analysis_summary.json',encoding='utf-8')); print(json.dumps(d['error_analysis'],ensure_ascii=False,indent=2))"
```

## 14. 判断标准

当前主结果B3参考：

```text
B3 Top20 MacroF1 = 0.02438, TP=21
B3 Top50 MacroF1 = 0.02271, TP=40
B3 Top100 TP = 53
zero_recall_queries = 31
```

B4通过条件：

```text
Top50 MacroF1 > B3
或 Top50 TP > B3 且FP没有明显恶化
或 Top100 TP > 53
或 zero_recall_queries < 31
```

如果B4没有提升，结论也有价值：

```text
已有候选池融合收益有限，下一步需要真正的外部受控扩展，例如引用/被引/相似论文扩展。
```
