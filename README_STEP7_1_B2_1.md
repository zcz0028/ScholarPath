# ScholarPath 第7.1步：B2.1 约束覆盖标注与轻量重排

## 1. 当前目的

B2直接多子查询检索结果不理想，但B2的约束拆解是有价值的：

```text
B1.2：候选池更强，Top100 TP=53
B2：约束拆解更有解释性，zero_recall_queries更低
```

所以本步骤不再让B2重新搜索，而是：

```text
B1.2候选池
→ B2约束拆解
→ 给每篇候选论文打约束覆盖标签
→ 轻量重排
→ 重新评测Top20/50/100
```

## 2. 是否请求OpenAlex？

不请求。

本步骤只读取：

```text
outputs/b1_2_raw_anchor_q3_p40/predictions_top100.jsonl
data/raw/RealScholarQuery/test.jsonl
```

因此：

```text
API调用 = 0
Token = 0
Cost = 0
```

## 3. 新增文件

```text
src/scholarpath/rerank/
├── __init__.py
└── constraint_rerank.py

scripts/
└── rerank_b1_with_constraints.py

tests/
└── test_constraint_rerank.py

README_STEP7_1_B2_1.md
```

## 4. 先跑测试

```powershell
pytest -q
```

## 5. 跑B2.1保守版

```powershell
python scripts/rerank_b1_with_constraints.py --candidate-dir outputs/b1_2_raw_anchor_q3_p40 --output-dir outputs/b2_1_constraint_rerank_conservative --rerank-mode conservative
```

保守版特点：

```text
主要保留B1.2原排序
只给约束覆盖较好的论文小幅加分
风险低
```

## 6. 跑B2.1混合版

```powershell
python scripts/rerank_b1_with_constraints.py --candidate-dir outputs/b1_2_raw_anchor_q3_p40 --output-dir outputs/b2_1_constraint_rerank_hybrid --rerank-mode hybrid
```

混合版特点：

```text
B1.2原排序 + 约束覆盖分
更激进一些
```

## 7. 查看对比结果

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2':'outputs/b2_openalex','B2.1_cons':'outputs/b2_1_constraint_rerank_conservative','B2.1_hybrid':'outputs/b2_1_constraint_rerank_hybrid'}; [print(name,'Top50 MacroF1=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn']) for name,path in runs.items()]"
```

查看Top100：

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2.1_cons':'outputs/b2_1_constraint_rerank_conservative','B2.1_hybrid':'outputs/b2_1_constraint_rerank_hybrid'}; [print(name,'Top100 MacroF1=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn']) for name,path in runs.items()]"
```

## 8. 做错误分析

```powershell
python scripts/analyze_b0_results.py --b0-dir outputs/b2_1_constraint_rerank_hybrid --output-dir outputs/b2_1_constraint_rerank_hybrid_analysis --focus-topk 50 --mode strict
```

查看：

```powershell
python -c "import json; d=json.load(open('outputs/b2_1_constraint_rerank_hybrid_analysis/analysis_summary.json',encoding='utf-8')); print(json.dumps(d['error_analysis'],ensure_ascii=False,indent=2))"
```

## 9. 判断标准

B2.1通过条件：

```text
Top50 MacroF1 >= B1.2
或 Top50 TP >= B1.2 且 FP没有明显恶化
或 Top20 TP提升，说明约束重排把相关论文前移
```

参考线：

```text
B0 Top50 MacroF1 = 0.02183, TP=31
B1.2 Top50 MacroF1 = 0.02053, TP=36
B2 Top50 MacroF1 = 0.01593, TP=26
```

如果B2.1没有超过B1.2，也不是失败，而是说明：

```text
仅靠词面约束覆盖仍不足以完成可靠排序
下一步必须进入B3语义Reranker
```
