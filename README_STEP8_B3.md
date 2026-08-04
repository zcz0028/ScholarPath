# ScholarPath 第8步：B3轻量语义Reranker

## 1. 当前目的

前面实验结论：

```text
B1.2：候选池增强有效
B2：直接多子查询检索噪声较大
B2.1：约束覆盖重排有效，尤其Top20提升明显
```

但B2.1仍没有超过B0的Top50 MacroF1，说明：

```text
词面约束覆盖有用，但不足以可靠判断论文是否真正回答原查询。
```

因此B3开始做语义相关性重排。

本补丁实现的是：

```text
B3.0：轻量语义特征Reranker
```

它不调用OpenAlex，不调用LLM，不消耗Token，只复用已有候选池。

## 2. B3.0输入

默认读取：

```text
outputs/b2_1_constraint_rerank_hybrid/predictions_top100.jsonl
data/raw/RealScholarQuery/test.jsonl
```

## 3. B3.0做什么

流程：

```text
B2.1_hybrid Top100候选
→ 查询清洗与关键词扩展
→ B2约束拆解
→ 提取论文标题/摘要/venue/concepts文本
→ 计算轻量语义相关性特征
→ 融合原排序、B2.1约束分、语义分
→ 输出Top20/50/100
→ 自动评测
```

主要特征：

```text
query_term_overlap
expanded_query_overlap
title_query_overlap
constraint_phrase_coverage
core_constraint_coverage
abstract_support
original_rank_score
b2_1_constraint_score
```

## 4. 是否有API/Token成本

没有。

```text
API calls = 0
Token = 0
Cost = 0
```

## 5. 新增文件

```text
src/scholarpath/rerank/semantic_rerank.py
scripts/run_b3_semantic_rerank.py
tests/test_semantic_rerank.py
README_STEP8_B3.md
```

并修改：

```text
src/scholarpath/rerank/__init__.py
```

## 6. 先跑测试

```powershell
pytest -q
```

## 7. 跑B3保守版

```powershell
python scripts/run_b3_semantic_rerank.py --candidate-dir outputs/b2_1_constraint_rerank_hybrid --output-dir outputs/b3_semantic_conservative --rerank-mode conservative
```

## 8. 跑B3混合版

```powershell
python scripts/run_b3_semantic_rerank.py --candidate-dir outputs/b2_1_constraint_rerank_hybrid --output-dir outputs/b3_semantic_hybrid --rerank-mode hybrid
```

## 9. 跑B3语义优先版

```powershell
python scripts/run_b3_semantic_rerank.py --candidate-dir outputs/b2_1_constraint_rerank_hybrid --output-dir outputs/b3_semantic_first --rerank-mode semantic_first
```

## 10. 查看Top20/Top50对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2.1':'outputs/b2_1_constraint_rerank_hybrid','B3_cons':'outputs/b3_semantic_conservative','B3_hybrid':'outputs/b3_semantic_hybrid','B3_sem':'outputs/b3_semantic_first'}; [print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts']) for name,path in runs.items()]"
```

## 11. 查看Top100对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1.2':'outputs/b1_2_raw_anchor_q3_p40','B2.1':'outputs/b2_1_constraint_rerank_hybrid','B3_cons':'outputs/b3_semantic_conservative','B3_hybrid':'outputs/b3_semantic_hybrid','B3_sem':'outputs/b3_semantic_first'}; [print(name,'Top100=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r['counts']) for name,path in runs.items()]"
```

## 12. 查看成本

```powershell
python -c "import json; d=json.load(open('outputs/b3_semantic_hybrid/run_summary.json',encoding='utf-8')); print(json.dumps({'queries':d['queries'],'retrieval':d['retrieval'],'tokens':d['tokens'],'cost':d['estimated_cost_usd']},ensure_ascii=False,indent=2))"
```

## 13. 判断标准

B3通过条件：

```text
Top20 MacroF1 >= B2.1_hybrid
或 Top50 MacroF1 >= B2.1_hybrid
或 Top50 TP >= B2.1_hybrid 且 FP没有明显恶化
```

参考线：

```text
B0 Top50 MacroF1 = 0.02183, TP=31
B2.1_hybrid Top20 MacroF1 = 0.02052, TP=17
B2.1_hybrid Top50 MacroF1 = 0.02117, TP=37
```

如果B3超过B0的Top50 MacroF1，说明轻量Reranker已经能成为当前主结果。

如果B3没有超过B2.1，说明纯规则/词项语义仍不足，需要进入B3.1：CrossEncoder或LLM Judge式Reranker。
