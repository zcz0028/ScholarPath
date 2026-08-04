# ScholarPath 第6.2步：B1.2 Raw-anchor候选聚合优化

## 1. 当前目的

B1_full和B1.1实验说明：

- 查询改写可以减少零召回；
- 但改写候选噪声较大；
- 简单删掉部分改写类型不能稳定超过B0；
- 主要瓶颈在多检索式候选聚合排序。

因此第6.2步不继续盲目扩大查询，而是加入：

```text
Raw-anchor候选聚合策略
```

核心思想：

```text
原始query是稳定主通道
cleaned query是辅助主通道
keyword_core是补充通道
补充通道不能轻易挤掉raw/cleaned中的高置信结果
```

## 2. 新增/修改文件

```text
src/scholarpath/retrieval/aggregation.py
scripts/run_b1_openalex.py
tests/test_candidate_aggregation.py
README_STEP6_2.md
```

## 3. 新增参数

```powershell
--aggregation-mode weighted
--aggregation-mode raw_anchor
--raw-page 100
```

含义：

- `weighted`：原B1聚合方式；
- `raw_anchor`：B1.2聚合方式；
- `raw-page`：raw原始查询单独取多少候选，默认在raw_anchor下取到最终最大Top-K，通常为100。

## 4. 推荐运行命令

```powershell
python scripts/run_b1_openalex.py --max-query-variants 3 --per-variant-page 40 --aggregation-mode raw_anchor --raw-page 100 --output-dir outputs/b1_2_raw_anchor_q3_p40
```

这表示：

```text
raw        取100篇，保证B0主通道不被截断
cleaned    取40篇
keyword    取40篇
聚合时raw/cleaned权重大，keyword只做补充
```

## 5. 对比命令

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1_full':'outputs/b1_openalex','B1.1_q3_p40':'outputs/b1_1_q3_p40','B1.2_raw_anchor':'outputs/b1_2_raw_anchor_q3_p40'}; 
for name,path in runs.items():
    d=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))
    r=d['evaluation']['top50_strict']
    print(name,'MacroF1=',r['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn'],'API=',d['retrieval']['actual_api_calls'],'Cost=',d['estimated_cost_usd'])"
```

PowerShell单行版：

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B1_full':'outputs/b1_openalex','B1.1_q3_p40':'outputs/b1_1_q3_p40','B1.2_raw_anchor':'outputs/b1_2_raw_anchor_q3_p40'}; [print(name,'MacroF1=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],'TP=',r['counts']['tp'],'FP=',r['counts']['fp'],'FN=',r['counts']['fn'],'API=',json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['retrieval']['actual_api_calls']) for name,path in runs.items()]"
```

## 6. 验收标准

B1.2通过条件：

1. 程序完整跑通，50条查询失败数为0；
2. Token仍为0；
3. Top50 strict Macro F1不低于B0；
4. 或者Top100 TP、zero_recall_queries明显优于B0，且Top50 F1不明显下降；
5. API调用不高于B1_full太多。

B0参考：

```text
Top50 strict Macro F1 = 0.02183
TP = 31
FP = 2198
API = 46
```

B1_full参考：

```text
Top50 strict Macro F1 = 0.01964
TP = 34
FP = 2442
API = 158
```
