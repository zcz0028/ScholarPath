# ScholarPath 第9步：B5 Selector精筛与相关性理由标签

## 1. 当前目的

B3已经成为当前主排序版本：

```text
B3_semantic_first Top20 / Top50 均优于B0
```

但B3仍存在两个问题：

```text
1. Top50 FP仍然较多；
2. 输出结果缺少“为什么推荐”的理由标签。
```

因此B5做Selector：

```text
B3 Top100候选
→ 读取B3语义特征、B2约束覆盖、原排序信息
→ 计算Selector分数
→ 打相关性标签
→ 生成理由标签
→ 过滤或重排低相关论文
→ 重新评测
```

## 2. 是否请求API或LLM？

不请求。

```text
OpenAlex API = 0
LLM Token = 0
Cost = 0
```

## 3. B5输入

默认输入：

```text
outputs/b3_semantic_first/predictions_top100.jsonl
data/raw/RealScholarQuery/test.jsonl
data/processed/realscholarquery_gold.jsonl
```

## 4. B5输出字段

每篇论文会带：

```text
b5_selector_score
b5_relevance_label
b5_reason_tags
b5_reason_text
b5_missing_tags
b5_decision
```

标签包括：

```text
highly_relevant
partially_relevant
weakly_relevant
low_relevance
```

理由标签示例：

```text
core_constraints_covered
title_matches_query
abstract_supports_query
semantic_score_high
strong_identifier_available
```

缺失标签示例：

```text
missing_core_constraints
weak_title_match
low_semantic_evidence
```

## 5. 新增文件

```text
src/scholarpath/selector/
├── __init__.py
└── selector.py

scripts/
└── run_b5_selector.py

tests/
└── test_selector.py

README_STEP9_B5.md
```

## 6. 先跑测试

```powershell
pytest -q
```

## 7. 跑B5排序版

排序版不强过滤，只重新排序并标注理由：

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b3_semantic_first --output-dir outputs/b5_selector_ranking --selector-mode ranking
```

## 8. 跑B5均衡精筛版

均衡版过滤低相关候选，目标是在不明显损失TP的情况下降低FP：

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b3_semantic_first --output-dir outputs/b5_selector_balanced --selector-mode balanced
```

## 9. 跑B5严格精筛版

严格版更偏Precision，可能FP下降明显，但TP也可能下降：

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b3_semantic_first --output-dir outputs/b5_selector_precision --selector-mode precision
```

## 10. 跑B5召回安全版

召回安全版主要保留B3排序，只过滤明显低相关：

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b3_semantic_first --output-dir outputs/b5_selector_recall_safe --selector-mode recall_safe
```

## 11. 查看Top20/Top50对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B3':'outputs/b3_semantic_first','B5_rank':'outputs/b5_selector_ranking','B5_bal':'outputs/b5_selector_balanced','B5_prec':'outputs/b5_selector_precision','B5_safe':'outputs/b5_selector_recall_safe'}; [print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts']) for name,path in runs.items()]"
```

## 12. 查看Top100对比

```powershell
python -c "import json,pathlib; runs={'B0':'outputs/b0_openalex','B3':'outputs/b3_semantic_first','B5_rank':'outputs/b5_selector_ranking','B5_bal':'outputs/b5_selector_balanced','B5_prec':'outputs/b5_selector_precision','B5_safe':'outputs/b5_selector_recall_safe'}; [print(name,'Top100=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r['counts']) for name,path in runs.items()]"
```

## 13. 查看标签统计

```powershell
Get-Content outputs\b5_selector_balanced\selector_label_stats.json
```

## 14. 查看理由样例

```powershell
Get-Content outputs\b5_selector_balanced\selector_reason_samples.md
```

## 15. 判断标准

B5通过条件：

```text
Top50 MacroF1 >= B3
或 FP明显低于B3且TP损失可接受
或 相关性理由标签可用于前端展示/答辩
```

B3参考：

```text
B3_semantic_first Top20 MacroF1 = 0.02438, TP=21
B3_semantic_first Top50 MacroF1 = 0.02271, TP=40
```

如果B5没有超过B3，也不代表失败。它仍可以作为：

```text
相关性理由标签模块
最终展示解释模块
人工可读筛选模块
```
