# ScholarPath 第12步：B5.2 Guard-aware Selector

## 1. 为什么做B5.2？

当前已经形成两个有效模块：

```text
B5 Selector：能显著降低FP，B5_on_b4_precision 当前MacroF1最高
B5.1 Guard：能识别硬约束，B5_1_guard_balanced_v4 当前主推荐略优于B4
```

但它们各自有不足：

```text
B5_precision：指标高，但没有显式利用硬约束校验结果
B5.1_guard_balanced_v4：解释可信，但精筛能力不如B5_precision
```

因此 B5.2 将二者融合：

```text
B5 Selector分数
+ B5.1 Guard规则分数
+ Guard决策(pass/soft_pass/downrank/reject)
+ 缺失约束/违规约束惩罚
→ Guard-aware final score
```

目标：

```text
1. 保留B5的高精度精筛能力；
2. 引入B5.1的硬约束可信性；
3. 尝试超过 B5_on_b4_precision；
4. 输出更可信的推荐理由。
```

## 2. 输入

默认输入：

```text
outputs/b5_1_guard_balanced_v4/predictions_top100.jsonl
```

也就是已经带有硬规则校验字段的候选。

## 3. 输出字段

每篇论文 raw 中新增：

```text
b5_2_selector_score
b5_2_selector_label
b5_2_guard_score
b5_2_guard_decision
b5_2_final_score
b5_2_decision
b5_2_reason_tags
b5_2_reason_text
```

## 4. 新增文件

```text
src/scholarpath/selector/
└── guard_aware_selector.py

scripts/
└── run_b5_2_guard_aware_selector.py

tests/
└── test_guard_aware_selector.py

README_STEP12_B5_2.md
```

## 5. 先跑测试

```powershell
pytest -q
```

## 6. 跑B5.2三个版本

### 6.1 标注排序版

不强过滤，只融合 Guard + Selector 重新排序：

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_ranking --mode ranking
```

### 6.2 均衡版

推荐先看：

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_balanced --mode balanced
```

### 6.3 高精度版

用于冲击 B5_on_b4_precision：

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_precision --mode precision
```

### 6.4 高精度安全版

保底每条 query 最多补到8条，避免输出过少：

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_precision_safe --mode precision_safe
```

## 7. 查看Top20/Top50对比

```powershell
python -c "import json,pathlib; runs={'B4':'outputs/b4_fusion_semantic','B5_prec':'outputs/b5_on_b4_precision','G_bal_v4':'outputs/b5_1_guard_balanced_v4','B5_2_rank':'outputs/b5_2_guard_aware_ranking','B5_2_bal':'outputs/b5_2_guard_aware_balanced','B5_2_prec':'outputs/b5_2_guard_aware_precision','B5_2_safe':'outputs/b5_2_guard_aware_precision_safe'}; [print(name,'SKIP missing') if not (pathlib.Path(path)/'run_summary.json').exists() else print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts']) for name,path in runs.items()]"
```

## 8. 查看Top100对比

```powershell
python -c "import json,pathlib; runs={'B4':'outputs/b4_fusion_semantic','B5_prec':'outputs/b5_on_b4_precision','G_bal_v4':'outputs/b5_1_guard_balanced_v4','B5_2_rank':'outputs/b5_2_guard_aware_ranking','B5_2_bal':'outputs/b5_2_guard_aware_balanced','B5_2_prec':'outputs/b5_2_guard_aware_precision','B5_2_safe':'outputs/b5_2_guard_aware_precision_safe'}; [print(name,'SKIP missing') if not (pathlib.Path(path)/'run_summary.json').exists() else print(name,'Top100=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r['counts']) for name,path in runs.items()]"
```

## 9. 查看B5.2统计

```powershell
Get-Content -Encoding UTF8 outputs\b5_2_guard_aware_precision\guard_aware_stats.json
```

## 10. 查看理由样例

```powershell
Get-Content -Encoding UTF8 outputs\b5_2_guard_aware_precision\guard_aware_reason_samples.md
```

balanced也看一下：

```powershell
Get-Content -Encoding UTF8 outputs\b5_2_guard_aware_balanced\guard_aware_reason_samples.md
```

## 11. 判断标准

B5.2通过条件：

```text
1. 如果 MacroF1 > B5_on_b4_precision，则B5.2成为最高F1版本；
2. 如果 MacroF1略低，但样例解释明显更可信，则作为“可信高精度模式”保留；
3. 如果 balanced 在不损失TP的情况下进一步降低FP，则作为主推荐版本；
4. 如果没有任何收益，则归档为融合消融，最终仍用 G_bal_v4 + B5_prec 双模式。
```
