# ScholarPath 第11步：B5.1 Constraint Guard硬规则校验器

## 1. 为什么要做B5.1？

B5_on_B4_precision 的指标最高：

```text
Top50 MacroF1 = 0.04239
TP = 22
FP = 378
```

但理由样例暴露出问题：

```text
1. query要求 exclude survey papers，但结果中仍可能出现 survey/review；
2. query要求 HotPotQA、RLHF、reward shaping 等硬条件，但候选可能只是泛LLM论文；
3. query要求 visual+audio、image+video、autoregressive transformer 等组合条件，仅靠语义分难以保证满足。
```

所以 B5.1 增加一层硬约束校验：

```text
B4候选
→ B5.1硬约束识别与校验
→ 过滤/降权违反硬约束的论文
→ 重新评测
```

## 2. B5.1不做什么？

```text
不重新检索
不调用OpenAlex
不调用LLM
不增加Token成本
```

## 3. B5.1主要校验哪些规则？

### 3.1 排除约束

识别：

```text
exclude survey papers
not survey
non-survey
excluding review
```

过滤或降权：

```text
survey
review
systematic review
comprehensive survey
literature review
overview
```

### 3.2 数据集硬约束

识别：

```text
HotPotQA
MS COCO / COCO
ActivityNet
WebVid
AudioSet
VGGSound
Ego4D
HowTo100M
MSR-VTT
ImageNet
MMLU
GSM8K
HumanEval
MBPP
```

如果 query 明确要求数据集，而论文标题/摘要中没有出现该数据集，则降权或过滤。

### 3.3 方法硬约束

识别：

```text
RLHF
reward shaping
autoregressive transformer
in-context learning
pre-training
reinforcement learning
large language model agent
ranking/search result reranking
```

### 3.4 模态与任务组合约束

识别：

```text
visual + audio
image + video description
long video
video generation
hallucination
foundation model
```

## 4. 新增输出字段

每篇论文 raw 中新增：

```text
b5_1_guard_score
b5_1_guard_decision
b5_1_satisfied_constraints
b5_1_missing_constraints
b5_1_violation_tags
b5_1_evidence_tags
b5_1_guard_reason
b5_1_final_score
```

决策包括：

```text
pass
soft_pass
downrank
reject
```

## 5. 新增文件

```text
src/scholarpath/guard/
├── __init__.py
└── constraint_guard.py

scripts/
└── run_b5_1_constraint_guard.py

tests/
└── test_constraint_guard.py

README_STEP11_B5_1.md
```

## 6. 先跑测试

```powershell
pytest -q
```

## 7. 跑B5.1标注版

只标注和重排，不强过滤：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_annotate --guard-mode annotate
```

## 8. 跑B5.1均衡版

推荐先看这个：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced --guard-mode balanced
```

## 9. 跑B5.1严格版

更偏高精度，不补足低相关候选：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision --guard-mode precision
```

## 10. 查看B4、B5、B5.1对比

```powershell
python -c "import json,pathlib; runs={'B4':'outputs/b4_fusion_semantic','B5_bal':'outputs/b5_on_b4_balanced','B5_prec':'outputs/b5_on_b4_precision','G_ann':'outputs/b5_1_guard_annotate','G_bal':'outputs/b5_1_guard_balanced','G_prec':'outputs/b5_1_guard_precision'}; [print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts']) for name,path in runs.items()]"
```

## 11. 查看Top100

```powershell
python -c "import json,pathlib; runs={'B4':'outputs/b4_fusion_semantic','B5_bal':'outputs/b5_on_b4_balanced','B5_prec':'outputs/b5_on_b4_precision','G_ann':'outputs/b5_1_guard_annotate','G_bal':'outputs/b5_1_guard_balanced','G_prec':'outputs/b5_1_guard_precision'}; [print(name,'Top100=',(r:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r['counts']) for name,path in runs.items()]"
```

## 12. 查看规则统计

```powershell
Get-Content -Encoding UTF8 outputs\b5_1_guard_balanced\guard_stats.json
```

## 13. 查看规则样例

```powershell
Get-Content -Encoding UTF8 outputs\b5_1_guard_balanced\guard_reason_samples.md
```

## 14. 可选：把B5接在B5.1后面

如果B5.1 balanced有价值，可以再跑：

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b5_1_guard_balanced --output-dir outputs/b5_on_guard_balanced --selector-mode balanced
python scripts/run_b5_selector.py --candidate-dir outputs/b5_1_guard_balanced --output-dir outputs/b5_on_guard_precision --selector-mode precision
```

## 15. 判断标准

B5.1通过条件：

```text
1. 能过滤明显违反硬约束的候选；
2. 样例理由比B5更可信；
3. Top50 MacroF1不明显低于B4；
4. 或FP显著下降且TP损失可接受；
5. 对exclude survey、HotPotQA、RLHF等query产生明确规则解释。
```

如果B5.1没有直接超过B5_precision，也可以作为系统可信性模块保留。
