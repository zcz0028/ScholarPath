# ScholarPath 第1周第3步：Precision、Recall与F1评测

## 当前目的

建立可重复、可审计的论文检索评测脚本，同时保留两套口径：

- `strict`：ScholarPath主评测口径，优先强ID，再用标准化标题兜底；
- `pasa_title`：复现PaSa只保留标题字母的兼容口径。

## 新增文件

```text
src/scholarpath/evaluation/
├── __init__.py
├── io.py
├── matching.py
└── metrics.py

scripts/run_evaluation.py

tests/
├── test_evaluation_io.py
├── test_evaluation_matching.py
└── test_evaluation_metrics.py

data/example/evaluation/
├── gold.jsonl
└── predictions.jsonl
```

## 输入格式

Gold支持第2步生成格式：

```json
{"qid":"q1","question":"...","gold_papers":[{"title":"Paper A","arxiv_id":"2401.01234"}]}
```

也支持原始RealScholarQuery格式：

```json
{"qid":"q1","question":"...","answer":["Paper A"],"answer_arxiv_id":["2401.01234"]}
```

预测格式：

```json
{"qid":"q1","papers":[{"title":"Paper A","arxiv_id":"2401.01234v2"}]}
```

`papers`也可以是标题字符串列表。

## 运行测试

```powershell
pytest -q
```

预期：

```text
26 passed
```

## 运行严格评测示例

```powershell
python scripts/run_evaluation.py --gold data/example/evaluation/gold.jsonl --predictions data/example/evaluation/predictions.jsonl --output-dir outputs/step3_strict_sample --mode strict --strict-qids
```

预期核心结果：

```text
Queries: 2
Macro P/R/F1: 0.750000 / 0.750000 / 0.750000
Micro P/R/F1: 0.666667 / 0.666667 / 0.666667
```

## 运行PaSa兼容口径

```powershell
python scripts/run_evaluation.py --gold data/example/evaluation/gold.jsonl --predictions data/example/evaluation/predictions.jsonl --output-dir outputs/step3_pasa_sample --mode pasa_title --strict-qids
```

## 输出文件

```text
summary.json      # Macro/Micro P、R、F1与Recall@K
per_query.jsonl   # 每条查询的完整匹配结果
errors.jsonl      # 每条查询的FP和FN
```

## 指标定义

每条查询：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

汇总同时输出：

- Macro Precision：查询级Precision平均；
- Macro Recall：查询级Recall平均；
- Macro F1：查询级F1平均；
- harmonic_of_macro_precision_recall：Macro P和Macro R的调和平均，仅作为补充；
- Micro Precision/Recall/F1：先汇总TP、FP、FN再计算；
- Recall@20、Recall@50、Recall@100。

注意：`Macro F1`不等于`Macro P与Macro R的调和平均`。

## 严格匹配规则

1. DOI、arXiv、OpenAlex或Semantic Scholar ID相同；
2. 若没有强ID匹配，再使用标准化标题精确匹配；
3. 若双方存在同类型强ID冲突，即使标题相同也不匹配；
4. 每篇预测最多匹配一篇Gold，每篇Gold最多匹配一次；
5. 默认先去除精确重复预测，避免重复结果虚增FP。

## 验收标准

1. 全部26项测试通过；
2. 示例严格评测结果符合预期；
3. `summary.json`同时包含Macro和Micro指标；
4. `per_query.jsonl`可查看每个TP的匹配理由；
5. `errors.jsonl`准确列出FP和FN；
6. 原始RealScholarQuery与第2步Gold格式都能读取；
7. 缺失查询会按空预测计分，而不是被静默跳过；
8. 多余qid会记录，使用`--strict-qids`时直接报错；
9. PaSa兼容口径与ScholarPath严格口径分开保存。

## 是否允许进入下一步

第3步通过后，才允许进入第4步“接入一个学术搜索API并跑通B0基线”。
