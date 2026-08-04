# B5.1 Constraint Guard 热修复

## 修复内容

修复 B5.1 规则触发过宽的问题。

原问题：

```text
query 只要出现 large language models
就会误触发 llm_agent / ranking_with_llm 等复合方法约束
```

导致 HotPotQA 测试中，明明论文满足 HotPotQA，却因为缺少 agent、ranking 等额外条件被降权。

修复后：

```text
llm_agent：必须同时出现 LLM + agent
ranking_with_llm：必须同时出现 ranking/search/retrieval + LLM
autoregressive_transformer：必须同时出现 autoregressive + transformer
```

## 覆盖文件

```text
src/scholarpath/guard/constraint_guard.py
```

## 覆盖后运行

```powershell
pytest -q
```

然后重新跑：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_annotate --guard-mode annotate
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced --guard-mode balanced
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision --guard-mode precision
```
