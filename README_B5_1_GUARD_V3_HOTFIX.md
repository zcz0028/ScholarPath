# B5.1 Constraint Guard v3 完整覆盖热修复

## 为什么需要v3？

v2 的测试文件覆盖成功了，但规则逻辑没有完整生效，导致新增4个规则测试失败。

v3 不再做片段替换，而是完整覆盖：

```text
src/scholarpath/guard/constraint_guard.py
tests/test_constraint_guard.py
```

## 修复重点

1. 增加 token 级触发逻辑，不只依赖短语精确匹配；
2. 稳定识别 smaller dataset / bigger datasets / better models；
3. 稳定识别 generate videos / video generation；
4. 稳定识别 image + video + description；
5. 稳定识别 rank search results by LLM；
6. 缺少约束时不再直接 `pass`，最多 `soft_pass`。

## 覆盖后运行

```powershell
pytest -q
```

然后重新跑：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced_v3 --guard-mode balanced
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision_v3 --guard-mode precision
```
