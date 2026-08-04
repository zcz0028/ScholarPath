# B5.1 Constraint Guard v2 热修复

## 修复目标

根据规则样例，补强以下问题：

1. `smaller dataset better than bigger datasets` 没有被识别为比较关系；
2. `generate videos` 没有触发 video generation 任务约束；
3. `image and video description` 没有稳定触发图像/视频描述任务约束；
4. `rank search results by LLM` 没有稳定触发 LLM reranking/search ranking 约束；
5. 候选缺少中等约束时仍可能被标为 `pass`，现在改为 `soft_pass`。

## 覆盖文件

```text
src/scholarpath/guard/constraint_guard.py
tests/test_constraint_guard.py
```

## 覆盖后运行

```powershell
pytest -q
```

然后重新跑：

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced_v2 --guard-mode balanced
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision_v2 --guard-mode precision
```

查看：

```powershell
Get-Content -Encoding UTF8 outputs\b5_1_guard_balanced_v2\guard_reason_samples.md
Get-Content -Encoding UTF8 outputs\b5_1_guard_precision_v2\guard_reason_samples.md
```
