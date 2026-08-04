# B5.1 Constraint Guard v4 最小热修复

## 修复原因

v3 中 `normalize_text()` 保留了句号等标点，导致句尾词无法匹配规则词表。

例如：

```text
generate videos.
LLM.
description.
```

会被分词为：

```text
videos.
llm.
description.
```

而规则词表中是：

```text
videos
llm
description
```

因此以下规则测试失败：

```text
task_video_generation
task_image_video_description
method_ranking_with_llm
```

## 修复内容

将归一化正则从：

```python
r"[^a-z0-9+.#]+"
```

改为：

```python
r"[^a-z0-9]+"
```

即去掉句号、#、+ 等非字母数字符号，避免句尾标点影响匹配。

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
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced_v4 --guard-mode balanced
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision_v4 --guard-mode precision
```
