# ScholarPath 第一周阶段验收总结

> 项目：ScholarPath  
> 赛题：科研场景下复杂学术查询的智能论文搜索与推荐  
> 阶段：Week 1 技术基线与核心链路验证  
> 当前状态：第一周核心验收目标已完成，测试通过 `85 passed in 3.41s`

---

## 1. 第一周验收结论

第一周主要目标是建立可复现的评测体系、完成最低可运行检索基线，并验证复杂学术查询场景下的候选召回、排序、精筛和解释链路。当前已完成：

- RealScholarQuery 数据读取与标准答案处理；
- strict / pasa_title 两类评测模式；
- Top20 / Top50 / Top100 指标统计；
- OpenAlex 基础检索基线；
- 多来源候选召回与融合；
- 轻量语义重排；
- Selector 精筛与相关性理由标签；
- Constraint Guard 硬约束校验；
- Guard-aware Selector 融合精筛；
- 全部后处理模块保持 `0 API / 0 Token` 成本；
- 单元测试通过：`85 passed in 3.41s`。

因此，第一周已经从“最低基线”推进到“可解释多模式推荐原型”，完成度超过原始第一周预期。

---

## 2. 数据与评测设置

### 2.1 数据集

当前使用数据：

```text
data/raw/RealScholarQuery/test.jsonl
```

处理后标准答案：

```text
data/processed/realscholarquery_gold.jsonl
```

数据统计：

| 项目 | 数值 |
|---|---:|
| 有效查询数 | 50 |
| 标准答案论文总数 | 791 |
| 每条 query 答案数最小值 | 1 |
| 每条 query 答案数最大值 | 65 |
| 每条 query 平均答案数 | 约 15.82 |
| 每条 query 答案数中位数 | 9.5 |
| 每条 query 答案数 P90 | 37 |

### 2.2 评测指标

当前评测支持：

```text
strict
pasa_title
Top20 / Top50 / Top100
Micro Precision / Recall / F1
Macro Precision / Recall / F1
TP / FP / FN
去重前后统计
```

本阶段主要使用 `strict` 模式进行主结果比较。

---

## 3. 第一周技术链路

当前形成的主链路为：

```text
B0 / B1.2 / B2 多来源候选召回
→ B4 多来源候选融合
→ B3 轻量语义重排
→ B5.1 Constraint Guard 硬约束校验
→ B5.2 Guard-aware Selector / B5 Precision Selector
```

其中：

- B4 解决候选覆盖不足；
- B5.1 解决硬约束识别和推荐解释可信性；
- B5 Precision 解决少量高可信推荐；
- B5.2 解决“规则可信 + 精筛降噪”的均衡推荐。

---

## 4. 阶段模块归档

| 阶段 | 模块名称 | 核心作用 | 实验结论 | 最终去留 |
|---|---|---|---|---|
| B0 | OpenAlex 原始检索基线 | 建立最低可运行检索基线 | Top100 TP=44 | 保留为基础基线 |
| B1.2 | Raw-anchor 查询改写候选池 | 利用原始查询锚点扩展候选 | Top100 TP=53，高于 B0 | 保留为候选源 |
| B2 | 约束感知多子查询召回 | 基于约束拆分生成多子查询 | 可减少零召回 query，但直接输出一般 | 保留为候选源 |
| B2.1 | 约束覆盖重排 | 对候选进行约束覆盖排序 | 排序略有改善，覆盖不变 | 保留为消融 |
| B3 | 轻量语义 Reranker | 提升前排排序 | Top20 明显提升 | 作为排序基础 |
| B4 | 多来源候选融合 + 语义重排 | 融合 B0/B1.2/B2 候选 | Top100 TP=61 | 保留为融合基线 |
| B5 | Selector 精筛 | 降低 FP，输出相关性标签 | precision 模式 MacroF1 最高 | 保留为高精度精选 |
| B5.1 | Constraint Guard | 识别硬约束并输出解释 | v4 后 Top20/Top50 优于 B4 | 保留为主推荐版本 |
| B5.2 | Guard-aware Selector | 融合 Selector 与 Guard | Top50 TP=48，FP 明显下降 | 保留为可信均衡精筛 |

---

## 5. 核心实验结果

### 5.1 主推荐链路结果

| 版本 | Top20 MacroF1 | Top20 TP/FP | Top50 MacroF1 | Top50 TP/FP | Top100 MacroF1 | Top100 TP/FP | 定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.01540 | 12 / 928 | 0.02183 | 31 / 2198 | 0.01935 | 44 / 4231 | 原始检索基线 |
| B3 | 0.02438 | 21 / 948 | 0.02271 | 40 / 2352 | 0.01878 | 53 / 4440 | 语义重排基线 |
| B4 | 0.04017 | 36 / 963 | 0.02610 | 44 / 2455 | 0.02040 | 61 / 4909 | 多来源融合基线 |
| B5.1 Guard v4 | **0.04114** | **37 / 962** | **0.02667** | **45 / 2454** | 0.02040 | 61 / 4896 | 当前主推荐版本 |

B5.1 Guard v4 相比 B4：

```text
Top20 TP: 36 → 37
Top20 FP: 963 → 962

Top50 TP: 44 → 45
Top50 FP: 2455 → 2454

Top100 TP: 61 → 61
Top100 FP: 4909 → 4896
```

说明硬约束校验在不损失 Top100 召回的情况下，小幅提升了前排命中并减少误推荐。

---

### 5.2 精筛版本结果

| 版本 | Top20 MacroF1 | Top20 TP/FP | Top50 MacroF1 | Top50 TP/FP | Top100 MacroF1 | Top100 TP/FP | 定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| B5_on_B4_precision | **0.04239** | 22 / 378 | **0.04239** | 22 / 378 | **0.04239** | 22 / 378 | 最高 F1 高精度精选 |
| B5.2 ranking | 0.03715 | 33 / 966 | 0.02923 | 49 / 2450 | 0.02040 | 61 / 4896 | Top50 TP 高，但 FP 仍高 |
| B5.2 balanced | 0.03830 | 34 / 965 | 0.03359 | 48 / 1814 | 0.02636 | 57 / 3205 | 可信均衡精筛 |
| B5.2 precision | 0.03070 | 27 / 658 | 0.02622 | 41 / 1489 | 0.02622 | 41 / 1489 | 过严，不作为核心结果 |
| B5.2 precision_safe | 0.03786 | 31 / 763 | 0.03338 | 45 / 1598 | 0.03031 | 45 / 1878 | 保守高精度备选 |

B5.2 balanced 相比 B5.1 Guard v4：

```text
Top50 TP: 45 → 48
Top50 FP: 2454 → 1814
Top50 MacroF1: 0.02667 → 0.03359
```

说明 Guard-aware Selector 能在保持较多正确论文的同时显著降低 FP，适合作为可信均衡精筛版本。

---

## 6. 最终采用版本

当前建议保留三个最终输出模式。

### 6.1 主推荐版本

```text
outputs/b5_1_guard_balanced_v4
```

适用场景：

```text
用户希望获得较完整的推荐列表，并希望结果带有硬约束解释。
```

采用理由：

- Top20 / Top50 均优于 B4；
- Top100 TP 不损失；
- FP 略有下降；
- 能解释数据集、方法、任务、模态、排除条件等硬约束。

---

### 6.2 最高 F1 高精度精选版本

```text
outputs/b5_on_b4_precision
```

适用场景：

```text
用户只希望获得少量高可信论文。
```

采用理由：

- 当前 MacroF1 最高：0.04239；
- FP 最低：378；
- 输出少但精度高。

不足：

- 输出数量较少；
- zero-recall query 较多；
- 不适合作为完整主推荐列表。

---

### 6.3 可信均衡精筛版本

```text
outputs/b5_2_guard_aware_balanced
```

适用场景：

```text
用户希望兼顾推荐数量、正确覆盖、误推荐控制和解释可信性。
```

采用理由：

- Top50 TP=48；
- 相比主推荐版本显著降低 FP；
- 融合 Selector 相关性与 Guard 硬约束校验；
- 理由解释比单独 Selector 更可信。

---

## 7. 成本收益分析

### 7.1 成本情况

从 B3 到 B5.2 的后处理模块均不新增外部检索调用和 LLM 推理：

| 模块 | 新增 API 调用 | 新增 Token | 说明 |
|---|---:|---:|---|
| B3 轻量语义重排 | 0 | 0 | 基于本地特征 |
| B4 候选融合 | 0 | 0 | 复用已有候选 |
| B5 Selector | 0 | 0 | 本地规则与特征打分 |
| B5.1 Constraint Guard | 0 | 0 | 本地硬约束校验 |
| B5.2 Guard-aware Selector | 0 | 0 | 本地融合打分 |

### 7.2 收益总结

| 模块 | 成本变化 | 主要收益 |
|---|---|---|
| B4 多来源融合 | 不新增 API/Token | Top100 TP 提升至 61 |
| B5 Selector | 不新增 API/Token | 得到当前最高 MacroF1 高精度版本 |
| B5.1 Guard | 不新增 API/Token | 提升硬约束解释可信性，并小幅提升前排命中 |
| B5.2 Guard-aware Selector | 不新增 API/Token | Top50 下明显降低 FP，同时保持较高 TP |

综合来看，当前系统在不增加额外 API 与 LLM 成本的前提下，通过多来源候选融合、语义重排、硬约束校验和精筛策略，提高了复杂学术查询下的推荐质量与可解释性。

---

## 8. 可复现实验命令

### 8.1 测试

```powershell
pytest -q
```

当前结果：

```text
85 passed in 3.41s
```

### 8.2 B4 多来源候选融合

```powershell
python scripts/run_b4_candidate_fusion.py --output-dir outputs/b4_fusion_semantic
```

### 8.3 B5 Selector 高精度模式

```powershell
python scripts/run_b5_selector.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_on_b4_precision --selector-mode precision
```

### 8.4 B5.1 Constraint Guard v4

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_balanced_v4 --guard-mode balanced
```

```powershell
python scripts/run_b5_1_constraint_guard.py --candidate-dir outputs/b4_fusion_semantic --output-dir outputs/b5_1_guard_precision_v4 --guard-mode precision
```

### 8.5 B5.2 Guard-aware Selector

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_ranking --mode ranking
```

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_balanced --mode balanced
```

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_precision --mode precision
```

```powershell
python scripts/run_b5_2_guard_aware_selector.py --candidate-dir outputs/b5_1_guard_balanced_v4 --output-dir outputs/b5_2_guard_aware_precision_safe --mode precision_safe
```

### 8.6 核心结果对比命令

```powershell
python -c "import json,pathlib; runs={'B4':'outputs/b4_fusion_semantic','B5_prec':'outputs/b5_on_b4_precision','G_bal_v4':'outputs/b5_1_guard_balanced_v4','B5_2_bal':'outputs/b5_2_guard_aware_balanced'}; [print(name,'Top20=',(r20:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top20_strict'])['macro']['f1'],r20['counts'],'Top50=',(r50:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top50_strict'])['macro']['f1'],r50['counts'],'Top100=',(r100:=json.load(open(pathlib.Path(path)/'run_summary.json',encoding='utf-8'))['evaluation']['top100_strict'])['macro']['f1'],r100['counts']) for name,path in runs.items()]"
```

---

## 9. 解释样例文件

建议验收时保留以下文件：

```text
outputs/b5_1_guard_balanced_v4/guard_reason_samples.md
outputs/b5_2_guard_aware_balanced/guard_aware_reason_samples.md
outputs/b5_on_b4_precision/selector_reason_samples.md
```

对应证明：

| 文件 | 证明内容 |
|---|---|
| `guard_reason_samples.md` | Constraint Guard 能识别硬约束并解释缺失条件 |
| `guard_aware_reason_samples.md` | Guard-aware Selector 能融合相关性与硬约束 |
| `selector_reason_samples.md` | Selector 能输出相关性理由标签 |

---

## 10. 当前不足

虽然第一周目标已完成，但当前系统仍存在以下不足：

1. 仍有部分 query 处于 zero-recall 状态，候选覆盖还不够充分；
2. 当前标准匹配主要依赖标题和强标识符，后续需要加入 DOI、URL、作者年份等更稳健匹配；
3. B5 Precision 虽然 F1 最高，但输出较少，不适合作为完整推荐；
4. B5.2 能降低 FP，但权重仍为手工设置，后续可考虑数据驱动调参；
5. 当前系统主要针对文本查询，Web 页面和图片输入尚未进入主链路；
6. 前端展示与 FastAPI 服务尚未完成。

---

## 11. 第二周计划

第二周建议重点推进以下任务：

### 11.1 继续降低 zero-recall queries

重点分析：

```text
zero-recall query
late-recall query
高 FP query
```

目标是提升候选覆盖，而不是盲目扩大候选规模。

### 11.2 增强候选召回

可尝试：

- 更稳健的实体抽取；
- 数据集名 / 方法名 / 任务名优先检索；
- 受控引文扩展，只作为候选池，不直接进入最终结果；
- DOI / URL / 作者年份辅助匹配。

### 11.3 建立系统服务原型

搭建：

```text
FastAPI 后端
查询输入接口
推荐结果输出接口
理由标签输出接口
```

输出包括：

- 推荐论文列表；
- 相关性标签；
- 硬约束满足情况；
- 缺失约束说明；
- 推荐模式切换。

### 11.4 准备前端展示

前端至少支持三种模式：

```text
完整推荐
高精度精选
可信均衡精筛
```

每篇论文展示：

- 标题；
- 年份；
- 来源；
- 推荐分数；
- 相关性理由；
- 硬约束满足/缺失标签。

### 11.5 整理答辩材料

准备：

- 总体流程图；
- 模块创新点；
- 实验结果表；
- 成本收益分析；
- 典型 query 案例；
- 当前不足与后续优化方向。

---

## 12. 第一周验收结论

本周已完成从基础检索到可解释精筛推荐的完整技术闭环。系统不仅具备可运行的学术检索基线，还构建了多来源候选融合、轻量语义重排、Selector 精筛、Constraint Guard 硬约束校验和 Guard-aware Selector 等核心模块。实验结果显示，B5.1 Guard v4 在不损失 Top100 召回的情况下提升了 Top20/Top50 前排命中；B5 Precision 获得当前最高 MacroF1；B5.2 Balanced 在 Top50 下显著降低 FP 并保持较高 TP。整体上，第一周已经形成可复现、可解释、可展示的 ScholarPath 初版原型，为第二周继续提升召回覆盖和构建系统演示奠定了基础。
