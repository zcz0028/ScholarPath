# ScholarPath Week 2 Day 2 人工复核

## 1. 工作目标

Day 2 对 ScholarPath 的论文身份解析、匹配和去重逻辑进行了系统检查，重点排查 DOI、URL、arXiv ID、OpenAlex ID、预印本与正式出版版本差异是否会造成伪 FN 或重复论文。

## 2. 实现内容

本次新增 strict_v2 评测模式，并保留原 strict 模式不变，以保证第一周实验结果可复现。

strict_v2 新增支持：

- DOI 字符串与 DOI URL 统一；
- arXiv 版本号归一化；
- 10.48550/arXiv DOI 与 arXiv ID 对齐；
- arXiv abs、pdf、export 和 ar5iv URL 解析；
- OpenAlex URL 与 W ID 统一；
- 预印本与正式出版版本识别；
- 标题、第一作者和年份辅助去重；
- 近似标题候选仅进入人工审计，不直接计为 TP。

## 3. 自动审计结果

- 查询数：50
- Legacy strict TP：61
- Strict v2 TP：61
- TP 变化：0
- Legacy 去重数：2
- Strict v2 去重数：2
- 新增自动匹配：0
- 疑似版本匹配候选：2
- API 调用数：0

## 4. 结果分析

strict_v2 与原 strict 模式得到相同的 TP 和去重数量，说明当前 B4 Top100 结果中没有发现由 DOI、arXiv、URL 或 OpenAlex ID 归一化不足造成的明显伪 FN。

因此，Day 1 识别出的 24 条 retrieval-miss 查询可以基本判定为真实候选召回失败，而不是评测匹配规则导致的假 zero-recall。

当前发现的 2 条近似版本候选只进入人工审计，不直接修改正式 TP，从而避免因放宽标题匹配规则引入误匹配。

## 5. Day 2 结论

当前第一周 strict 评测结果具有较好的稳定性。第二周后续工作不应继续围绕放宽匹配规则提高指标，而应把重点转向学术锚点提取和定向救援召回。

Day 3 将基于 Day 1 的 24 条 retrieval-miss 查询，开发 Anchor-aware Query Planner，重点生成数据集名、基准名、方法名、任务名、模型名和论文标题锚点驱动的检索式。

[x] pytest 显示 110 passed
[x] Legacy strict TP = 61
[x] Strict v2 TP = 61
[x] TP delta = 0
[x] Legacy duplicates removed = 2
[x] V2 duplicates removed = 2
[x] Possible version matches = 2
[x] API calls = 0
[x] strict 第一周行为未被改变
[x] 近似标题没有直接计为 TP