# Week 2 Day 7 — Competition Frontend

## 设计原则

1. 高保真视觉基线：ScholarPath 深蓝顶栏、白色卡片、蓝绿少量强调色。
2. API-first：任何论文、分数、成本、路径、Pipeline 都来自 Day 6 API。
3. Query-level 与 Paper-level 解释分离：AI 检索思路解释“系统如何理解与搜索”，右侧详情解释“为什么推荐当前论文”。
4. Benchmark 与 Live 都保留；Benchmark 用于可复现实验演示，Live 用于真实新查询。
5. UI i18n 只翻译界面文本，不改写科研 Query 和论文 metadata。

## 真实 API 映射

- `/api/queries` → Benchmark 示例与 qid
- `/api/search` → 查询理解、计划、结果、Pipeline、成本与延迟
- `parsed_constraints` → Research Understanding
- `query_plan` → Search Plans
- `academic_anchors` → Academic Anchors
- `pipeline.stages` → Retrieval Enhancements + System Pipeline
- `results[]` → Recommended Papers
- `reason_text` / `reason_tags` → Why Recommended
- `constraint_evidence` / `score` → Match Evidence
- `citation_path` → Citation Path（仅存在时展示）
- `retrieval_sources` → Retrieval Sources

## 不造数据

视觉稿中的 Match 0.92、Cache 82%、API Calls 128、Citation Count 等若 API 未提供，正式实现不显示这些占位数据。
