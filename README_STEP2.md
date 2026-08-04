# ScholarPath 第1周第2步：论文统一标识与去重机制

## 当前目的

把OpenAlex、Semantic Scholar、arXiv及其他来源的论文记录转换为统一 `PaperRecord`，并以保守、可审计的规则完成实体对齐和去重。

统一标识优先级：

```text
DOI → arXiv ID → OpenAlex ID → Semantic Scholar ID → 标准化标题指纹
```

RealScholarQuery对齐优先级：

```text
arXiv ID精确匹配 → 去版本号匹配 → 标准化标题精确匹配
```

模糊标题只进入人工复核，不自动合并。

## 新增文件

```text
src/scholarpath/paper/
├── __init__.py
├── schema.py
├── normalizers.py
├── resolver.py
└── gold.py
scripts/
├── build_gold_records.py
└── resolve_papers.py
tests/
├── test_paper_normalizers.py
├── test_paper_resolver.py
└── test_gold_builder.py
data/example/papers_with_duplicates.jsonl
```

## 运行测试

```powershell
pytest -q
```

预期：原4项测试加新增10项测试，共 `14 passed`。

## 构建RealScholarQuery统一标准答案

先把你当前项目中的真实数据保留在：

```text
data/raw/RealScholarQuery/test.jsonl
```

运行：

```powershell
python scripts/build_gold_records.py --input data/raw/RealScholarQuery/test.jsonl --output data/processed/realscholarquery_gold.jsonl --report outputs/step2_gold_report.json
```

预期核心结果：

```text
Queries: 50
Gold papers: 791
Length mismatches: 0
```

## 运行去重示例

```powershell
python scripts/resolve_papers.py --input data/example/papers_with_duplicates.jsonl --output-dir outputs/step2_resolver_sample
```

输出：

```text
resolved_papers.jsonl
﹣ duplicate_groups.jsonl
﹣ review_candidates.jsonl
﹣ resolver_report.json
```

预期示例结果：

```text
Input records: 5
Resolved entities: 3
Removed duplicates: 2
Exact-ID merges: 1
Title/author/year merges: 1
Review candidate pairs: 1
```

## 输入格式

推荐统一格式：

```json
{
  "source": "openalex",
  "source_record_id": "W123456",
  "title": "Example Paper",
  "authors": ["Alice Smith", "Bob Lee"],
  "year": 2024,
  "doi": "https://doi.org/10.1000/example",
  "arxiv_id": "2401.01234v2",
  "openalex_id": "https://openalex.org/W123456",
  "semantic_scholar_id": null,
  "venue": "Example Conference",
  "url": "https://example.org/paper",
  "abstract": "..."
}
```

代码也兼容常见OpenAlex和Semantic Scholar字段，如 `display_name`、`publication_year`、`paperId`、`externalIds`、`authorships`。

## 验收标准

1. DOI URL和纯DOI归一化一致；
2. arXiv版本号被正确去除；
3. OpenAlex ID统一为 `W数字`；
4. 强标识相同的记录合并；
5. arXiv与正式版本在标题、第一作者和年份证据充分时合并；
6. 冲突DOI/arXiv ID不自动合并；
7. 模糊标题只进入复核文件；
8. 50条查询、791条标准答案成功转换；
9. 所有合并组保留来源、成员和理由；
10. 所有测试通过。

满足后才允许进入第3步“Precision、Recall和F1评测脚本”。
