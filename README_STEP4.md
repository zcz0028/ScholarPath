# ScholarPath 第1周第4步：OpenAlex API与B0最低基线

## 当前目标

只实现：

```text
RealScholarQuery原始question
→ OpenAlex全文搜索
→ 时间截止过滤
→ Top20/Top50/Top100
→ 统一PaperRecord
→ strict与pasa_title评测
→ API、Token、成本和延迟统计
```

本步骤不进行查询改写、约束分解、Reranker、Selector、引文扩展、LLM调用或前端开发。

## 为什么选择OpenAlex

- 官方REST API；
- 可按标题、摘要和全文搜索；
- 可用`to_publication_date`控制时间泄漏；
- 返回DOI、OpenAlex ID、作者、年份、venue等统一实体字段；
- 免费API Key每天有免费额度；
- B0每条查询只调用一次，50条查询首次完整运行预计50次搜索调用。

## API Key

不要把Key写进代码、README或Git。

在OpenAlex账户设置中创建免费API Key后，只在当前PowerShell终端设置：

```powershell
$env:OPENALEX_API_KEY="你的Key"
```

检查：

```powershell
if ($env:OPENALEX_API_KEY) { "Key已设置" } else { "Key未设置" }
```

不要把Key截图或发给他人。

## 安装

```powershell
python -m pip install -r requirements.txt
pytest -q
```

预期33项测试通过。

## 连接测试

```powershell
python scripts/check_openalex.py
```

预期：HTTP状态200，返回3篇论文。

## 先运行1条查询

```powershell
python scripts/run_b0_openalex.py --limit-queries 1 --output-dir outputs/b0_openalex_smoke
```

此模式只验证真实API流程，不执行完整F1，因为预测qid不完整。

## 完整运行50条查询

确保第2步已经生成：

```text
data/processed/realscholarquery_gold.jsonl
```

运行：

```powershell
python scripts/run_b0_openalex.py --output-dir outputs/b0_openalex
```

首次运行预计：

- 查询数：50；
- 实际API搜索调用：50（无重试时）；
- Token：0；
- Top20、Top50、Top100各生成一份预测；
- 自动生成strict和pasa_title两种评测结果。

第二次使用相同参数运行时会命中缓存，预计实际API调用为0、缓存命中为50。

## 输出

```text
outputs/b0_openalex/
├── raw_predictions_top100.jsonl
├── predictions_top20.jsonl
├── predictions_top50.jsonl
├── predictions_top100.jsonl
├── query_logs.jsonl
├── run_summary.json
└── evaluation/
    ├── top20/
    │   ├── strict/
    │   └── pasa_title/
    ├── top50/
    └── top100/
```

## B0定义

- 查询：原始`question`，不改写；
- 检索：OpenAlex `search`；
- 排序：OpenAlex默认相关性顺序；
- 时间截止：`published_time - 7天`，与PaSa公开运行代码一致；
- 每条查询：一次搜索API调用；
- LLM：不使用，因此Token必须为0；
- 候选：最多100条；
- 评测：分别计算Top20、Top50和Top100。

## 成本口径

代码默认按照每1000次搜索调用1美元估算，并明确标记为估算值。首次50条、无重试时约为0.05美元；OpenAlex免费Key当前每天提供免费额度，因此实际付费通常为0，但实验表仍记录估算API成本，便于后续策略比较。

## 验收标准

1. `pytest -q`全部通过；
2. `check_openalex.py`返回HTTP 200；
3. 单查询真实检索成功；
4. 完整运行包含50个qid；
5. 无重试时首次实际API调用等于50；
6. Token总数等于0；
7. 三个Top-K预测文件全部生成；
8. strict和pasa_title共6组评测结果生成；
9. `query_logs.jsonl`逐查询记录延迟、缓存、调用次数和错误；
10. `run_summary.json`记录平均延迟、P50、P95、总成本和F1。

## 常见错误

### OPENALEX_API_KEY is not set

当前PowerShell窗口没有设置Key：

```powershell
$env:OPENALEX_API_KEY="你的Key"
```

### 401或403

Key无效、复制时带空格，或账户/API权限异常。重新复制Key并只在当前终端设置。

### 429

触发限流。程序会读取`Retry-After`并指数退避重试。不要同时启动多个完整B0进程。

### 首次运行API调用不是50

可能原因：

- 之前已有缓存；
- 某些请求重试；
- 某些查询失败；
- 使用了`--limit-queries`。

需要完全重跑时使用新缓存目录，不建议频繁使用`--refresh-cache`浪费额度。

### F1很低

这是B0预期现象。复杂自然语言查询直接送入关键词搜索，通常Precision和Recall都有限。B0的作用是建立后续B1—B7的真实最低参照，不应为了提高当前分数提前加入查询改写或Selector。

## 是否进入下一步

只有完整50条B0运行成功、6组评测结果和成本日志齐全后，才允许进入B1“查询关键词提取与查询改写”。
