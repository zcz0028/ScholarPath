# ScholarPath Week 2 Day 6：比赛级 FastAPI 后端

## 目标

Day 6 不重写算法内核，而是把已经冻结的 ScholarPath 检索、解释和实验产物封装为稳定服务。

后端提供两类运行模式：

- `benchmark`：读取已评测的 Day-4 Top-K，确保答辩稳定、与离线指标一致；
- `live`：对新查询执行 Academic Query Planner → OpenAlex → 去重 → lightweight semantic rerank。

Day-5 Citation Expansion 当前只作为候选发现和引用路径解释，不冒充已经评测过的最终排名。

## API

- `GET /health`
- `GET /api/system`
- `GET /api/queries`
- `POST /api/search`
- `GET /api/queries/{qid}/diagnosis`
- `GET /api/experiments`

## 安全边界

1. API 查询列表只从 RealScholarQuery 原始数据读取 `qid`、`question`、`source_meta`，绝不向服务层暴露 `answer` / `answer_arxiv_id`。
2. Benchmark Search 只读取冻结预测，不读取 Gold。
3. Live 模式不回退到 Benchmark 结果冒充实时搜索。
4. Day-5 citation path 只作为解释证据；最终排序仍使用已评测的 Day-4 Top-K。
5. 所有异常返回结构化错误，不返回 Python traceback。

## 启动

```cmd
python -m pip install -r requirements-day6.txt
pytest -q
python -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

## Benchmark Search 示例

```json
{
  "query": "Papers that explore using large language models for mining factors in stock exchange analysis.",
  "qid": "RealScholarQuery_48",
  "mode": "benchmark",
  "top_k": 20,
  "enable_citation": true
}
```

## Live Search 示例

```json
{
  "query": "recent papers about vision-language agents for game playing",
  "mode": "live",
  "top_k": 20,
  "enable_citation": false
}
```

Live 模式需要当前终端存在 `OPENALEX_API_KEY`。
