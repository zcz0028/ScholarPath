# ScholarPath Week 2 Day 7 — Competition Frontend

本补丁实现 ScholarPath 比赛级前端主页面。视觉以已确认的两张高保真设计图为基线，但所有运行数据均来自 Day 6 FastAPI；没有 Mock 论文、固定指标或虚构 Citation Path。

## 技术栈

- React 18
- TypeScript
- Vite 5（兼容 Node 18）
- Lucide React
- 原生 CSS（避免为了样式引入额外 UI 框架）

## 安装

```cmd
cd apps\web
npm install
```

## 启动后端

在项目根目录：

```cmd
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

## 启动前端

另开一个终端：

```cmd
cd apps\web
npm run dev
```

浏览器打开：

```text
http://127.0.0.1:5173
```

## 功能边界

已实现：

- Benchmark / Live 模式切换
- Benchmark 查询来自 `GET /api/queries`
- Search 来自 `POST /api/search`
- 中文 / EN UI 切换
- AI 检索思路折叠 / 展开
- `parsed_constraints` / `query_plan` / `academic_anchors` / pipeline 真实展示
- Top20 论文真实列表
- 论文详情空状态与选中状态
- Why Recommended / Match Evidence / Citation Path / Retrieval Sources
- OpenAlex / DOI / arXiv / Venue / Year 等真实 metadata
- 查看原文
- Citation Path 存在时才启用“查看引用路径”
- Pipeline 使用真实 `pipeline.stages`
- Header 指标使用真实 `cost` 和 `latency_ms`

明确不实现：

- 论文对比
- Mock 数据
- 前端自行计算 Match 0.92 等不存在于 API 的指标
- 自动翻译论文标题、Query、作者或 Venue
- 未被 API 返回的 Citation Graph / citation count

## Day 7 验收

```cmd
npm run build
```

随后人工检查 Benchmark `RealScholarQuery_48` 与一个 Live 查询。
