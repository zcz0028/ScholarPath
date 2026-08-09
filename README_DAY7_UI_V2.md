# ScholarPath Day7 UI Patch V2

## 目标
V2 只调整前端页面状态与视觉层级，不新增后端字段、不伪造论文/指标数据。

### 已冻结交互
1. 未搜索：极简中央搜索首页。
2. 搜索后：切换到 Research Workspace。
3. Benchmark / Live 使用同一页面骨架，只改变模式语义和提示。
4. 中 / EN 只切换 UI 文案。
5. Query、论文标题、作者、Venue、DOI、OpenAlex ID 等 API/用户内容保持原文。
6. API 无数据时继续显示 `—`、空状态或隐藏，不造假。
7. 搜索后保留：AI 检索思路（可折叠）+ 论文列表 + 论文详情 + System Pipeline。

## 覆盖方法
在项目根目录（能看到 apps 文件夹的位置）解压本补丁并覆盖同名文件。

建议先备份/提交当前分支：
    git status
    git add .
    git commit -m "checkpoint before day7 ui v2"

然后覆盖补丁。

## 运行
后端：
    python -m uvicorn scholarpath.api.app:app --reload

前端：
    cd apps\web
    npm run build
    npm run dev

浏览器打开 Vite 输出的本地地址。

## 验收
- 初始页面不再显示左右两个大空白 Card。
- 搜索框位于页面中部偏上。
- Benchmark 显示真实 Benchmark 示例问题。
- Live 与 Benchmark 初始布局一致。
- 成功搜索后，搜索框上移并出现 Search Reasoning、结果、论文详情。
- 点击论文后右侧展示 API 返回的真实数据。
- 中/EN 切换后，论文标题和 Query 不发生翻译。
- 未运行时顶部 Metrics 显示 `—`。
- `npm run build` 应正常通过。

## 兼容性
V2 同时移除了组件中的 `String.prototype.replaceAll`，继续兼容当前项目 TypeScript target。
