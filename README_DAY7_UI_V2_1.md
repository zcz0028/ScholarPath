# ScholarPath Day7 UI Patch V2.1

本补丁基于 Day7 UI V2，只做比赛展示页视觉收敛，不改 API、不新增业务功能、不添加假数据。

## V2.1 调整
- 搜索后的 Research Workspace 扩展到约 94–95vw，明显减少左右留白。
- Results / Paper Details 调整为约 70:30 的宽屏比例。
- AI 检索思路展开态压缩高度，使首屏可以同时看到推理、论文与详情。
- 真实 reason tags 做 UI 可读映射；底层 API 值不变。
- Retrieval Sources 做可读映射，例如 `day4_rescue` → `Day4 Rescue Retrieval`。
- 论文列表、详情区、Pipeline 的间距和高度进一步压缩。
- 搜索前极简首页保持 V2 设计，不改变。
- 中文/EN 仍只切换 UI 文案；Query、论文标题、作者、Venue 等保持原文。

## 覆盖
从项目根目录覆盖 `apps/web` 中的同名文件即可。

## 验收
```cmd
cd apps\web
npm run build
npm run dev
```

预期：生产构建通过；搜索前页面与 V2 基本一致，搜索后页面宽度明显增加。
