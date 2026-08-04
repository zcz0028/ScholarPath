# ScholarPath 第1周第1步：PaSa与参考数据分析

本代码包只完成第1步：分析PaSa/官方参考数据格式，不实现检索、Agent、前端、图片识别或论文总结。

## 环境

- Python 3.10+
- 运行脚本本身仅依赖Python标准库
- 单元测试使用pytest

## 安装

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 运行示例

```bash
python scripts/inspect_benchmark.py \
  --input data/example/pasa_like_sample.jsonl \
  --output-dir outputs/step1_sample
```

真实PaSa数据下载并放置为：

```text
data/raw/RealScholarQuery/test.jsonl
```

再运行：

```bash
python scripts/inspect_benchmark.py \
  --input data/raw/RealScholarQuery/test.jsonl \
  --output-dir outputs/step1_realscholarquery \
  --fail-on-error
```

## 输出

```text
outputs/step1_realscholarquery/
├── dataset_profile.json
└── dataset_profile.md
```

## 验收命令

```bash
pytest -q
```

只有在以下条件全部满足后，才允许进入“论文统一标识与去重机制”：

1. 真实数据可以完整读取；
2. 每条记录的查询字段已确认；
3. 标准答案字段已确认；
4. 标准答案元素类型已确认；
5. 查询日期字段及格式已确认；
6. 无无法解释的结构错误；
7. 已明确PaSa复现口径与比赛官方口径不能混用。
