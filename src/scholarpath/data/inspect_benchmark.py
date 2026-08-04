from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

ARXIV_RE = re.compile(
    r"^(?:arxiv:)?(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7}))(?:v\d+)?$",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)

QUERY_FIELD_CANDIDATES = ("question", "query", "user_query", "search_query")
ANSWER_FIELD_CANDIDATES = ("answer", "answers", "relevant_papers", "gold_papers")
DATE_PATH_CANDIDATES = (
    ("source_meta", "published_time"),
    ("query_date",),
    ("date",),
    ("published_time",),
)


@dataclass
class Issue:
    line_no: int
    level: str
    code: str
    message: str


@dataclass
class Profile:
    input_path: str
    total_records: int = 0
    valid_json_records: int = 0
    invalid_json_records: int = 0
    query_field: str | None = None
    answer_field: str | None = None
    date_path: list[str] | None = None
    top_level_fields: Counter[str] = field(default_factory=Counter)
    field_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    answer_item_types: Counter[str] = field(default_factory=Counter)
    answer_identifier_kinds: Counter[str] = field(default_factory=Counter)
    answer_counts: list[int] = field(default_factory=list)
    empty_query_count: int = 0
    empty_answer_count: int = 0
    duplicate_query_count: int = 0
    date_parse_success: int = 0
    date_parse_failure: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add_issue(self, line_no: int, level: str, code: str, message: str) -> None:
        self.issues.append(Issue(line_no, level, code, message))


def json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"{exc.msg} at column {exc.colno}"


def get_nested(record: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def choose_present_field(records: list[dict[str, Any]], candidates: Iterable[str]) -> str | None:
    scores = Counter()
    for record in records:
        for field_name in candidates:
            if field_name in record:
                scores[field_name] += 1
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def choose_present_path(
    records: list[dict[str, Any]], candidates: Iterable[tuple[str, ...]]
) -> tuple[str, ...] | None:
    scores: Counter[tuple[str, ...]] = Counter()
    for record in records:
        for path in candidates:
            exists, _ = get_nested(record, path)
            if exists:
                scores[path] += 1
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def classify_answer_item(item: Any) -> str:
    if isinstance(item, str):
        text = item.strip()
        if ARXIV_RE.fullmatch(text):
            return "arxiv_id"
        if DOI_RE.fullmatch(text):
            return "doi"
        if text.startswith(("http://", "https://")):
            return "url"
        return "title_string"
    if isinstance(item, dict):
        lowered = {str(key).lower() for key in item}
        if "doi" in lowered:
            return "dict_with_doi"
        if "arxiv_id" in lowered or "arxiv" in lowered:
            return "dict_with_arxiv"
        if "title" in lowered:
            return "dict_with_title"
        if "id" in lowered:
            return "dict_with_id"
        return "dict_other"
    return f"other_{json_type_name(item)}"


def parse_supported_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            pass
    return False


def percentile_nearest_rank(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def inspect_jsonl(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    profile = Profile(input_path=str(path))
    valid_records: list[tuple[int, dict[str, Any]]] = []

    for line_no, obj, error in iter_jsonl(path):
        profile.total_records += 1
        if error is not None:
            profile.invalid_json_records += 1
            profile.add_issue(line_no, "error", "invalid_json", error)
            continue
        if not isinstance(obj, dict):
            profile.invalid_json_records += 1
            profile.add_issue(
                line_no,
                "error",
                "record_not_object",
                f"Expected JSON object, got {json_type_name(obj)}",
            )
            continue

        profile.valid_json_records += 1
        valid_records.append((line_no, obj))
        for key, value in obj.items():
            profile.top_level_fields[key] += 1
            profile.field_types[key][json_type_name(value)] += 1

    records_only = [record for _, record in valid_records]
    profile.query_field = choose_present_field(records_only, QUERY_FIELD_CANDIDATES)
    profile.answer_field = choose_present_field(records_only, ANSWER_FIELD_CANDIDATES)
    selected_date_path = choose_present_path(records_only, DATE_PATH_CANDIDATES)
    profile.date_path = list(selected_date_path) if selected_date_path else None

    if profile.query_field is None:
        profile.add_issue(0, "error", "query_field_missing", "No supported query field found.")
    if profile.answer_field is None:
        profile.add_issue(0, "error", "answer_field_missing", "No supported answer field found.")
    if selected_date_path is None:
        profile.add_issue(0, "warning", "date_field_missing", "No supported query-date field found.")

    seen_queries: Counter[str] = Counter()

    for line_no, record in valid_records:
        if profile.query_field is not None:
            query = record.get(profile.query_field)
            if not isinstance(query, str) or not query.strip():
                profile.empty_query_count += 1
                profile.add_issue(
                    line_no,
                    "error",
                    "invalid_query",
                    f"Field '{profile.query_field}' must be a non-empty string.",
                )
            else:
                seen_queries[query.strip()] += 1

        if profile.answer_field is not None:
            answers = record.get(profile.answer_field)
            if not isinstance(answers, list):
                profile.add_issue(
                    line_no,
                    "error",
                    "invalid_answers",
                    f"Field '{profile.answer_field}' must be a list.",
                )
            else:
                profile.answer_counts.append(len(answers))
                if len(answers) == 0:
                    profile.empty_answer_count += 1
                    profile.add_issue(
                        line_no,
                        "warning",
                        "empty_answers",
                        "Gold-answer list is empty.",
                    )
                for item in answers:
                    item_type = json_type_name(item)
                    profile.answer_item_types[item_type] += 1
                    profile.answer_identifier_kinds[classify_answer_item(item)] += 1

        if selected_date_path is not None:
            exists, value = get_nested(record, selected_date_path)
            if not exists:
                profile.date_parse_failure += 1
                profile.add_issue(
                    line_no,
                    "warning",
                    "date_missing",
                    f"Date path '{'.'.join(selected_date_path)}' is missing.",
                )
            elif parse_supported_date(value):
                profile.date_parse_success += 1
            else:
                profile.date_parse_failure += 1
                profile.add_issue(
                    line_no,
                    "warning",
                    "date_unparseable",
                    f"Unsupported date value: {value!r}",
                )

    profile.duplicate_query_count = sum(count - 1 for count in seen_queries.values() if count > 1)

    answer_stats = {
        "min": min(profile.answer_counts) if profile.answer_counts else None,
        "max": max(profile.answer_counts) if profile.answer_counts else None,
        "mean": statistics.fmean(profile.answer_counts) if profile.answer_counts else None,
        "median": statistics.median(profile.answer_counts) if profile.answer_counts else None,
        "p90_nearest_rank": percentile_nearest_rank(profile.answer_counts, 0.90),
    }

    issue_counts = Counter(issue.level for issue in profile.issues)

    return {
        "input_path": profile.input_path,
        "record_counts": {
            "total_nonempty_lines": profile.total_records,
            "valid_json_objects": profile.valid_json_records,
            "invalid_records": profile.invalid_json_records,
        },
        "detected_schema": {
            "query_field": profile.query_field,
            "answer_field": profile.answer_field,
            "date_path": profile.date_path,
        },
        "top_level_fields": dict(profile.top_level_fields),
        "field_types": {
            key: dict(counter) for key, counter in sorted(profile.field_types.items())
        },
        "answer_analysis": {
            "item_json_types": dict(profile.answer_item_types),
            "identifier_kinds": dict(profile.answer_identifier_kinds),
            "answer_count_statistics": answer_stats,
            "empty_answer_records": profile.empty_answer_count,
        },
        "quality": {
            "empty_or_invalid_queries": profile.empty_query_count,
            "duplicate_query_records": profile.duplicate_query_count,
            "date_parse_success": profile.date_parse_success,
            "date_parse_failure": profile.date_parse_failure,
            "issue_counts": dict(issue_counts),
        },
        "issues": [
            {
                "line_no": issue.line_no,
                "level": issue.level,
                "code": issue.code,
                "message": issue.message,
            }
            for issue in profile.issues
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "未检测到"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ".".join(value)
    return str(value)


def report_to_markdown(report: dict[str, Any]) -> str:
    schema = report["detected_schema"]
    counts = report["record_counts"]
    answers = report["answer_analysis"]
    quality = report["quality"]
    stats = answers["answer_count_statistics"]

    dominant_answer_kind = "未检测到"
    if answers["identifier_kinds"]:
        dominant_answer_kind = max(
            answers["identifier_kinds"].items(), key=lambda item: item[1]
        )[0]

    lines = [
        "# 数据集结构检查报告",
        "",
        f"- 输入文件：`{report['input_path']}`",
        f"- 非空记录行数：{counts['total_nonempty_lines']}",
        f"- 有效JSON对象：{counts['valid_json_objects']}",
        f"- 无效记录：{counts['invalid_records']}",
        "",
        "## 自动识别的核心字段",
        "",
        f"- 查询字段：`{_fmt(schema['query_field'])}`",
        f"- 标准答案字段：`{_fmt(schema['answer_field'])}`",
        f"- 查询日期路径：`{_fmt(schema['date_path'])}`",
        f"- 标准答案主要表示形式：`{dominant_answer_kind}`",
        "",
        "## 标准答案数量统计",
        "",
        f"- 最小值：{_fmt(stats['min'])}",
        f"- 最大值：{_fmt(stats['max'])}",
        f"- 平均值：{_fmt(stats['mean'])}",
        f"- 中位数：{_fmt(stats['median'])}",
        f"- P90（nearest-rank）：{_fmt(stats['p90_nearest_rank'])}",
        f"- 空答案记录：{answers['empty_answer_records']}",
        "",
        "## 数据质量",
        "",
        f"- 空或非法查询：{quality['empty_or_invalid_queries']}",
        f"- 重复查询记录：{quality['duplicate_query_records']}",
        f"- 日期解析成功：{quality['date_parse_success']}",
        f"- 日期解析失败：{quality['date_parse_failure']}",
        f"- 错误数：{quality['issue_counts'].get('error', 0)}",
        f"- 警告数：{quality['issue_counts'].get('warning', 0)}",
        "",
        "## 标准答案类型分布",
        "",
        "```json",
        json.dumps(answers["identifier_kinds"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 顶层字段覆盖",
        "",
        "```json",
        json.dumps(report["top_level_fields"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 是否允许进入下一步",
        "",
    ]

    blocking_errors = quality["issue_counts"].get("error", 0)
    if (
        blocking_errors == 0
        and schema["query_field"] is not None
        and schema["answer_field"] is not None
        and counts["valid_json_objects"] > 0
    ):
        lines.append(
            "**条件通过。** 可以进入“论文统一标识与去重机制”，但必须先人工确认本报告识别出的字段与比赛/数据说明一致。"
        )
    else:
        lines.append(
            "**不允许。** 当前存在结构性错误或核心字段未识别，必须先修复数据或补充字段映射。"
        )

    if report["issues"]:
        lines.extend(["", "## 问题明细（最多展示前100条）", ""])
        for issue in report["issues"][:100]:
            lines.append(
                f"- 第{issue['line_no']}行 [{issue['level']}] "
                f"`{issue['code']}`：{issue['message']}"
            )

    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "dataset_profile.json"
    md_path = output_path / "dataset_profile.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect ScholarPath/PaSa-style JSONL benchmark files."
    )
    parser.add_argument("--input", required=True, help="Path to a JSONL benchmark file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/step1",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code when validation errors are found.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = inspect_jsonl(args.input)
    json_path, md_path = save_report(report, args.output_dir)

    error_count = report["quality"]["issue_counts"].get("error", 0)
    print(f"[OK] JSON report: {json_path}")
    print(f"[OK] Markdown report: {md_path}")
    print(f"[INFO] Validation errors: {error_count}")

    if args.fail_on_error and error_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
