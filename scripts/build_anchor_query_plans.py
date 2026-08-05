from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.query.academic_query_planner import (  # noqa: E402
    AcademicQueryPlan,
    AcademicQueryPlanner,
    PlannerConfig,
    summarize_plans,
)


DEFAULT_DAY1_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day1_diagnostics"
    / "query_stage_diagnostics.jsonl"
)
DEFAULT_GOLD_INPUT = PROJECT_ROOT / "data" / "processed" / "realscholarquery_gold.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week2_day3_query_plans"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}, line {line_number}: expected a JSON object."
                )
            output.append(value)
    return output


def load_query_records(
    path: Path,
    *,
    failure_types: set[str] | None,
) -> tuple[list[dict[str, str]], str]:
    raw_records = read_jsonl(path)
    records: list[dict[str, str]] = []
    seen_qids: set[str] = set()

    for raw in raw_records:
        qid = str(raw.get("qid") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not qid or not question:
            continue

        failure_type = str(raw.get("failure_type") or "").strip()
        if failure_types and failure_type not in failure_types:
            continue
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid in input: {qid}")

        # Deliberately retain only qid/question/failure_type. Gold papers, if the
        # fallback input happens to contain them, never enter the planner.
        records.append(
            {
                "qid": qid,
                "question": question,
                "failure_type": failure_type,
            }
        )
        seen_qids.add(qid)

    source_type = (
        "day1_diagnostics"
        if any("failure_type" in item for item in raw_records)
        else "question_source_only"
    )
    return records, source_type


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_coverage_csv(path: Path, plans: list[AcademicQueryPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "qid",
                "anchor_count",
                "anchor_types",
                "derived_alias_count",
                "planned_query_count",
                "plan_types",
                "estimated_api_calls",
                "plan_1",
                "plan_2",
            ),
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "qid": plan.qid,
                    "anchor_count": len(plan.anchors),
                    "anchor_types": ";".join(
                        sorted({anchor.anchor_type for anchor in plan.anchors})
                    ),
                    "derived_alias_count": len(plan.derived_aliases),
                    "planned_query_count": len(plan.planned_queries),
                    "plan_types": ";".join(
                        query.plan_type for query in plan.planned_queries
                    ),
                    "estimated_api_calls": sum(
                        query.estimated_api_calls for query in plan.planned_queries
                    ),
                    "plan_1": (
                        plan.planned_queries[0].text
                        if len(plan.planned_queries) >= 1
                        else ""
                    ),
                    "plan_2": (
                        plan.planned_queries[1].text
                        if len(plan.planned_queries) >= 2
                        else ""
                    ),
                }
            )


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_review_markdown(
    path: Path,
    plans: list[AcademicQueryPlan],
    summary: dict[str, Any],
    source_path: Path,
    failure_types: list[str],
) -> None:
    plan_type_counts = Counter(
        query.plan_type for plan in plans for query in plan.planned_queries
    )
    lines = [
        "# ScholarPath Week 2 Day 3：Anchor-aware Query Planner 审核报告",
        "",
        "## 1. 运行范围",
        "",
        f"- 输入文件：`{source_path.as_posix()}`",
        f"- 目标失败类型：{', '.join(failure_types) if failure_types else '全部查询'}",
        f"- 查询数量：{summary['query_count']}",
        f"- 生成定向检索式：{summary['planned_query_count']}",
        f"- Day 4 预计新增 API 调用：{summary['estimated_api_calls']}",
        "- Day 3 实际 API 调用：0",
        "- Planner 只读取 qid 和 question，不读取 Gold 论文。",
        "",
        "## 2. 计划类型统计",
        "",
    ]
    for plan_type, count in sorted(plan_type_counts.items()):
        lines.append(f"- `{plan_type}`：{count}")

    lines.extend(
        [
            "",
            "## 3. Query 级检索计划",
            "",
            "| QID | 学术锚点 | Query 1 | Query 2 | 预计调用 |",
            "|---|---|---|---|---:|",
        ]
    )
    for plan in plans:
        anchor_texts = []
        seen: set[str] = set()
        for anchor in plan.anchors:
            key = anchor.normalized_text
            if key in seen:
                continue
            anchor_texts.append(f"{anchor.anchor_type}:{anchor.text}")
            seen.add(key)
        query_1 = plan.planned_queries[0].text if plan.planned_queries else ""
        query_2 = (
            plan.planned_queries[1].text if len(plan.planned_queries) >= 2 else ""
        )
        lines.append(
            "| `{qid}` | {anchors} | `{q1}` | `{q2}` | {calls} |".format(
                qid=markdown_escape(plan.qid),
                anchors=markdown_escape("；".join(anchor_texts) or "无显式锚点"),
                q1=markdown_escape(query_1),
                q2=markdown_escape(query_2),
                calls=sum(
                    query.estimated_api_calls for query in plan.planned_queries
                ),
            )
        )

    uncovered = [plan.qid for plan in plans if not plan.planned_queries]
    lines.extend(
        [
            "",
            "## 4. Day 3 验收结论",
            "",
            f"- 无检索计划的查询：{len(uncovered)}",
            f"- 生成两条检索式的查询：{summary['queries_with_two_plans']}",
            f"- 预计 Day 4 最大新增调用：{summary['estimated_api_calls']}",
            "- 所有检索式均为确定性规则生成，可记录锚点、计划类型和生成原因。",
            "- `exclude_survey` 等负约束保存在 filters 中，不在召回阶段直接删除候选。",
            "- Day 3 不运行 OpenAlex，不产生新增检索成本。",
            "",
            "Day 4 应只执行本文件中冻结的定向检索式，并单独统计新增候选、Gold 命中、zero-recall 恢复数量和调用成本。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic anchor-aware rescue query plans."
    )
    parser.add_argument(
        "--input",
        help=(
            "Input JSONL. Defaults to Day-1 query diagnostics; if that file is "
            "missing, falls back to the processed query file and plans all queries."
        ),
    )
    parser.add_argument(
        "--failure-type",
        action="append",
        dest="failure_types",
        default=None,
        help=(
            "Only plan records with this failure type. May be repeated. "
            "Default: retrieval_miss."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=2,
        help="Maximum planned retrieval queries per benchmark query.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    failure_types = arguments.failure_types or ["retrieval_miss"]

    if arguments.input:
        input_path = Path(arguments.input)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
    elif DEFAULT_DAY1_INPUT.is_file():
        input_path = DEFAULT_DAY1_INPUT
    else:
        input_path = DEFAULT_GOLD_INPUT
        # The fallback file has no failure_type field, so plan every question.
        failure_types = []

    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records, source_type = load_query_records(
            input_path.resolve(),
            failure_types=set(failure_types) if failure_types else None,
        )
        planner = AcademicQueryPlanner(
            PlannerConfig(max_queries=arguments.max_queries)
        )
        plans = [
            planner.plan(record["qid"], record["question"])
            for record in records
        ]
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    summary = summarize_plans(plans)
    summary.update(
        {
            "input_file": input_path.resolve().as_posix(),
            "input_source_type": source_type,
            "failure_types": failure_types,
            "max_queries_per_query": arguments.max_queries,
            "planner_version": "anchor_planner_v1",
            "analysis_only_uses_gold": False,
        }
    )

    plan_dicts = [plan.to_dict() for plan in plans]
    write_json(output_dir / "planner_summary.json", summary)
    write_jsonl(output_dir / "query_plans.jsonl", plan_dicts)
    write_jsonl(
        output_dir / "uncovered_queries.jsonl",
        [plan.to_dict() for plan in plans if not plan.planned_queries],
    )
    write_coverage_csv(output_dir / "planner_coverage.csv", plans)
    write_review_markdown(
        output_dir / "day3_query_plan_review.md",
        plans,
        summary,
        input_path.resolve(),
        failure_types,
    )

    print(f"Query plans written to: {output_dir.resolve()}")
    print(f"Input source: {source_type}")
    print(f"Query count: {summary['query_count']}")
    print(f"Planned query count: {summary['planned_query_count']}")
    print(f"Queries with two plans: {summary['queries_with_two_plans']}")
    print(f"Queries with zero plans: {summary['queries_with_zero_plans']}")
    print(f"Estimated Day-4 API calls: {summary['estimated_api_calls']}")
    print("API calls executed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
