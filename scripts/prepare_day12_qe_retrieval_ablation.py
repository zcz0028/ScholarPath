from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PLANS = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day3_query_plans"
    / "query_plans.jsonl"
)
DEFAULT_RESIDUAL_FAILURES = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day12_failure_analysis"
    / "residual_failures.jsonl"
)
DEFAULT_EXPANSION_CASES = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day12_query_expansion"
    / "query_expansion_cases.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day12_query_expansion_retrieval"
    / "query_plans_qe_r9.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def build_qe_plan_record(
    source_record: dict[str, Any],
    *,
    expanded_query: str,
) -> dict[str, Any]:
    result = copy.deepcopy(source_record)
    planned = list(result.get("planned_queries") or [])
    if not planned:
        raise ValueError(
            f"Query plan has no planned_queries: {result.get('qid')}"
        )

    first = copy.deepcopy(planned[0])
    first["text"] = expanded_query

    if "plan_type" in first:
        first["plan_type"] = "day12_qe_residual"
    if "reason" in first:
        first["reason"] = (
            "Day12-3 residual-query lightweight academic expansion ablation."
        )
    if "estimated_api_calls" in first:
        first["estimated_api_calls"] = 1

    result["planned_queries"] = [first]
    result["day12_qe_ablation"] = {
        "scope": "residual_retrieval_failures_only",
        "plan_count": 1,
        "expanded_query": expanded_query,
        "production_pipeline_modified": False,
    }
    return result


def prepare_qe_plans(
    *,
    source_plans: list[dict[str, Any]],
    residual_failures: list[dict[str, Any]],
    expansion_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_qid = {
        str(row.get("qid")): row
        for row in source_plans
    }
    expansion_by_qid = {
        str(row.get("qid")): row
        for row in expansion_cases
    }

    residual_qids = [
        str(row.get("qid"))
        for row in residual_failures
        if str(row.get("b4_failure_type") or "") == "retrieval_miss"
    ]

    prepared: list[dict[str, Any]] = []
    missing: list[str] = []

    for qid in residual_qids:
        source = source_by_qid.get(qid)
        expansion = expansion_by_qid.get(qid)
        if source is None or expansion is None:
            missing.append(qid)
            continue

        expanded_query = str(
            expansion.get("expanded_query")
            or expansion.get("original_query")
            or source.get("question")
            or ""
        ).strip()
        if not expanded_query:
            missing.append(qid)
            continue

        prepared.append(
            build_qe_plan_record(
                source,
                expanded_query=expanded_query,
            )
        )

    if missing:
        raise ValueError(
            "Missing source plan or expansion for residual qids: "
            + ", ".join(sorted(missing))
        )

    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Day12-3 QE-R9 query plans by reusing the frozen "
            "Day3 plan schema and replacing only the retrieval text."
        )
    )
    parser.add_argument(
        "--source-plans",
        default=str(DEFAULT_SOURCE_PLANS),
    )
    parser.add_argument(
        "--residual-failures",
        default=str(DEFAULT_RESIDUAL_FAILURES),
    )
    parser.add_argument(
        "--expansion-cases",
        default=str(DEFAULT_EXPANSION_CASES),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_plans = read_jsonl(Path(args.source_plans))
    residual = read_jsonl(Path(args.residual_failures))
    expansions = read_jsonl(Path(args.expansion_cases))

    prepared = prepare_qe_plans(
        source_plans=source_plans,
        residual_failures=residual,
        expansion_cases=expansions,
    )

    output = Path(args.output)
    write_jsonl(output, prepared)

    print("[Day12-3-2] QE residual query plans prepared.")
    print(f"[Day12-3-2] target_queries = {len(prepared)}")
    print(f"[Day12-3-2] output = {output}")
    print("[Day12-3-2] planned API calls = one plan per target query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
