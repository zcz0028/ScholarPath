from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DIAGNOSTICS = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day1_diagnostics"
    / "query_stage_diagnostics.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            records.append(item)

    return records


def print_stage_information(
    stage_name: str,
    stage: dict[str, Any],
) -> None:
    candidate_count = stage.get("candidate_count", 0)
    matched_count = stage.get("matched_gold_count", 0)
    best_rank = stage.get("best_matched_rank")
    matched_ranks = stage.get("matched_ranks", [])
    matched_titles = stage.get("matched_gold_titles", [])
    match_types = stage.get("match_types", [])

    print(f"  [{stage_name}]")
    print(f"    candidate_count: {candidate_count}")
    print(f"    matched_gold_count: {matched_count}")
    print(f"    best_matched_rank: {best_rank}")
    print(f"    matched_ranks: {matched_ranks}")

    if matched_titles:
        print("    matched_gold_titles:")

        for title in matched_titles:
            print(f"      - {title}")

    if match_types:
        print(f"    match_types: {match_types}")


def print_query(item: dict[str, Any]) -> None:
    print("=" * 100)
    print(f"QID: {item.get('qid')}")
    print(f"Failure type: {item.get('failure_type')}")
    print(f"Priority: {item.get('priority')}")
    print(f"Gold count: {item.get('gold_count')}")
    print()
    print("Question:")
    print(item.get("question", ""))
    print()

    anchors = item.get("anchors") or item.get("extracted_anchors")

    if anchors:
        print("Anchors:")
        print(
            json.dumps(
                anchors,
                ensure_ascii=False,
                indent=2,
            )
        )
        print()

    stages = item.get("stages", {})

    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue

            stage_name = str(
                stage.get("stage_name", "unknown_stage")
            )
            print_stage_information(stage_name, stage)

    elif isinstance(stages, dict):
        preferred_order = [
            "b0_top100",
            "b1_2_top100",
            "b2_top100",
            "b4_pool",
            "b4_top20",
            "b4_top50",
            "b4_top100",
            "b5_1_top20",
            "b5_1_top50",
            "b5_1_top100",
            "b5_precision",
            "b5_2_top20",
            "b5_2_top50",
            "b5_2_top100",
        ]

        printed_names: set[str] = set()

        for stage_name in preferred_order:
            stage = stages.get(stage_name)

            if isinstance(stage, dict):
                print_stage_information(stage_name, stage)
                printed_names.add(stage_name)

        for stage_name, stage in stages.items():
            if (
                stage_name not in printed_names
                and isinstance(stage, dict)
            ):
                print_stage_information(stage_name, stage)

    suggested_actions = item.get("suggested_actions", [])

    if suggested_actions:
        print()
        print("Suggested actions:")

        for action in suggested_actions:
            print(f"  - {action}")

    print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ScholarPath Day-1 query-level "
            "diagnostic results."
        )
    )

    parser.add_argument(
        "qids",
        nargs="*",
        help=(
            "QIDs to inspect. When omitted, all non-success "
            "queries are displayed."
        ),
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_DIAGNOSTICS),
        help="Path to query_stage_diagnostics.jsonl.",
    )

    parser.add_argument(
        "--failure-type",
        help=(
            "Only display this failure type, such as "
            "ranking_loss_top100 or retrieval_miss."
        ),
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    input_path = Path(arguments.input).resolve()

    if not input_path.is_file():
        print(f"Diagnostics file not found: {input_path}")
        return 1

    records = load_jsonl(input_path)
    requested_qids = set(arguments.qids)

    selected: list[dict[str, Any]] = []

    for item in records:
        qid = str(item.get("qid", ""))
        failure_type = str(item.get("failure_type", ""))

        if requested_qids and qid not in requested_qids:
            continue

        if (
            arguments.failure_type
            and failure_type != arguments.failure_type
        ):
            continue

        if (
            not requested_qids
            and not arguments.failure_type
            and failure_type == "success"
        ):
            continue

        selected.append(item)

    if not selected:
        print("No matching diagnostic records found.")
        return 1

    for item in selected:
        print_query(item)

    print(f"Displayed queries: {len(selected)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())