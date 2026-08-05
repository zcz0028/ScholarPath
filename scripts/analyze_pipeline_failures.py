from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


# 当前文件位置：
# ScholarPath_week1_step1/scripts/analyze_pipeline_failures.py
#
# 项目采用 src 目录结构，因此直接执行脚本时，
# 需要先将项目根目录下的 src 加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# 注意：所有 scholarpath 导入都必须放在上面的 sys.path 处理之后。
from scholarpath.evaluation.pipeline_diagnostics import (
    diagnose_pipeline,
    load_pipeline_inputs,
    write_diagnostic_outputs,
)

DEFAULT_STAGES = OrderedDict(
    [
        ("b0_top100", "outputs/b0_openalex/predictions_top100.jsonl"),
        (
            "b1_2_top100",
            "outputs/b1_2_raw_anchor_q3_p40/predictions_top100.jsonl",
        ),
        ("b2_top100", "outputs/b2_openalex/predictions_top100.jsonl"),
        (
            "b4_pool",
            "outputs/b4_fusion_semantic/fused_candidates_before_rerank.jsonl",
        ),
        ("b4_top20", "outputs/b4_fusion_semantic/predictions_top20.jsonl"),
        ("b4_top50", "outputs/b4_fusion_semantic/predictions_top50.jsonl"),
        ("b4_top100", "outputs/b4_fusion_semantic/predictions_top100.jsonl"),
        (
            "b5_1_top20",
            "outputs/b5_1_guard_balanced_v4/predictions_top20.jsonl",
        ),
        (
            "b5_1_top50",
            "outputs/b5_1_guard_balanced_v4/predictions_top50.jsonl",
        ),
        (
            "b5_1_top100",
            "outputs/b5_1_guard_balanced_v4/predictions_top100.jsonl",
        ),
        (
            "b5_precision",
            "outputs/b5_on_b4_precision/predictions_top100.jsonl",
        ),
        (
            "b5_2_top20",
            "outputs/b5_2_guard_aware_balanced/predictions_top20.jsonl",
        ),
        (
            "b5_2_top50",
            "outputs/b5_2_guard_aware_balanced/predictions_top50.jsonl",
        ),
        (
            "b5_2_top100",
            "outputs/b5_2_guard_aware_balanced/predictions_top100.jsonl",
        ),
    ]
)


def project_root() -> Path:
    return PROJECT_ROOT


def parse_stage(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--stage must use NAME=PATH format, for example "
            "b4_top100=outputs/b4_fusion_semantic/predictions_top100.jsonl"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("Stage name and path must be non-empty.")
    return name, path


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate query-level ScholarPath pipeline diagnostics."
    )
    parser.add_argument(
        "--gold",
        default="data/processed/realscholarquery_gold.jsonl",
        help="Gold JSONL path relative to the project root.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        type=parse_stage,
        default=None,
        metavar="NAME=PATH",
        help=(
            "Stage prediction JSONL. Repeat this option to override the default "
            "stage list. The option order becomes the pipeline order."
        ),
    )
    parser.add_argument(
        "--target-stage",
        default="b4_top100",
        help="Stage used to define zero-recall queries. Default: b4_top100.",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "pasa_title"),
        default="strict",
        help="Formal matching mode. Default: strict.",
    )
    parser.add_argument(
        "--expected-query-count",
        type=int,
        default=50,
        help="Expected number of unique qids. Use 0 to disable this check.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/week2_day1_diagnostics",
        help="Output directory relative to the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    stage_specs = OrderedDict(args.stage or DEFAULT_STAGES.items())

    if args.target_stage not in stage_specs:
        print(
            f"Error: target stage '{args.target_stage}' is not in stage list: "
            + ", ".join(stage_specs),
            file=sys.stderr,
        )
        return 2

    gold_path = resolve_path(root, args.gold)
    stage_paths = OrderedDict(
        (name, resolve_path(root, path)) for name, path in stage_specs.items()
    )
    output_dir = resolve_path(root, args.output_dir)

    missing = [str(path) for path in [gold_path, *stage_paths.values()] if not path.is_file()]
    if missing:
        print("Error: missing input files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    try:
        gold_records, stage_records = load_pipeline_inputs(gold_path, stage_paths)
        diagnoses, summary, matching_issues = diagnose_pipeline(
            gold_records,
            stage_records,
            target_stage=args.target_stage,
            mode=args.mode,
            expected_query_count=(
                args.expected_query_count if args.expected_query_count > 0 else None
            ),
        )
        write_diagnostic_outputs(
            output_dir,
            diagnoses,
            summary,
            matching_issues,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Diagnostics written to: {output_dir}")
    print(f"Query count: {summary.query_count}")
    print(f"Target stage: {summary.target_stage}")
    print(f"Target total TP: {summary.target_total_tp}")
    print(f"Target zero-recall queries: {summary.target_zero_recall_count}")
    print("Failure types:")
    for name, count in sorted(summary.failure_type_counts.items()):
        print(f"  {name}: {count}")
    print(f"Possible matching issues: {summary.possible_matching_issue_count}")
    print("API calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
