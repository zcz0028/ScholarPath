from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from scholarpath.evaluation.pipeline_diagnostics import (
    diagnose_pipeline,
    load_pipeline_inputs,
)


DEFAULT_GOLD = "data/processed/realscholarquery_gold.jsonl"

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
        (
            "b4_top100",
            "outputs/b4_fusion_semantic/predictions_top100.jsonl",
        ),
    ]
)

DEFAULT_DAY4_PREDICTIONS = (
    "outputs/week2_day4_rescue/predictions_top100.jsonl"
)
DEFAULT_RECOVERED = "outputs/week2_day4_rescue/recovered_queries.jsonl"
DEFAULT_UNRECOVERED = "outputs/week2_day4_rescue/unrecovered_queries.jsonl"

DEFAULT_DAY8_SUMMARY = (
    "outputs/week2_day8_evidence_rerank/ablation_summary.json"
)
DEFAULT_DAY9_COVERAGE = (
    "outputs/week2_day9_citation/coverage_summary.json"
)
DEFAULT_DAY11_SUMMARY = (
    "outputs/week2_day11_evaluation/evaluation_summary.json"
)

DEFAULT_OUTPUT_DIR = "outputs/week2_day12_failure_analysis"


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def failure_family(failure_type: str) -> str:
    if failure_type == "success":
        return "success"
    if failure_type == "retrieval_miss":
        return "retrieval"
    if failure_type == "fusion_pool_miss":
        return "fusion"
    if failure_type.startswith("ranking_loss"):
        return "ranking"
    if failure_type == "possible_matching_issue":
        return "matching_audit"
    if failure_type in {
        "selector_drop",
        "guard_drop",
        "guard_aware_drop",
    }:
        return "post_ranking_filter"
    return "other"


def load_day4_recovery(
    recovered_path: Path,
    unrecovered_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovered = read_jsonl(recovered_path)
    unrecovered = read_jsonl(unrecovered_path)
    return recovered, unrecovered


def build_failure_summary(
    diagnoses: list[Any],
    diagnostic_summary: Any,
) -> dict[str, Any]:
    type_counts = Counter(
        diagnosis.failure_type for diagnosis in diagnoses
    )
    family_counts = Counter(
        failure_family(diagnosis.failure_type)
        for diagnosis in diagnoses
    )
    priority_counts = Counter(
        diagnosis.priority for diagnosis in diagnoses
    )

    failed = [
        diagnosis
        for diagnosis in diagnoses
        if diagnosis.failure_type != "success"
    ]

    return {
        "schema_version": "day12.failure.v1",
        "taxonomy_source": (
            "scholarpath.evaluation.pipeline_diagnostics.classify_failure"
        ),
        "target_stage": diagnostic_summary.target_stage,
        "matching_mode": diagnostic_summary.matching_mode,
        "query_count": diagnostic_summary.query_count,
        "target_query_hit_count": diagnostic_summary.target_query_hit_count,
        "target_zero_recall_count": (
            diagnostic_summary.target_zero_recall_count
        ),
        "query_success_rate": safe_rate(
            diagnostic_summary.target_query_hit_count,
            diagnostic_summary.query_count,
        ),
        "failure_query_count": len(failed),
        "failure_type_counts": dict(sorted(type_counts.items())),
        "failure_family_counts": dict(sorted(family_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "possible_matching_issue_count": (
            diagnostic_summary.possible_matching_issue_count
        ),
        "stage_query_hit_counts": (
            diagnostic_summary.stage_query_hit_counts
        ),
        "stage_total_tp": diagnostic_summary.stage_total_tp,
        "stage_candidate_counts": (
            diagnostic_summary.stage_candidate_counts
        ),
        "taxonomy_note": (
            "Failure types are reused without modification from the "
            "existing pipeline diagnostics implementation."
        ),
    }


def build_recovery_summary(
    recovered: list[dict[str, Any]],
    unrecovered: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_zero_recall = len(recovered) + len(unrecovered)

    recovered_tp_gain = sum(
        max(0, int(row.get("tp_delta", 0) or 0))
        for row in recovered
    )

    current_tp_recovered = sum(
        max(0, int(row.get("current_tp", 0) or 0))
        for row in recovered
    )

    remaining_false_negatives = sum(
        len(row.get("remaining_false_negatives") or [])
        for row in unrecovered
    )

    return {
        "schema_version": "day12.recovery.v1",
        "baseline_zero_recall_query_count": baseline_zero_recall,
        "recovered_query_count": len(recovered),
        "unrecovered_query_count": len(unrecovered),
        "query_recovery_rate": safe_rate(
            len(recovered),
            baseline_zero_recall,
        ),
        "recovered_tp_gain": recovered_tp_gain,
        "current_tp_on_recovered_queries": current_tp_recovered,
        "remaining_false_negative_count_on_unrecovered_queries": (
            remaining_false_negatives
        ),
        "interpretation": (
            "A recovered query is a previously zero-hit query for which "
            "the Day4 rescue artifacts report current_tp > 0. "
            "An unrecovered query remains at current_tp == 0."
        ),
    }


def build_residual_failures(
    unrecovered: list[dict[str, Any]],
    diagnosis_by_qid: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in unrecovered:
        qid = str(item.get("qid", ""))
        diagnosis = diagnosis_by_qid.get(qid)

        remaining = item.get("remaining_false_negatives") or []

        row = {
            "qid": qid,
            "question": item.get("question", ""),
            "baseline_tp": int(item.get("baseline_tp", 0) or 0),
            "current_tp": int(item.get("current_tp", 0) or 0),
            "remaining_false_negative_count": len(remaining),
            "remaining_false_negatives": remaining,
        }

        if diagnosis is not None:
            row.update(
                {
                    "b4_failure_type": diagnosis.failure_type,
                    "b4_failure_family": failure_family(
                        diagnosis.failure_type
                    ),
                    "b4_failure_stage": diagnosis.failure_stage,
                    "priority": diagnosis.priority,
                    "suggested_actions": list(
                        diagnosis.suggested_actions
                    ),
                    "possible_matching_issue_count": (
                        diagnosis.possible_matching_issue_count
                    ),
                }
            )
        else:
            row.update(
                {
                    "b4_failure_type": None,
                    "b4_failure_family": None,
                    "b4_failure_stage": None,
                    "priority": None,
                    "suggested_actions": [],
                    "possible_matching_issue_count": 0,
                }
            )

        rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["remaining_false_negative_count"]),
            str(row["qid"]),
        )
    )
    return rows


def build_robustness_summary(
    *,
    day8: dict[str, Any],
    day9: dict[str, Any],
    day11: dict[str, Any],
    recovery_summary: dict[str, Any],
) -> dict[str, Any]:
    competition = day11.get("competition_metrics") or {}
    headline = competition.get("headline_metrics") or {}

    day8_baseline = day8.get("baseline") or {}
    day8_recommended = day8.get("recommended") or {}

    return {
        "schema_version": "day12.robustness.v1",
        "offline_only": True,
        "network_calls": 0,
        "llm_calls": 0,
        "ranking_modified": False,
        "retrieval_recovery": {
            "baseline_zero_recall_query_count": (
                recovery_summary["baseline_zero_recall_query_count"]
            ),
            "recovered_query_count": (
                recovery_summary["recovered_query_count"]
            ),
            "unrecovered_query_count": (
                recovery_summary["unrecovered_query_count"]
            ),
            "query_recovery_rate": (
                recovery_summary["query_recovery_rate"]
            ),
        },
        "ranking": {
            "baseline_ndcg_at_10": day8_baseline.get(
                "mean_ndcg_at_10"
            ),
            "selected_variant": day8_recommended.get("variant"),
            "selected_ndcg_at_10": day8_recommended.get(
                "mean_ndcg_at_10"
            ),
            "selected_alpha": day8_recommended.get("alpha"),
            "selected_beta": day8_recommended.get("beta"),
            "relative_ndcg_improvement": headline.get(
                "day8_relative_ndcg_improvement"
            ),
        },
        "explainability": {
            "constraint_evidence_match_rate_at_100": headline.get(
                "constraint_evidence_match_rate_at_100"
            ),
            "paper_evidence_support_rate_at_100": headline.get(
                "paper_evidence_support_rate_at_100"
            ),
            "reason_generation_rate_at_100": headline.get(
                "reason_generation_rate_at_100"
            ),
            "evidence_backed_reason_rate_at_100": headline.get(
                "evidence_backed_reason_rate_at_100"
            ),
            "abstract_availability_rate_at_100": headline.get(
                "abstract_availability_rate_at_100"
            ),
        },
        "citation": {
            "scope": "single_query_case_study",
            "query_count": day9.get("query_count"),
            "top_k": day9.get("top_k"),
            "coverage": day9.get("coverage"),
            "failed_paths": day9.get("failed_paths"),
        },
        "reporting_boundaries": [
            (
                "Day12 reuses frozen artifacts and does not alter "
                "retrieval or ranking."
            ),
            (
                "Failure taxonomy comes from the existing "
                "pipeline_diagnostics implementation."
            ),
            (
                "Evidence-backed explanation coverage must be reported "
                "together with the current 0% abstract availability."
            ),
            (
                "Day9 citation coverage is a single-query case study "
                "and must not be reported as full benchmark coverage."
            ),
            (
                "The frozen E3 ranking uses alpha=1.0 and beta=0.0; "
                "therefore no ranking gain should be attributed to "
                "evidence or citation features."
            ),
        ],
    }


def build_report(
    failure_summary: dict[str, Any],
    recovery_summary: dict[str, Any],
    residual_failures: list[dict[str, Any]],
    robustness: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.extend(
        [
            "# ScholarPath Day12 Failure & Robustness Analysis",
            "",
            "## 1. Scope",
            "",
            (
                "This report performs offline failure, recovery, and "
                "robustness analysis over frozen ScholarPath artifacts."
            ),
            "",
            "- Network calls: **0**",
            "- LLM calls: **0**",
            "- Ranking modified: **No**",
            "- Failure taxonomy redefined: **No**",
            "",
            "## 2. Pipeline Failure Taxonomy",
            "",
            (
                "The analysis reuses the existing "
                "`pipeline_diagnostics.classify_failure` taxonomy."
            ),
            "",
            "| Failure type | Queries |",
            "|---|---:|",
        ]
    )

    for name, count in failure_summary[
        "failure_type_counts"
    ].items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(
        [
            "",
            "### Failure families",
            "",
            "| Family | Queries |",
            "|---|---:|",
        ]
    )

    for name, count in failure_summary[
        "failure_family_counts"
    ].items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(
        [
            "",
            "## 3. Day4 Recovery",
            "",
            (
                f"- Baseline zero-recall queries: "
                f"**{recovery_summary['baseline_zero_recall_query_count']}**"
            ),
            (
                f"- Recovered queries: "
                f"**{recovery_summary['recovered_query_count']}**"
            ),
            (
                f"- Unrecovered queries: "
                f"**{recovery_summary['unrecovered_query_count']}**"
            ),
            (
                f"- Query recovery rate: "
                f"**{recovery_summary['query_recovery_rate']:.2%}**"
            ),
            (
                f"- Recovered TP gain: "
                f"**{recovery_summary['recovered_tp_gain']}**"
            ),
            "",
            "## 4. Residual Failure Cases",
            "",
            "| QID | Failure | Priority | Remaining FN | Question |",
            "|---|---|---|---:|---|",
        ]
    )

    for item in residual_failures:
        question = str(item.get("question", "")).replace("|", "\\|")
        if len(question) > 100:
            question = question[:97] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["qid"]),
                    str(item.get("b4_failure_type") or "n/a"),
                    str(item.get("priority") or "n/a"),
                    str(item["remaining_false_negative_count"]),
                    question,
                ]
            )
            + " |"
        )

    ranking = robustness["ranking"]
    explainability = robustness["explainability"]
    citation = robustness["citation"]

    lines.extend(
        [
            "",
            "## 5. Robustness View",
            "",
            "### Ranking",
            "",
            (
                f"- Baseline NDCG@10: "
                f"**{ranking['baseline_ndcg_at_10']}**"
            ),
            (
                f"- Selected variant: "
                f"**{ranking['selected_variant']}**"
            ),
            (
                f"- Selected NDCG@10: "
                f"**{ranking['selected_ndcg_at_10']}**"
            ),
            (
                f"- Relative NDCG improvement: "
                f"**{ranking['relative_ndcg_improvement']}**"
            ),
            "",
            "### Explainability",
            "",
            (
                f"- Constraint evidence match rate @100: "
                f"**{explainability['constraint_evidence_match_rate_at_100']}**"
            ),
            (
                f"- Paper evidence support rate @100: "
                f"**{explainability['paper_evidence_support_rate_at_100']}**"
            ),
            (
                f"- Reason generation rate @100: "
                f"**{explainability['reason_generation_rate_at_100']}**"
            ),
            (
                f"- Evidence-backed reason rate @100: "
                f"**{explainability['evidence_backed_reason_rate_at_100']}**"
            ),
            (
                f"- Abstract availability rate @100: "
                f"**{explainability['abstract_availability_rate_at_100']}**"
            ),
            "",
            "### Citation",
            "",
            (
                f"- Case-study query count: "
                f"**{citation['query_count']}**"
            ),
            f"- Top-K: **{citation['top_k']}**",
            f"- Citation-path coverage: **{citation['coverage']}**",
            f"- Failed paths: **{citation['failed_paths']}**",
            "",
            (
                "> Citation coverage is a single-query case study, "
                "not full benchmark coverage."
            ),
            "",
            "## 6. Competition-safe Interpretation",
            "",
            (
                "ScholarPath shows progressive recovery across the "
                "retrieval pipeline while retaining explicit accounting "
                "of unresolved failure cases. Ranking quality improves "
                "under the frozen Day8 evaluation, while the "
                "explainability layer exposes its current evidence-data "
                "limitations instead of hiding unsupported cases."
            ),
            "",
            (
                "The Day9 citation-path result demonstrates feasibility "
                "on a case study only. It must not be generalized to "
                "the complete 50-query benchmark."
            ),
            "",
            (
                "The frozen E3 configuration uses alpha=1.0 and "
                "beta=0.0. Consequently, the measured Day8 ranking gain "
                "must not be attributed to evidence or citation "
                "features."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ScholarPath Day12 offline failure and robustness "
            "analysis."
        )
    )
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument(
        "--day4-predictions",
        default=DEFAULT_DAY4_PREDICTIONS,
    )
    parser.add_argument("--recovered", default=DEFAULT_RECOVERED)
    parser.add_argument("--unrecovered", default=DEFAULT_UNRECOVERED)
    parser.add_argument(
        "--day8-summary",
        default=DEFAULT_DAY8_SUMMARY,
    )
    parser.add_argument(
        "--day9-coverage",
        default=DEFAULT_DAY9_COVERAGE,
    )
    parser.add_argument(
        "--day11-summary",
        default=DEFAULT_DAY11_SUMMARY,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT

    gold_path = resolve_path(root, args.gold)
    day4_predictions = resolve_path(root, args.day4_predictions)
    recovered_path = resolve_path(root, args.recovered)
    unrecovered_path = resolve_path(root, args.unrecovered)
    day8_path = resolve_path(root, args.day8_summary)
    day9_path = resolve_path(root, args.day9_coverage)
    day11_path = resolve_path(root, args.day11_summary)
    output_dir = resolve_path(root, args.output_dir)

    stage_paths = OrderedDict(
        (
            name,
            resolve_path(root, relative_path),
        )
        for name, relative_path in DEFAULT_STAGES.items()
    )

    # Day4 is intentionally appended as an observation stage. The
    # diagnostic target remains frozen B4 Top100 so that Day12 does not
    # redefine the existing failure taxonomy.
    stage_paths["day4_rescue_top100"] = day4_predictions

    required = [
        gold_path,
        *stage_paths.values(),
        recovered_path,
        unrecovered_path,
        day8_path,
        day9_path,
        day11_path,
    ]

    missing = [path for path in required if not path.is_file()]
    if missing:
        print("[Day12] Missing required inputs:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    try:
        gold_records, stage_records = load_pipeline_inputs(
            gold_path,
            stage_paths,
        )

        diagnoses, diagnostic_summary, _ = diagnose_pipeline(
            gold_records,
            stage_records,
            target_stage="b4_top100",
            mode="strict",
            expected_query_count=50,
        )

        recovered, unrecovered = load_day4_recovery(
            recovered_path,
            unrecovered_path,
        )

        failure_summary = build_failure_summary(
            diagnoses,
            diagnostic_summary,
        )
        recovery_summary = build_recovery_summary(
            recovered,
            unrecovered,
        )

        diagnosis_by_qid = {
            diagnosis.qid: diagnosis
            for diagnosis in diagnoses
        }

        residual_failures = build_residual_failures(
            unrecovered,
            diagnosis_by_qid,
        )

        day8 = read_json(day8_path)
        day9 = read_json(day9_path)
        day11 = read_json(day11_path)

        robustness = build_robustness_summary(
            day8=day8,
            day9=day9,
            day11=day11,
            recovery_summary=recovery_summary,
        )

        report = build_report(
            failure_summary,
            recovery_summary,
            residual_failures,
            robustness,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            output_dir / "failure_summary.json",
            failure_summary,
        )
        write_json(
            output_dir / "recovery_summary.json",
            recovery_summary,
        )
        write_jsonl(
            output_dir / "residual_failures.jsonl",
            residual_failures,
        )
        write_json(
            output_dir / "robustness_summary.json",
            robustness,
        )
        (
            output_dir / "failure_analysis_report.md"
        ).write_text(
            report,
            encoding="utf-8",
            newline="\n",
        )

    except (
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        print(f"[Day12] Error: {error}", file=sys.stderr)
        return 2

    print("[Day12] Failure and robustness analysis generated.")
    print(f"[Day12] output_dir = {output_dir}")
    print()
    print("=== B4 Failure Taxonomy ===")
    for name, count in failure_summary[
        "failure_type_counts"
    ].items():
        print(f"{name}: {count}")

    print()
    print("=== Day4 Recovery ===")
    print(
        "Baseline zero-recall queries:",
        recovery_summary["baseline_zero_recall_query_count"],
    )
    print(
        "Recovered queries:",
        recovery_summary["recovered_query_count"],
    )
    print(
        "Unrecovered queries:",
        recovery_summary["unrecovered_query_count"],
    )
    print(
        "Query recovery rate:",
        round(recovery_summary["query_recovery_rate"], 6),
    )
    print(
        "Recovered TP gain:",
        recovery_summary["recovered_tp_gain"],
    )

    print()
    print("=== Robustness Boundaries ===")
    print("Offline only: True")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("Ranking modified: False")
    print(
        "Citation scope:",
        robustness["citation"]["scope"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())