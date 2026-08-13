from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.f1_failure_decomposition import (
    build_failure_row,
    choose_priority_queries,
    derive_go_no_go,
    summarize_failure_rows,
)
from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records

DEFAULT_ORACLE = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_f1_diagnostic"
    / "oracle_k_per_query.jsonl"
)
DEFAULT_ADAPTIVE = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_adaptive_cutoff"
    / "oof_cutoff_predictions.jsonl"
)
DEFAULT_ADAPTIVE_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_adaptive_cutoff"
    / "oof_predictions.jsonl"
)
DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "realscholarquery_gold.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_f1_failure_decomposition"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}"
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def write_family_csv(
    path: Path,
    family_summaries: dict[str, Any],
) -> None:
    fields = [
        "family",
        "query_count",
        "query_share",
        "oracle_gap_sum_vs_fixed",
        "oracle_gap_macro_contribution",
        "mean_oracle_gap_vs_fixed",
        "mean_gold_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, values in family_summaries.items():
            writer.writerow(
                {
                    "family": family,
                    **{
                        field: values.get(field)
                        for field in fields
                        if field != "family"
                    },
                }
            )


def render_report(
    summary: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    families = summary["family_summaries"]
    adaptive = summary.get("adaptive_error_summary")

    lines = [
        "# Day13-3 F1 Failure Decomposition & Optimization Decision",
        "",
        "## Boundary",
        "",
        "- API calls: **0**",
        "- LLM calls: **0**",
        "- Ranking modified: **No**",
        "- Production config modified: **No**",
        "- Gold usage: **offline failure analysis only**",
        "",
        "## Internal F1 Anchors",
        "",
        (
            f"- Fixed K={summary['fixed_k']} Macro-F1: "
            f"**{summary['fixed_macro_f1_recomputed']:.6f}**"
        ),
        (
            "- Top20 Macro-F1: "
            f"**{summary['top20_macro_f1_recomputed']:.6f}**"
        ),
        (
            "- Oracle-K Macro-F1: "
            f"**{summary['oracle_macro_f1_recomputed']:.6f}**"
        ),
        (
            "- Oracle gap vs fixed policy: "
            f"**{summary['oracle_gap_vs_fixed_macro_f1']:+.6f}**"
        ),
        "",
        "## Failure Families",
        "",
        "| Family | Queries | Share | Oracle-gap contribution | Mean gap/query | Mean gold count | Primary source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    source_map = {
        "retrieval_ceiling": "Retrieval",
        "deep_ranked_hit": "Ranking",
        "mid_ranked_hit": "Ranking",
        "cutoff_mismatch": "Selection",
        "already_good": "Freeze",
    }

    for family, values in families.items():
        lines.append(
            "| {family} | {count} | {share:.1%} | {contrib:.6f} | "
            "{gap:.6f} | {gold:.2f} | {source} |".format(
                family=family,
                count=values["query_count"],
                share=values["query_share"],
                contrib=values[
                    "oracle_gap_macro_contribution"
                ],
                gap=values["mean_oracle_gap_vs_fixed"],
                gold=values["mean_gold_count"],
                source=source_map[family],
            )
        )

    lines += [
        "",
        "## Adaptive Cutoff Post-mortem",
        "",
    ]
    if adaptive:
        lines += [
            (
                "- OOF adaptive Macro-F1 recomputed from predicted K: "
                f"**{adaptive['macro_f1']:.6f}**"
            ),
            (
                "- Gain vs fixed policy: "
                f"**{adaptive['gain_vs_fixed_macro_f1']:+.6f}**"
            ),
            (
                "- Regret vs Oracle: "
                f"**{adaptive['regret_vs_oracle_macro_f1']:.6f}**"
            ),
            (
                "- Mean absolute cutoff error: "
                f"**{adaptive['mean_absolute_cutoff_error']:.2f} ranks**"
            ),
            (
                "- Over-return / under-return / exact: "
                f"**{adaptive['over_return_queries']} / "
                f"{adaptive['under_return_queries']} / "
                f"{adaptive['exact_cutoff_queries']}**"
            ),
        ]
    else:
        lines.append(
            "Day13-2 OOF cutoff predictions were not found. "
            "The decomposition remains valid, but adaptive post-mortem "
            "is omitted."
        )

    lines += [
        "",
        "## Go / No-Go",
        "",
        f"**Primary decision: `{decision['primary_decision']}`**",
        "",
        decision["decision_text"],
        "",
        "## Interpretation",
        "",
        (
            "`retrieval_ceiling` means no strict gold paper appears anywhere "
            "in E3 Top100, so ranking or cutoff changes cannot recover it."
        ),
        (
            "`deep_ranked_hit` and `mid_ranked_hit` mean a relevant paper is "
            "present but appears too deep for the simple fixed result boundary."
        ),
        (
            "`cutoff_mismatch` means at least one relevant paper is already "
            "inside Top5, but the query-level oracle prefers a materially "
            "different boundary."
        ),
        (
            "`already_good` means the fixed K policy is already within the "
            "configured F1 tolerance of Oracle-K."
        ),
        "",
        (
            "Oracle-K and all family labels use gold information and are "
            "analysis-only. They must not become inference features and "
            "must not be reported as competition platform scores."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Day13-3 F1 failure decomposition from Day13-1 "
            "Oracle-K and optional Day13-2 OOF cutoff artifacts."
        )
    )
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument("--adaptive", default=str(DEFAULT_ADAPTIVE))
    parser.add_argument(
        "--adaptive-predictions",
        default=str(DEFAULT_ADAPTIVE_PREDICTIONS),
    )
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--priority-limit", type=int, default=15)
    args = parser.parse_args()

    oracle_path = Path(args.oracle)
    adaptive_path = Path(args.adaptive)
    adaptive_predictions_path = Path(args.adaptive_predictions)
    gold_path = Path(args.gold)
    output_dir = Path(args.output_dir)

    if not oracle_path.is_file():
        print(
            f"[Day13-3] Missing oracle artifact: {oracle_path}",
            file=sys.stderr,
        )
        return 2

    oracle_rows = read_jsonl(oracle_path)
    adaptive_by_qid: dict[str, dict[str, Any]] = {}
    if adaptive_path.is_file():
        adaptive_by_qid = {
            str(row["qid"]): row
            for row in read_jsonl(adaptive_path)
        }

    # Reuse ScholarPath's canonical evaluator for the OOF adaptive result.
    # This matters because prediction deduplication can shift the effective
    # cutoff ranks; recomputing F1 only from Oracle hit ranks can therefore
    # disagree with the Day13-2 OOF metric.
    adaptive_f1_by_qid: dict[str, float] = {}
    if (
        adaptive_predictions_path.is_file()
        and gold_path.is_file()
    ):
        adaptive_eval = evaluate_records(
            gold_records=load_gold(gold_path),
            prediction_records=load_predictions(
                adaptive_predictions_path
            ),
            config=EvaluationConfig(
                mode="strict",
                recall_at=(20, 50, 100),
                deduplicate=True,
            ),
        )
        adaptive_f1_by_qid = {
            item.qid: float(item.f1)
            for item in adaptive_eval.per_query
        }

    rows = [
        build_failure_row(
            oracle_row,
            adaptive_row=adaptive_by_qid.get(
                str(oracle_row.get("qid") or "")
            ),
            adaptive_query_f1=adaptive_f1_by_qid.get(
                str(oracle_row.get("qid") or "")
            ),
            fixed_k=args.fixed_k,
            top_k=args.top_k,
            epsilon=args.epsilon,
        )
        for oracle_row in oracle_rows
    ]

    summary = summarize_failure_rows(rows)
    decision = derive_go_no_go(summary)
    priority = choose_priority_queries(
        rows,
        limit=args.priority_limit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "per_query_failure_decomposition.jsonl",
        [row.to_dict() for row in rows],
    )
    write_json(
        output_dir / "failure_family_summary.json",
        summary,
    )
    write_family_csv(
        output_dir / "failure_family_table.csv",
        summary["family_summaries"],
    )
    write_jsonl(
        output_dir / "priority_queries.jsonl",
        priority,
    )
    write_json(
        output_dir / "optimization_decision.json",
        decision,
    )
    (output_dir / "day13_3_report.md").write_text(
        render_report(summary, decision) + "\n",
        encoding="utf-8",
    )

    print("[Day13-3] API calls = 0")
    print("[Day13-3] LLM calls = 0")
    print("[Day13-3] ranking modified = False")
    print("[Day13-3] gold usage = offline analysis only")
    print()
    print("=== Failure Families ===")
    for family, values in summary["family_summaries"].items():
        print(
            f"{family}: "
            f"{values['query_count']} queries, "
            "oracle-gap contribution="
            f"{values['oracle_gap_macro_contribution']:.6f}"
        )

    adaptive = summary.get("adaptive_error_summary")
    if adaptive:
        print()
        print("=== Adaptive Post-mortem ===")
        print(
            "OOF adaptive MacroF1 =",
            round(float(adaptive["macro_f1"]), 6),
        )
        print(
            "Mean absolute cutoff error =",
            round(
                float(adaptive["mean_absolute_cutoff_error"]),
                3,
            ),
        )

    print()
    print("=== Day13-3 Decision ===")
    print("primary_decision =", decision["primary_decision"])
    print(decision["decision_text"])
    print("[OUTPUT]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
