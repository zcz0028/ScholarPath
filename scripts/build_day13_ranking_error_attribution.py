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

from scholarpath.evaluation.ranking_error_attribution import (
    FEATURE_SPECS,
    build_query_attribution,
    derive_ranking_decision,
    feature_only_counterfactual,
    pairwise_feature_summary,
)

DEFAULT_E3 = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day8_evidence_rerank"
    / "predictions_e3_selected.jsonl"
)
DEFAULT_ORACLE = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_f1_diagnostic"
    / "oracle_k_per_query.jsonl"
)
DEFAULT_FAILURES = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_f1_failure_decomposition"
    / "per_query_failure_decomposition.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_ranking_error_attribution"
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
                    f"Expected object at {path}:{line_no}"
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


def write_feature_csv(
    path: Path,
    pairwise: dict[str, Any],
    counterfactual: dict[str, Any],
) -> None:
    fields = [
        "feature",
        "higher_is_better",
        "pair_count",
        "pairwise_tp_win_rate_vs_blocking_fp",
        "first_tp_mean_oriented_advantage",
        "first_tp_median_oriented_advantage",
        "queries_with_positive_first_tp_advantage",
        "query_count",
        "current_label_hits_at_5",
        "feature_only_label_hits_at_5",
        "delta_label_hits_at_5",
        "current_label_hits_at_20",
        "feature_only_label_hits_at_20",
        "delta_label_hits_at_20",
        "queries_improved_at_20",
        "queries_worsened_at_20",
    ]
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature, _ in FEATURE_SPECS:
            writer.writerow(
                {
                    "feature": feature,
                    **pairwise[feature],
                    **{
                        key: value
                        for key, value in counterfactual[feature].items()
                        if key != "analysis_only"
                        and key != "ranking_query_count"
                    },
                }
            )


def render_report(
    *,
    ranking_rows: list[dict[str, Any]],
    pairwise: dict[str, Any],
    counterfactual: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    sorted_features = sorted(
        pairwise,
        key=lambda feature: (
            float(
                pairwise[feature][
                    "pairwise_tp_win_rate_vs_blocking_fp"
                ]
            ),
            int(
                counterfactual[feature][
                    "delta_label_hits_at_20"
                ]
            ),
        ),
        reverse=True,
    )

    lines = [
        "# Day13-4 Ranking Error Attribution",
        "",
        "## Boundary",
        "",
        "- API calls: **0**",
        "- LLM calls: **0**",
        "- E3 ranking modified: **No**",
        "- Production config modified: **No**",
        "- Gold/matched ranks used: **offline diagnostic only**",
        "",
        "## Ranking-depth scope",
        "",
        (
            f"- Ranking-depth queries analyzed: "
            f"**{len(ranking_rows)}**"
        ),
        (
            "- Scope: `deep_ranked_hit` + `mid_ranked_hit` from "
            "Day13-3; if that artifact is absent, the script falls "
            "back to Oracle-K queries whose first strict hit is after rank 5."
        ),
        "",
        "## Feature separation",
        "",
        "| Feature | TP win rate vs blocking FP | First-TP mean advantage | Δ label hits@5 | Δ label hits@20 |",
        "|---|---:|---:|---:|---:|",
    ]

    for feature in sorted_features:
        p = pairwise[feature]
        c = counterfactual[feature]
        lines.append(
            "| {f} | {w:.3f} | {a:+.4f} | {d5:+d} | {d20:+d} |".format(
                f=feature,
                w=p["pairwise_tp_win_rate_vs_blocking_fp"],
                a=p["first_tp_mean_oriented_advantage"],
                d5=c["delta_label_hits_at_5"],
                d20=c["delta_label_hits_at_20"],
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        (
            f"**`{decision['primary_decision']}`**"
        ),
        "",
        decision["decision_text"],
        "",
    ]

    if decision.get("candidate_signals"):
        lines += [
            "Candidate signals for a *small blended* follow-up:",
            "",
        ]
        for item in decision["candidate_signals"]:
            lines.append(
                "- `{feature}`: pairwise win={win:.3f}, "
                "Δhits@20={d20:+d}, Δhits@5={d5:+d}".format(
                    feature=item["feature"],
                    win=item["pairwise_win_rate"],
                    d20=item["delta_label_hits_at_20"],
                    d5=item["delta_label_hits_at_5"],
                )
            )
        lines.append("")

    lines += [
        "## Guardrail",
        "",
        decision["guardrail"],
        "",
        "## Interpretation boundary",
        "",
        (
            "The feature-only counterfactual deliberately uses strict "
            "matched-rank labels and is **not** a leaderboard metric, "
            "production reranker, or competition-score estimate. Its only "
            "purpose is to identify whether an already-existing signal can "
            "move known relevant papers above E3 blocking false positives."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Day13-4 ranking-error attribution on the frozen Day8 E3 "
            "candidate ranking."
        )
    )
    parser.add_argument("--e3", default=str(DEFAULT_E3))
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument(
        "--failure-decomposition",
        default=str(DEFAULT_FAILURES),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    e3_path = Path(args.e3)
    oracle_path = Path(args.oracle)
    failure_path = Path(args.failure_decomposition)
    output_dir = Path(args.output_dir)

    if not e3_path.is_file():
        print(f"[Day13-4] Missing E3 artifact: {e3_path}", file=sys.stderr)
        return 2
    if not oracle_path.is_file():
        print(
            f"[Day13-4] Missing Oracle-K artifact: {oracle_path}",
            file=sys.stderr,
        )
        return 2

    e3_rows = read_jsonl(e3_path)
    e3_by_qid = {str(row["qid"]): row for row in e3_rows}
    oracle_rows = read_jsonl(oracle_path)
    oracle_by_qid = {
        str(row["qid"]): row
        for row in oracle_rows
    }

    ranking_rows: list[dict[str, Any]] = []
    if failure_path.is_file():
        for row in read_jsonl(failure_path):
            if row.get("family") not in {
                "deep_ranked_hit",
                "mid_ranked_hit",
            }:
                continue
            ranking_rows.append(
                {
                    "qid": str(row["qid"]),
                    "question": str(row.get("question") or ""),
                    "family": str(row["family"]),
                    "gold_count": int(row.get("gold_count") or 0),
                    "first_hit_rank": int(row["first_hit_rank"]),
                    "hit_ranks": [
                        int(x)
                        for x in row.get("hit_ranks") or []
                    ],
                }
            )
    else:
        for row in oracle_rows:
            first = row.get("first_hit_rank")
            if first is None or int(first) <= 5:
                continue
            ranking_rows.append(
                {
                    "qid": str(row["qid"]),
                    "question": str(row.get("question") or ""),
                    "family": (
                        "deep_ranked_hit"
                        if int(first) > 20
                        else "mid_ranked_hit"
                    ),
                    "gold_count": int(row.get("gold_count") or 0),
                    "first_hit_rank": int(first),
                    "hit_ranks": [
                        int(x)
                        for x in row.get("hit_ranks") or []
                    ],
                }
            )

    missing = sorted(
        {
            str(row["qid"])
            for row in ranking_rows
        }
        - set(e3_by_qid)
    )
    if missing:
        raise ValueError(
            f"Ranking qids missing from E3: {missing}"
        )

    query_attribution = [
        build_query_attribution(
            qid=str(row["qid"]),
            question=str(row.get("question") or ""),
            family=str(row["family"]),
            gold_count=int(row.get("gold_count") or 0),
            hit_ranks=row.get("hit_ranks") or [],
            papers=e3_by_qid[str(row["qid"])].get("papers") or [],
        ).to_dict()
        for row in ranking_rows
    ]

    pairwise = pairwise_feature_summary(
        ranking_rows=ranking_rows,
        paper_records_by_qid=e3_by_qid,
    )
    counterfactual = feature_only_counterfactual(
        ranking_rows=ranking_rows,
        paper_records_by_qid=e3_by_qid,
    )
    decision = derive_ranking_decision(
        pairwise_summary=pairwise,
        counterfactual=counterfactual,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "query_ranking_attribution.jsonl",
        query_attribution,
    )
    write_json(
        output_dir / "feature_separation_summary.json",
        pairwise,
    )
    write_json(
        output_dir / "feature_counterfactual_summary.json",
        counterfactual,
    )
    write_feature_csv(
        output_dir / "feature_attribution_table.csv",
        pairwise,
        counterfactual,
    )
    write_json(
        output_dir / "optimization_decision.json",
        decision,
    )
    (output_dir / "day13_4_report.md").write_text(
        render_report(
            ranking_rows=ranking_rows,
            pairwise=pairwise,
            counterfactual=counterfactual,
            decision=decision,
        ) + "\n",
        encoding="utf-8",
    )

    print("[Day13-4] API calls = 0")
    print("[Day13-4] LLM calls = 0")
    print("[Day13-4] E3 ranking modified = False")
    print(
        "[Day13-4] ranking-depth queries =",
        len(ranking_rows),
    )
    print()
    print("=== Top attribution signals ===")
    ranked_features = sorted(
        pairwise,
        key=lambda feature: (
            pairwise[feature][
                "pairwise_tp_win_rate_vs_blocking_fp"
            ],
            counterfactual[feature][
                "delta_label_hits_at_20"
            ],
        ),
        reverse=True,
    )
    for feature in ranked_features[:6]:
        print(
            feature,
            "pairwise_win=",
            round(
                float(
                    pairwise[feature][
                        "pairwise_tp_win_rate_vs_blocking_fp"
                    ]
                ),
                4,
            ),
            "delta_hits@20=",
            counterfactual[feature]["delta_label_hits_at_20"],
            "delta_hits@5=",
            counterfactual[feature]["delta_label_hits_at_5"],
        )

    print()
    print("=== Day13-4 Decision ===")
    print(
        "primary_decision =",
        decision["primary_decision"],
    )
    print(decision["decision_text"])
    print("[OUTPUT]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
