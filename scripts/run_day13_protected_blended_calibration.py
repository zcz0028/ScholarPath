from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records
from scholarpath.evaluation.protected_blended_calibration import (
    BETA_GRID,
    assign_query_folds,
    binary_ndcg_at_10,
    candidate_identity_unchanged,
    rerank_prediction_records,
    select_beta,
    strict_top100_tp_and_zero_recall,
    subset_records,
)

DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "realscholarquery_gold.jsonl"
)
DEFAULT_E3 = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day8_evidence_rerank"
    / "predictions_e3_selected.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day13_protected_blended_calibration"
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def serialize_paper(paper: Any) -> dict[str, Any]:
    to_dict = getattr(paper, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict(include_raw=True)
        except TypeError:
            return to_dict()
    if isinstance(paper, Mapping):
        return dict(paper)
    raise TypeError(f"Unsupported paper value: {type(paper)!r}")


def serialize_records(
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(records):
        record = records[qid]
        rows.append(
            {
                "qid": str(qid),
                "question": str(record.get("question") or ""),
                "papers": [
                    serialize_paper(paper)
                    for paper in record.get("papers") or []
                ],
            }
        )
    return rows


def evaluate_at_k(
    *,
    gold_records: Mapping[str, Mapping[str, Any]],
    prediction_records: Mapping[str, Mapping[str, Any]],
    k: int,
) -> dict[str, Any]:
    truncated: dict[str, dict[str, Any]] = {}
    for qid, record in prediction_records.items():
        item = dict(record)
        item["papers"] = list(record.get("papers") or [])[:k]
        truncated[str(qid)] = item

    result = evaluate_records(
        gold_records=dict(gold_records),
        prediction_records=truncated,
        config=EvaluationConfig(
            mode="strict",
            recall_at=(k,),
            deduplicate=True,
        ),
    )
    summary = result.summary()
    macro = summary.get("macro") or {}
    counts = summary.get("counts") or {}
    return {
        "k": k,
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "tp": int(counts.get("tp", 0)),
        "fp": int(counts.get("fp", 0)),
        "fn": int(counts.get("fn", 0)),
        "query_count": int(summary.get("query_count", 0)),
    }


def evaluate_bundle(
    *,
    gold_records: Mapping[str, Mapping[str, Any]],
    prediction_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    at5 = evaluate_at_k(
        gold_records=gold_records,
        prediction_records=prediction_records,
        k=5,
    )
    at20 = evaluate_at_k(
        gold_records=gold_records,
        prediction_records=prediction_records,
        k=20,
    )
    ndcg = binary_ndcg_at_10(
        gold_records=gold_records,
        prediction_records=prediction_records,
    )
    top100_tp, zero_recall = strict_top100_tp_and_zero_recall(
        gold_records=gold_records,
        prediction_records=prediction_records,
    )
    return {
        "macro_f1_at_5": at5["macro_f1"],
        "macro_precision_at_5": at5["macro_precision"],
        "macro_recall_at_5": at5["macro_recall"],
        "tp_at_5": at5["tp"],
        "macro_f1_at_20": at20["macro_f1"],
        "tp_at_20": at20["tp"],
        "ndcg_at_10": ndcg,
        "top100_tp": top100_tp,
        "zero_recall_queries": zero_recall,
    }


def render_report(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    oof = summary["oof_blended"]
    protection = summary["protection"]
    decision = summary["decision"]

    lines = [
        "# Day13-5 Protected Blended Calibration Ablation",
        "",
        "## Boundary",
        "",
        "- API calls: **0**",
        "- LLM calls: **0**",
        "- Candidate set changed: **No**",
        "- Gold used at validation inference: **No**",
        "- Parameters selected: **training queries only**",
        "",
        "## Model",
        "",
        "```text",
        "score = (1 - beta) * normalized_E3_score",
        "      + beta * B4_best_source_rank_prior",
        "```",
        "",
        "Beta grid: `" + ", ".join(str(x) for x in summary["beta_grid"]) + "`",
        "",
        "## OOF result",
        "",
        "| Metric | Frozen E3 | OOF blended | Delta |",
        "|---|---:|---:|---:|",
        f"| Macro-F1@5 | {baseline['macro_f1_at_5']:.6f} | {oof['macro_f1_at_5']:.6f} | {oof['macro_f1_at_5'] - baseline['macro_f1_at_5']:+.6f} |",
        f"| Macro-F1@20 | {baseline['macro_f1_at_20']:.6f} | {oof['macro_f1_at_20']:.6f} | {oof['macro_f1_at_20'] - baseline['macro_f1_at_20']:+.6f} |",
        f"| NDCG@10 | {baseline['ndcg_at_10']:.6f} | {oof['ndcg_at_10']:.6f} | {oof['ndcg_at_10'] - baseline['ndcg_at_10']:+.6f} |",
        f"| Top20 TP | {baseline['tp_at_20']} | {oof['tp_at_20']} | {oof['tp_at_20'] - baseline['tp_at_20']:+d} |",
        f"| Top100 TP | {baseline['top100_tp']} | {oof['top100_tp']} | {oof['top100_tp'] - baseline['top100_tp']:+d} |",
        f"| Zero-recall queries | {baseline['zero_recall_queries']} | {oof['zero_recall_queries']} | {oof['zero_recall_queries'] - baseline['zero_recall_queries']:+d} |",
        "",
        "## Protection",
        "",
    ]

    for key, value in protection.items():
        lines.append(f"- `{key}`: **{value}**")

    lines += [
        "",
        "## Final decision",
        "",
        f"**`{decision['status']}`**",
        "",
        decision["reason"],
        "",
        "This is an offline ablation. A rejected calibration must not be wired "
        "into the production/API path. An accepted calibration should still "
        "be integrated only in a separate follow-up commit after UI/API "
        "regression checks.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Day13-5 protected query-level-CV blended calibration "
            "on frozen Day8 E3 candidates."
        )
    )
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--e3", default=str(DEFAULT_E3))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--beta-grid",
        nargs="+",
        type=float,
        default=list(BETA_GRID),
    )
    parser.add_argument(
        "--ndcg-tolerance",
        type=float,
        default=0.005,
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    e3_path = Path(args.e3)
    output_dir = Path(args.output_dir)

    if not gold_path.is_file():
        print(f"[Day13-5] Missing gold: {gold_path}", file=sys.stderr)
        return 2
    if not e3_path.is_file():
        print(f"[Day13-5] Missing E3: {e3_path}", file=sys.stderr)
        return 2

    beta_grid = sorted(
        {
            round(float(beta), 10)
            for beta in args.beta_grid
            if 0.0 <= float(beta) <= 0.10
        }
    )
    if 0.0 not in beta_grid:
        beta_grid.insert(0, 0.0)

    gold_records = load_gold(gold_path)
    e3_records = load_predictions(e3_path)

    qids = sorted(gold_records)
    if set(qids) != set(e3_records):
        raise ValueError("Gold/E3 qid sets differ")

    baseline = evaluate_bundle(
        gold_records=gold_records,
        prediction_records=e3_records,
    )

    folds = assign_query_folds(qids, n_folds=args.folds)
    fold_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    oof_records: dict[str, dict[str, Any]] = {}

    for fold in range(args.folds):
        validation_qids = sorted(
            qid for qid in qids if folds[qid] == fold
        )
        train_qids = sorted(
            qid for qid in qids if folds[qid] != fold
        )

        train_gold = subset_records(gold_records, train_qids)
        train_e3 = subset_records(e3_records, train_qids)

        candidate_rows: list[dict[str, Any]] = []
        for beta in beta_grid:
            reranked_train = rerank_prediction_records(
                train_e3,
                beta=beta,
            )
            metrics = evaluate_bundle(
                gold_records=train_gold,
                prediction_records=reranked_train,
            )
            row = {
                "fold": fold,
                "beta": beta,
                "macro_f1_at_5": metrics["macro_f1_at_5"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "macro_f1_at_20": metrics["macro_f1_at_20"],
                "tp_at_20": metrics["tp_at_20"],
            }
            candidate_rows.append(row)
            grid_rows.append(row)

        selected = dict(select_beta(candidate_rows))
        selected_beta = float(selected["beta"])

        validation_e3 = subset_records(
            e3_records,
            validation_qids,
        )
        validation_gold = subset_records(
            gold_records,
            validation_qids,
        )
        validation_blended = rerank_prediction_records(
            validation_e3,
            beta=selected_beta,
        )
        validation_metrics = evaluate_bundle(
            gold_records=validation_gold,
            prediction_records=validation_blended,
        )

        for qid, record in validation_blended.items():
            oof_records[qid] = record

        fold_rows.append(
            {
                "fold": fold,
                "train_query_count": len(train_qids),
                "validation_query_count": len(validation_qids),
                "selected_beta": selected_beta,
                "train_macro_f1_at_5": selected["macro_f1_at_5"],
                "train_ndcg_at_10": selected["ndcg_at_10"],
                "validation_macro_f1_at_5": (
                    validation_metrics["macro_f1_at_5"]
                ),
                "validation_macro_f1_at_20": (
                    validation_metrics["macro_f1_at_20"]
                ),
                "validation_ndcg_at_10": (
                    validation_metrics["ndcg_at_10"]
                ),
                "validation_top20_tp": validation_metrics["tp_at_20"],
            }
        )

    if set(oof_records) != set(qids):
        raise AssertionError("OOF prediction coverage is incomplete")

    oof_metrics = evaluate_bundle(
        gold_records=gold_records,
        prediction_records=oof_records,
    )

    identity_ok = candidate_identity_unchanged(
        e3_records,
        oof_records,
    )
    top100_tp_ok = (
        oof_metrics["top100_tp"] == baseline["top100_tp"]
    )
    zero_recall_ok = (
        oof_metrics["zero_recall_queries"]
        == baseline["zero_recall_queries"]
    )
    f1_improved = (
        oof_metrics["macro_f1_at_5"]
        > baseline["macro_f1_at_5"] + 1e-12
    )
    ndcg_protected = (
        oof_metrics["ndcg_at_10"]
        >= baseline["ndcg_at_10"] - args.ndcg_tolerance
    )

    protection = {
        "candidate_identity_unchanged": identity_ok,
        "top100_tp_preserved": top100_tp_ok,
        "zero_recall_preserved": zero_recall_ok,
        "oof_macro_f1_at_5_improved": f1_improved,
        "ndcg_at_10_protected": ndcg_protected,
        "gold_used_at_validation_inference": False,
        "api_calls": 0,
        "llm_calls": 0,
    }

    accepted = all(
        [
            identity_ok,
            top100_tp_ok,
            zero_recall_ok,
            f1_improved,
            ndcg_protected,
        ]
    )

    decision = {
        "status": (
            "accept_protected_blended_calibration"
            if accepted
            else "reject_blended_calibration_keep_frozen_e3"
        ),
        "reason": (
            "OOF blended calibration improved Macro-F1@5 while preserving "
            "candidate identity, Top100 TP, zero-recall count, and NDCG guard."
            if accepted
            else
            "At least one held-out protection criterion failed. Keep the "
            "frozen Day8 E3 ranking and end ranking calibration work."
        ),
    }

    summary = {
        "schema_version": "day13.protected-blended-calibration.v1",
        "query_count": len(qids),
        "fold_count": args.folds,
        "beta_grid": beta_grid,
        "score_formula": (
            "(1-beta)*query_minmax(day8_final_score) + "
            "beta*(1/log2(b4_best_source_rank+1))"
        ),
        "selection_objective": (
            "train Macro-F1@5; tie-break NDCG@10; then smaller beta"
        ),
        "baseline": baseline,
        "oof_blended": oof_metrics,
        "folds": fold_rows,
        "protection": protection,
        "decision": decision,
        "experiment_boundary": {
            "offline_only": True,
            "network_calls": 0,
            "llm_calls": 0,
            "gold_used_for_training_selection": True,
            "gold_used_at_validation_inference": False,
            "candidate_pool_modified": False,
            "production_ranking_modified": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "calibration_summary.json",
        summary,
    )
    write_json(
        output_dir / "protection_audit.json",
        protection,
    )
    write_json(
        output_dir / "optimization_decision.json",
        decision,
    )
    write_json(
        output_dir / "fold_results.json",
        {
            "folds": fold_rows,
            "fold_assignments": folds,
        },
    )
    write_json(
        output_dir / "training_grid.json",
        {
            "rows": grid_rows,
        },
    )
    write_jsonl(
        output_dir / "oof_blended_predictions.jsonl",
        serialize_records(oof_records),
    )
    (output_dir / "day13_5_report.md").write_text(
        render_report(summary) + "\n",
        encoding="utf-8",
    )

    print("[Day13-5] API calls = 0")
    print("[Day13-5] LLM calls = 0")
    print("[Day13-5] validation inference uses gold = False")
    print()
    for row in fold_rows:
        print(
            f"[Fold {row['fold']}] "
            f"beta={row['selected_beta']:.2f} "
            f"val_F1@5={row['validation_macro_f1_at_5']:.6f} "
            f"val_NDCG@10={row['validation_ndcg_at_10']:.6f}"
        )

    print()
    print("=== Day13-5 OOF ===")
    print(
        "E3 MacroF1@5 =",
        round(float(baseline["macro_f1_at_5"]), 6),
    )
    print(
        "OOF MacroF1@5 =",
        round(float(oof_metrics["macro_f1_at_5"]), 6),
    )
    print(
        "E3 NDCG@10 =",
        round(float(baseline["ndcg_at_10"]), 6),
    )
    print(
        "OOF NDCG@10 =",
        round(float(oof_metrics["ndcg_at_10"]), 6),
    )
    print(
        "Top100 TP =",
        baseline["top100_tp"],
        "->",
        oof_metrics["top100_tp"],
    )
    print(
        "Zero recall =",
        baseline["zero_recall_queries"],
        "->",
        oof_metrics["zero_recall_queries"],
    )
    print("decision =", decision["status"])
    print("[OUTPUT]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
