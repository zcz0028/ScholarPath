from __future__ import annotations

import argparse
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
    assign_query_folds, binary_ndcg_at_10, candidate_identity_unchanged,
    select_beta, strict_top100_tp_and_zero_recall, subset_records,
)
from scholarpath.evaluation.comprehensive_evidence_calibration import (
    BETA_GRID, rerank_prediction_records,
)

DEFAULT_GOLD = PROJECT_ROOT / "data/processed/realscholarquery_gold.jsonl"
DEFAULT_E3 = PROJECT_ROOT / "outputs/week2_day8_evidence_rerank/predictions_e3_selected.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/week2_day14_comprehensive_evidence_ablation"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def serialize_paper(paper: Any) -> dict[str, Any]:
    if isinstance(paper, Mapping): return dict(paper)
    fn = getattr(paper, "to_dict", None)
    if callable(fn):
        try: return fn(include_raw=True)
        except TypeError: return fn()
    raise TypeError(type(paper))


def write_jsonl(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(records):
            r = records[qid]
            f.write(json.dumps({"qid": qid, "question": str(r.get("question") or ""),
                                "papers": [serialize_paper(p) for p in r.get("papers") or []]},
                               ensure_ascii=False) + "\n")


def evaluate_at_k(gold, pred, k: int) -> dict[str, Any]:
    truncated = {}
    for qid, record in pred.items():
        item = dict(record); item["papers"] = list(record.get("papers") or [])[:k]; truncated[qid] = item
    result = evaluate_records(dict(gold), truncated, EvaluationConfig(mode="strict", recall_at=(k,), deduplicate=True))
    s = result.summary(); m = s.get("macro") or {}; c = s.get("counts") or {}
    return {"macro_f1": float(m.get("f1", 0.0)), "tp": int(c.get("tp", 0))}


def evaluate_bundle(gold, pred) -> dict[str, Any]:
    a5, a20 = evaluate_at_k(gold, pred, 5), evaluate_at_k(gold, pred, 20)
    tp100, zero = strict_top100_tp_and_zero_recall(gold_records=gold, prediction_records=pred)
    return {"macro_f1_at_5": a5["macro_f1"], "macro_f1_at_20": a20["macro_f1"],
            "tp_at_20": a20["tp"], "ndcg_at_10": binary_ndcg_at_10(gold_records=gold, prediction_records=pred),
            "top100_tp": tp100, "zero_recall_queries": zero}


def main() -> int:
    p = argparse.ArgumentParser(description="Day14-1D protected comprehensive-evidence ablation")
    p.add_argument("--gold", default=str(DEFAULT_GOLD)); p.add_argument("--e3", default=str(DEFAULT_E3))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR)); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--beta-grid", nargs="+", type=float, default=list(BETA_GRID)); p.add_argument("--ndcg-tolerance", type=float, default=0.0)
    args = p.parse_args(); out = Path(args.output_dir)
    gold, e3 = load_gold(Path(args.gold)), load_predictions(Path(args.e3))
    qids = sorted(gold)
    if set(qids) != set(e3): raise ValueError("Gold/E3 qid sets differ")
    grid = sorted({round(float(x), 10) for x in args.beta_grid if 0 <= float(x) <= .10} | {0.0})
    baseline = evaluate_bundle(gold, e3); folds = assign_query_folds(qids, n_folds=args.folds)
    fold_rows, grid_rows, oof = [], [], {}
    for fold in range(args.folds):
        val_q = sorted(q for q in qids if folds[q] == fold); train_q = sorted(q for q in qids if folds[q] != fold)
        train_g, train_e = subset_records(gold, train_q), subset_records(e3, train_q)
        candidates = []
        for beta in grid:
            metrics = evaluate_bundle(train_g, rerank_prediction_records(train_e, beta=beta))
            row = {"fold": fold, "beta": beta, **metrics}; candidates.append(row); grid_rows.append(row)
        selected = dict(select_beta(candidates)); beta = float(selected["beta"])
        val_e, val_g = subset_records(e3, val_q), subset_records(gold, val_q)
        val_r = rerank_prediction_records(val_e, beta=beta); val_m = evaluate_bundle(val_g, val_r); oof.update(val_r)
        base_val = evaluate_bundle(val_g, val_e)
        fold_rows.append({"fold": fold, "selected_beta": beta, "train_query_count": len(train_q),
                          "validation_query_count": len(val_q), "train_macro_f1_at_5": selected["macro_f1_at_5"],
                          "train_ndcg_at_10": selected["ndcg_at_10"], "validation_macro_f1_at_5": val_m["macro_f1_at_5"],
                          "validation_ndcg_at_10": val_m["ndcg_at_10"], "validation_f1_delta": val_m["macro_f1_at_5"]-base_val["macro_f1_at_5"]})
    metrics = evaluate_bundle(gold, oof)
    identity = candidate_identity_unchanged(e3, oof); tp_ok = metrics["top100_tp"] == baseline["top100_tp"]
    zero_ok = metrics["zero_recall_queries"] == baseline["zero_recall_queries"]
    f1_ok = metrics["macro_f1_at_5"] > baseline["macro_f1_at_5"] + 1e-12
    ndcg_ok = metrics["ndcg_at_10"] >= baseline["ndcg_at_10"] - args.ndcg_tolerance
    non_single_fold = sum(r["validation_f1_delta"] > 1e-12 for r in fold_rows) >= 2
    protection = {"candidate_identity_unchanged": identity, "top100_tp_preserved": tp_ok, "zero_recall_preserved": zero_ok,
                  "oof_macro_f1_at_5_improved": f1_ok, "ndcg_at_10_protected": ndcg_ok,
                  "gain_not_single_fold_only": non_single_fold, "gold_used_at_validation_inference": False, "api_calls": 0, "llm_calls": 0}
    accepted = all([identity, tp_ok, zero_ok, f1_ok, ndcg_ok, non_single_fold])
    decision = {"status": "accept_comprehensive_evidence_score" if accepted else "reject_comprehensive_evidence_keep_frozen_e3",
                "reason": "Held-out comprehensive evidence reranking passed every protection guard." if accepted else "At least one held-out guard failed; keep frozen E3 and do not modify production ranking."}
    summary = {"schema_version": "day14.comprehensive-evidence-ablation.v1", "query_count": len(qids), "fold_count": args.folds,
               "beta_grid": grid, "score_formula": "(1-beta)*query_minmax(E3)+beta*comprehensive_evidence",
               "evidence_formula": "weighted_coverage*matched_confidence*field_quality*specificity_factor-generic_entity_only_penalty",
               "selection_objective": "train Macro-F1@5; tie-break NDCG@10; then smaller beta", "baseline": baseline,
               "oof_comprehensive_evidence": metrics, "folds": fold_rows, "protection": protection, "decision": decision,
               "experiment_boundary": {"offline_only": True, "network_calls": 0, "llm_calls": 0,
                 "gold_used_for_training_selection": True, "gold_used_at_validation_inference": False,
                 "candidate_pool_modified": False, "production_ranking_modified": False}}
    out.mkdir(parents=True, exist_ok=True); write_json(out/"ablation_summary.json", summary); write_json(out/"protection_audit.json", protection)
    write_json(out/"optimization_decision.json", decision); write_json(out/"fold_results.json", {"folds": fold_rows, "fold_assignments": folds})
    write_json(out/"training_grid.json", {"rows": grid_rows}); write_jsonl(out/"oof_comprehensive_evidence_predictions.jsonl", oof)
    print("[Day14-1D] Comprehensive Evidence Score Protected Ablation"); print("API calls = 0\nLLM calls = 0\nvalidation inference uses gold = False\n")
    for r in fold_rows: print(f"[Fold {r['fold']}] beta={r['selected_beta']:.2f} val_F1@5={r['validation_macro_f1_at_5']:.6f} delta={r['validation_f1_delta']:+.6f}")
    print("\n=== Day14-1D OOF ==="); print("E3 MacroF1@5 =", round(baseline["macro_f1_at_5"],6)); print("OOF MacroF1@5 =", round(metrics["macro_f1_at_5"],6))
    print("E3 NDCG@10 =", round(baseline["ndcg_at_10"],6)); print("OOF NDCG@10 =", round(metrics["ndcg_at_10"],6))
    print("Top100 TP =", baseline["top100_tp"], "->", metrics["top100_tp"]); print("Zero recall =", baseline["zero_recall_queries"], "->", metrics["zero_recall_queries"])
    print("decision =", decision["status"]); print("[OUTPUT]", out); return 0

if __name__ == "__main__": raise SystemExit(main())
