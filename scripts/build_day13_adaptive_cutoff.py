from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.matching import deduplicate_predictions, match_papers
from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records
from scholarpath.selector.adaptive_cutoff import AdaptiveCutoffModel, DEFAULT_KS, extract_cutoff_features

DEFAULT_GOLD = PROJECT_ROOT / "data/processed/realscholarquery_gold.jsonl"
DEFAULT_E3 = PROJECT_ROOT / "outputs/week2_day8_evidence_rerank/predictions_e3_selected.jsonl"
DEFAULT_DAY13_1 = PROJECT_ROOT / "outputs/week2_day13_f1_diagnostic"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/week2_day13_adaptive_cutoff"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8-sig") if x.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_fold(qid: str, folds: int, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()
    return int(digest[:12], 16) % folds


def per_query_oracle_k(gold, predictions, max_k: int = 100) -> tuple[int, float]:
    deduped, _ = deduplicate_predictions(predictions)
    usable = deduped[:max_k]
    best_k, best_f1 = 1, -1.0
    for k in range(1, len(usable) + 1):
        m = match_papers(predictions=usable[:k], gold=gold, mode="strict")
        p = m.tp / k if k else 0.0
        r = m.tp / len(gold) if gold else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1 + 1e-15:
            best_k, best_f1 = k, f1
    return (best_k if usable else 0), max(best_f1, 0.0)


def query_f1_from_hit_ranks(hit_ranks, gold_count: int, k: int) -> float:
    tp = sum(1 for rank in hit_ranks if rank <= k)
    precision = tp / k if k else 0.0
    recall = tp / gold_count if gold_count else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def best_train_fixed_k(train_qids, oracle_rows, ks) -> int:
    best = None
    for k in ks:
        f1s = [query_f1_from_hit_ranks(oracle_rows[q]["hit_ranks"], int(oracle_rows[q]["gold_count"]), k) for q in train_qids]
        score = fmean(f1s) if f1s else 0.0
        candidate = (score, -k, k)
        if best is None or candidate > best:
            best = candidate
    return int(best[2])


def evaluate_records_list(gold, rows):
    pred = {r["qid"]: {"qid": r["qid"], "question": r.get("question"), "papers": load_predictions_from_row(r)} for r in rows}
    return evaluate_records(gold_records=gold, prediction_records=pred, config=EvaluationConfig(mode="strict", recall_at=(20,50,100), deduplicate=True))


def load_predictions_from_row(row):
    # Reuse the canonical parser so evaluation receives PaperRecord objects.
    from scholarpath.evaluation.io import parse_prediction_papers
    return parse_prediction_papers(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Day13-2 leakage-safe gold-free adaptive cutoff with query-level CV")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--e3", default=str(DEFAULT_E3))
    ap.add_argument("--day13-1-dir", default=str(DEFAULT_DAY13_1))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neighbors", type=int, default=7)
    ap.add_argument("--seed", default="scholarpath-day13-2-v1")
    ap.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS))
    args = ap.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")
    gold = load_gold(args.gold)
    pred = load_predictions(args.e3)
    raw_rows = {r["qid"]: r for r in read_jsonl(Path(args.e3))}
    qids = sorted(set(gold) & set(pred) & set(raw_rows))
    ks = tuple(sorted({k for k in args.k if 1 <= k <= 100}))

    features = {qid: extract_cutoff_features(raw_rows[qid]) for qid in qids}
    oracle_rows = {r["qid"]: r for r in read_jsonl(Path(args.day13_1_dir) / "oracle_k_per_query.jsonl")}
    missing_oracle = sorted(set(qids) - set(oracle_rows))
    if missing_oracle:
        raise SystemExit(f"Day13-1 oracle rows missing qids: {missing_oracle}")
    oracle = {qid: int(oracle_rows[qid]["best_k"]) for qid in qids}
    folds = {qid: stable_fold(qid, args.folds, args.seed) for qid in qids}

    oof_rows, cutoff_rows, fold_rows = [], [], []
    leakage_ok = True
    for fold in range(args.folds):
        val_qids = [q for q in qids if folds[q] == fold]
        train_qids = [q for q in qids if folds[q] != fold]
        if not val_qids or not train_qids:
            raise SystemExit(f"Fold {fold} is empty; change --seed or --folds")
        train_best_k = best_train_fixed_k(train_qids, oracle_rows, ks)
        model = AdaptiveCutoffModel.fit(
            [features[q] for q in train_qids], [oracle[q] for q in train_qids],
            neighbor_count=args.neighbors, fallback_k=train_best_k, allowed_ks=ks,
        )
        predicted_ks = []
        for qid in val_qids:
            k, reason = model.predict(features[qid])
            predicted_ks.append(k)
            raw = raw_rows[qid]
            row = dict(raw)
            row["papers"] = list(raw.get("papers") or [])[:k]
            row["day13_2_cutoff"] = {"predicted_k": k, "fold": fold, "gold_used_at_inference": False, "policy": "robust_scaled_knn_oracle_k_train_only", **reason}
            oof_rows.append(row)
            cutoff_rows.append({"qid": qid, "fold": fold, "predicted_k": k, "oracle_k_analysis_only": oracle[qid], "features": features[qid], "reason": reason})
        fold_rows.append({"fold": fold, "train_query_count": len(train_qids), "validation_query_count": len(val_qids), "train_fixed_k": train_best_k, "validation_predicted_k_mean": fmean(predicted_ks)})
        leakage_ok = leakage_ok and not (set(train_qids) & set(val_qids))

    order = {qid: i for i, qid in enumerate(qids)}
    oof_rows.sort(key=lambda r: order[r["qid"]])
    cutoff_rows.sort(key=lambda r: order[r["qid"]])
    result = evaluate_records_list(gold, oof_rows)
    summary = result.summary()

    day13_1_dir = Path(args.day13_1_dir)
    global_summary = json.loads((day13_1_dir / "global_k_summary.json").read_text(encoding="utf-8"))
    oracle_summary = json.loads((day13_1_dir / "oracle_k_summary.json").read_text(encoding="utf-8"))
    best_fixed = global_summary["best_fixed"]
    top20 = global_summary["top20_reference"]
    adaptive_f1 = float(summary["macro"]["f1"])

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "oof_predictions.jsonl", oof_rows)
    write_jsonl(out / "oof_cutoff_predictions.jsonl", cutoff_rows)
    write_json(out / "fold_assignments.json", {"seed": args.seed, "folds": args.folds, "assignments": folds})
    write_json(out / "cv_fold_results.json", fold_rows)
    write_json(out / "feature_summary.json", {"feature_names": sorted(next(iter(features.values())).keys()), "query_count": len(qids), "features_are_gold_free": True})
    audit = {"schema_version":"day13.adaptive-cutoff.audit.v1", "query_level_cv":True, "train_validation_disjoint":leakage_ok, "gold_used_to_build_training_targets":True, "gold_used_at_inference":False, "validation_oracle_k_used_by_model":False, "ranking_modified":False, "api_calls":0, "llm_calls":0}
    write_json(out / "leakage_audit.json", audit)
    decision = "adaptive_beats_best_fixed" if adaptive_f1 > float(best_fixed["macro_f1"]) else "keep_best_fixed_k"
    final = {"schema_version":"day13.adaptive-cutoff.v1", "evaluation_protocol":{"matching_mode":"strict","deduplicate":True,"oof_query_level_cv":True,"folds":args.folds,"gold_used_at_inference":False,"ranking_modified":False,"api_calls":0,"llm_calls":0}, "top20_macro_f1":float(top20["macro_f1"]), "best_fixed_k":int(best_fixed["k"]), "best_fixed_macro_f1":float(best_fixed["macro_f1"]), "oof_adaptive_macro_f1":adaptive_f1, "oof_adaptive_macro_precision":float(summary["macro"]["precision"]), "oof_adaptive_macro_recall":float(summary["macro"]["recall"]), "oracle_macro_f1_analysis_only":float(oracle_summary["oracle_macro_f1"]), "gain_vs_top20":adaptive_f1-float(top20["macro_f1"]), "gain_vs_best_fixed":adaptive_f1-float(best_fixed["macro_f1"]), "decision":decision}
    write_json(out / "adaptive_cutoff_summary.json", final)
    report = f"""# Day13-2 Gold-free Adaptive Cutoff + Query-level CV\n\n## Boundary\n\n- 5-fold query-level out-of-fold evaluation: **Yes**\n- Gold used at inference: **No**\n- Validation Oracle-K visible to model: **No**\n- Ranking modified: **No**\n- API calls: **0**\n- LLM calls: **0**\n\n## Result\n\n| Policy | Macro-F1 |\n|---|---:|\n| Fixed Top20 | {final['top20_macro_f1']:.6f} |\n| Best fixed K={final['best_fixed_k']} | {final['best_fixed_macro_f1']:.6f} |\n| OOF adaptive cutoff | {final['oof_adaptive_macro_f1']:.6f} |\n| Oracle-K (analysis only) | {final['oracle_macro_f1_analysis_only']:.6f} |\n\nDecision: **{decision}**.\n\nThe adaptive policy is judged only by out-of-fold predictions. Gold labels create training targets inside each training fold and are never features or inference inputs. Internal strict F1 is an iteration metric, not a claimed competition score.\n"""
    (out / "day13_2_report.md").write_text(report, encoding="utf-8")

    print("[Day13-2] API calls = 0")
    print("[Day13-2] LLM calls = 0")
    print("[Day13-2] ranking modified = False")
    print("[Day13-2] gold used at inference = False")
    print(f"[Day13-2] OOF MacroF1 = {adaptive_f1:.6f}")
    print(f"[Day13-2] Best fixed K={best_fixed['k']} MacroF1 = {float(best_fixed['macro_f1']):.6f}")
    print(f"[Day13-2] Decision = {decision}")
    print(f"[OUTPUT] {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
