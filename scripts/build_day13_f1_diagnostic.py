from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records
from scholarpath.evaluation.matching import deduplicate_predictions, match_papers
from scholarpath.paper.schema import PaperRecord

DEFAULT_GOLD = PROJECT_ROOT / "data" / "processed" / "realscholarquery_gold.jsonl"
DEFAULT_E3 = PROJECT_ROOT / "outputs" / "week2_day8_evidence_rerank" / "predictions_e3_selected.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week2_day13_f1_diagnostic"
DEFAULT_KS = (1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,80,100)


@dataclass(slots=True)
class QueryOracle:
    qid: str
    question: str
    gold_count: int
    candidate_count: int
    best_k: int
    best_f1: float
    best_precision: float
    best_recall: float
    best_tp: int
    top20_f1: float
    top20_tp: int
    first_hit_rank: int | None
    last_hit_rank: int | None
    hit_ranks: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_count": self.gold_count,
            "candidate_count": self.candidate_count,
            "best_k": self.best_k,
            "best_f1": self.best_f1,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_tp": self.best_tp,
            "top20_f1": self.top20_f1,
            "top20_tp": self.top20_tp,
            "first_hit_rank": self.first_hit_rank,
            "last_hit_rank": self.last_hit_rank,
            "hit_ranks": self.hit_ranks,
        }


def safe_divide(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def f1_score(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r) if p + r > 0 else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def truncate_prediction_records(records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = dict(record)
        item["papers"] = list(record.get("papers") or [])[:k]
        output.append(item)
    return output


def evaluate_fixed_k(
    *,
    gold_path: Path,
    e3_path: Path,
    k: int,
) -> dict[str, Any]:
    """
    Evaluate the frozen Day8 E3 ranking at a fixed output cutoff K.

    Important:
    load_gold() / load_predictions() perform ScholarPath's canonical
    deserialization and convert paper dictionaries into PaperRecord objects.
    We must truncate only after canonical loading; otherwise match_papers()
    receives raw dict objects instead of PaperRecord instances.
    """
    gold_records = load_gold(gold_path)
    prediction_records = load_predictions(e3_path)

    truncated_predictions: dict[str, dict[str, Any]] = {}

    for qid, record in prediction_records.items():
        truncated_record = dict(record)
        truncated_record["papers"] = list(record.get("papers") or [])[:k]
        truncated_predictions[qid] = truncated_record

    result = evaluate_records(
        gold_records=gold_records,
        prediction_records=truncated_predictions,
        config=EvaluationConfig(
            mode="strict",
            recall_at=(k,),
            deduplicate=True,
        ),
    )

    summary = result.summary()

    counts = summary.get("counts") or {}
    macro = summary.get("macro") or {}
    micro = summary.get("micro") or {}

    return {
        "k": k,
        "query_count": int(summary.get("query_count", 0)),
        "tp": int(counts.get("tp", 0)),
        "fp": int(counts.get("fp", 0)),
        "fn": int(counts.get("fn", 0)),
        "macro_precision": float(
            macro.get("precision", 0.0)
        ),
        "macro_recall": float(
            macro.get("recall", 0.0)
        ),
        "macro_f1": float(
            macro.get("f1", 0.0)
        ),
        "micro_precision": float(
            micro.get("precision", 0.0)
        ),
        "micro_recall": float(
            micro.get("recall", 0.0)
        ),
        "micro_f1": float(
            micro.get("f1", 0.0)
        ),
        "zero_hit_queries": sum(
            1
            for item in result.per_query
            if item.tp == 0
        ),
        "returned_papers": sum(
            len(
                record.get("papers") or []
            )
            for record in truncated_predictions.values()
        ),
    }


def query_oracle(*, qid: str, question: str, gold: list[PaperRecord], predictions: list[PaperRecord], max_k: int) -> QueryOracle:
    deduped, _ = deduplicate_predictions(predictions)
    usable = deduped[: min(len(deduped), max_k)]
    full_match = match_papers(predictions=usable, gold=gold, mode="strict")
    hit_ranks = sorted(pair.prediction_index + 1 for pair in full_match.pairs)

    best_k = 1 if usable else 0
    best_f1 = -1.0
    best_precision = best_recall = 0.0
    best_tp = 0
    top20_f1 = 0.0
    top20_tp = 0

    for k in range(1, len(usable) + 1):
        m = match_papers(predictions=usable[:k], gold=gold, mode="strict")
        tp = m.tp
        p = safe_divide(tp, k)
        r = safe_divide(tp, len(gold))
        f1 = f1_score(p, r)
        if k == min(20, len(usable)):
            top20_f1 = f1
            top20_tp = tp
        if f1 > best_f1 + 1e-15:
            best_k, best_f1, best_precision, best_recall, best_tp = k, f1, p, r, tp

    if not usable:
        best_f1 = 0.0

    return QueryOracle(
        qid=qid,
        question=question,
        gold_count=len(gold),
        candidate_count=len(usable),
        best_k=best_k,
        best_f1=best_f1,
        best_precision=best_precision,
        best_recall=best_recall,
        best_tp=best_tp,
        top20_f1=top20_f1,
        top20_tp=top20_tp,
        first_hit_rank=hit_ranks[0] if hit_ranks else None,
        last_hit_rank=hit_ranks[-1] if hit_ranks else None,
        hit_ranks=hit_ranks,
    )


def oracle_bucket(k: int) -> str:
    if k <= 5: return "1-5"
    if k <= 10: return "6-10"
    if k <= 20: return "11-20"
    if k <= 50: return "21-50"
    return "51-100"


def build_oracle_summary(rows: list[QueryOracle], best_fixed: dict[str, Any]) -> dict[str, Any]:
    buckets = {"1-5":0,"6-10":0,"11-20":0,"21-50":0,"51-100":0}
    for row in rows:
        buckets[oracle_bucket(row.best_k)] += 1
    ks = [row.best_k for row in rows]
    oracle_macro_f1 = fmean(row.best_f1 for row in rows) if rows else 0.0
    top20_macro_f1 = fmean(row.top20_f1 for row in rows) if rows else 0.0
    return {
        "schema_version": "day13.oracle-k.v1",
        "query_count": len(rows),
        "oracle_macro_precision": fmean(row.best_precision for row in rows) if rows else 0.0,
        "oracle_macro_recall": fmean(row.best_recall for row in rows) if rows else 0.0,
        "oracle_macro_f1": oracle_macro_f1,
        "current_top20_macro_f1_recomputed": top20_macro_f1,
        "best_fixed_k": int(best_fixed["k"]),
        "best_fixed_macro_f1": float(best_fixed["macro_f1"]),
        "oracle_macro_f1_gain_vs_top20": oracle_macro_f1 - top20_macro_f1,
        "oracle_macro_f1_gain_vs_best_fixed": oracle_macro_f1 - float(best_fixed["macro_f1"]),
        "oracle_best_k_distribution": buckets,
        "oracle_best_k_min": min(ks) if ks else 0,
        "oracle_best_k_median": median(ks) if ks else 0,
        "oracle_best_k_mean": fmean(ks) if ks else 0.0,
        "oracle_best_k_max": max(ks) if ks else 0,
        "queries_with_no_strict_hit_in_top100": sum(1 for row in rows if not row.hit_ranks),
        "interpretation_boundary": "Oracle-K uses gold per query and is analysis-only; it is not a production selector or competition score.",
    }


def render_report(*, global_rows, global_summary, oracle_summary) -> str:
    lines = [
        "# Day13-1 Global-K + Oracle-K F1 Diagnostic","",
        "## Experiment Boundary","",
        "- API calls: **0**","- LLM calls: **0**","- Day8 E3 ranking modified: **No**",
        "- Gold usage: **offline evaluation/analysis only**","",
        "## Global-K Curve","",
        "| K | TP | FP | FN | Macro P | Macro R | Macro F1 | Micro F1 | Zero-hit Queries |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(f"| {row['k']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['macro_precision']:.6f} | {row['macro_recall']:.6f} | {row['macro_f1']:.6f} | {row['micro_f1']:.6f} | {row['zero_hit_queries']} |")
    best = global_summary["best_fixed"]
    top20 = global_summary["top20_reference"]
    lines += [
        "","## Fixed-K Finding","",
        f"- Best fixed K: **{best['k']}**",
        f"- Best fixed Macro-F1: **{best['macro_f1']:.6f}**",
        f"- Current Top20 Macro-F1: **{top20['macro_f1']:.6f}**",
        f"- Fixed-K gain vs Top20: **{global_summary['best_fixed_macro_f1_gain_vs_top20']:+.6f}**",
        "","## Oracle-K Diagnostic","",
        f"- Oracle adaptive Macro-F1: **{oracle_summary['oracle_macro_f1']:.6f}**",
        f"- Oracle gain vs best fixed K: **{oracle_summary['oracle_macro_f1_gain_vs_best_fixed']:+.6f}**",
        f"- Queries with no strict hit in Top100: **{oracle_summary['queries_with_no_strict_hit_in_top100']}**","",
        "Oracle best-K distribution:","",
    ]
    for bucket,count in oracle_summary["oracle_best_k_distribution"].items():
        lines.append(f"- {bucket}: **{count}** queries")
    lines += ["","## Decision Guidance",""]
    adaptive_gap = oracle_summary["oracle_macro_f1_gain_vs_best_fixed"]
    fixed_gain = global_summary["best_fixed_macro_f1_gain_vs_top20"]
    if adaptive_gap >= 0.02:
        lines.append("There is a **material adaptive-cutoff headroom signal**. A Day13-2 gold-free selector/cutoff model is worth investigating with query-level cross-validation.")
    elif fixed_gain > 0.005:
        lines.append("Most available gain appears to come from a better **global fixed K** rather than a complex adaptive selector.")
    else:
        lines.append("Selection/cutoff headroom appears limited under the current strict protocol; further selector work should be deprioritized unless competition output rules change.")
    lines += ["","## Reporting Boundary","",
              "These F1 values use ScholarPath's internal strict identity protocol. The competition has not published its complete F1 implementation, so these values are for iteration and relative comparison only and must not be reported as competition score.",""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--e3", default=str(DEFAULT_E3))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--oracle-max-k", type=int, default=100)
    args = parser.parse_args()

    gold_path, e3_path, output_dir = Path(args.gold), Path(args.e3), Path(args.output_dir)
    if not gold_path.is_file() or not e3_path.is_file():
        print("[Day13-1] Missing required input.", file=sys.stderr)
        return 2

    ks = sorted({k for k in args.k if 1 <= k <= 100})
    if 20 not in ks: ks.append(20); ks.sort()

    print("[Day13-1] API calls = 0")
    print("[Day13-1] LLM calls = 0")
    print("[Day13-1] ranking modified = False")

    global_rows = []
    for k in ks:
        row = evaluate_fixed_k(gold_path=gold_path, e3_path=e3_path, k=k)
        global_rows.append(row)
        print(f"[Global-K] K={k:>3} TP={row['tp']:>3} MacroF1={row['macro_f1']:.6f} MacroP={row['macro_precision']:.6f} MacroR={row['macro_recall']:.6f}")

    best_fixed = max(global_rows, key=lambda row: (float(row["macro_f1"]), -int(row["k"])))
    top20 = next(row for row in global_rows if row["k"] == 20)
    global_summary = {
        "schema_version": "day13.global-k.v1",
        "evaluation_protocol": {
            "matching_mode":"strict","deduplicate":True,"ranking_source":str(e3_path),
            "gold_usage":"offline_evaluation_only","api_calls":0,"llm_calls":0,"ranking_modified":False,
        },
        "k_values": ks,
        "best_fixed": best_fixed,
        "top20_reference": top20,
        "best_fixed_macro_f1_gain_vs_top20": float(best_fixed["macro_f1"]) - float(top20["macro_f1"]),
    }

    gold_records = load_gold(gold_path)
    prediction_records = load_predictions(e3_path)
    oracle_rows = []
    for qid in sorted(gold_records):
        g = gold_records[qid]
        p = prediction_records[qid]
        oracle_rows.append(query_oracle(
            qid=qid,
            question=str(p.get("question") or g.get("question") or ""),
            gold=list(g.get("papers") or []),
            predictions=list(p.get("papers") or []),
            max_k=min(max(1,args.oracle_max_k),100),
        ))
    oracle_summary = build_oracle_summary(oracle_rows, best_fixed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"global_k_curve.csv", global_rows, [
        "k","query_count","tp","fp","fn","macro_precision","macro_recall","macro_f1",
        "micro_precision","micro_recall","micro_f1","zero_hit_queries","returned_papers"
    ])
    write_json(output_dir/"global_k_summary.json", global_summary)
    write_jsonl(output_dir/"oracle_k_per_query.jsonl", [x.to_dict() for x in oracle_rows])
    write_json(output_dir/"oracle_k_summary.json", oracle_summary)
    (output_dir/"day13_f1_diagnostic_report.md").write_text(
        render_report(global_rows=global_rows, global_summary=global_summary, oracle_summary=oracle_summary) + "\n",
        encoding="utf-8",
    )

    print("\n=== Day13-1 Summary ===")
    print("Best fixed K:", best_fixed["k"], "MacroF1=", round(float(best_fixed["macro_f1"]),6))
    print("Top20 MacroF1=", round(float(top20["macro_f1"]),6))
    print("Oracle MacroF1=", round(float(oracle_summary["oracle_macro_f1"]),6))
    print("Oracle gain vs best fixed=", round(float(oracle_summary["oracle_macro_f1_gain_vs_best_fixed"]),6))
    print("No-hit queries in Top100=", oracle_summary["queries_with_no_strict_hit_in_top100"])
    print("[OUTPUT]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
