from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records

DEFAULT_GOLD = PROJECT_ROOT / "data" / "processed" / "realscholarquery_gold.jsonl"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "outputs" / "week2_day4_rescue" / "run_summary.json"
DEFAULT_QE_DIR = PROJECT_ROOT / "outputs" / "week2_day12_query_expansion_retrieval"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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


def baseline_metric(summary: dict[str, Any], k: int) -> dict[str, Any]:
    item = (summary.get("evaluation") or {}).get(f"top{k}_strict") or {}
    counts = item.get("counts") or {}
    macro = item.get("macro") or {}
    micro = item.get("micro") or {}
    return {
        "tp": int(counts.get("tp", 0)),
        "fp": int(counts.get("fp", 0)),
        "fn": int(counts.get("fn", 0)),
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "micro_precision": float(micro.get("precision", 0.0)),
        "micro_recall": float(micro.get("recall", 0.0)),
        "micro_f1": float(micro.get("f1", 0.0)),
        "recall_at_k": float(macro.get(f"recall@{k}", macro.get("recall", 0.0))),
    }


def evaluate_prediction_file(gold_path: Path, prediction_path: Path, k: int):
    gold = load_gold(gold_path)
    preds = load_predictions(prediction_path)
    t0 = time.perf_counter()
    result = evaluate_records(
        gold_records=gold,
        prediction_records=preds,
        config=EvaluationConfig(mode="strict", recall_at=(20, 50, 100), deduplicate=True),
    )
    elapsed = time.perf_counter() - t0
    s = result.summary()
    counts = s.get("counts") or {}
    macro = s.get("macro") or {}
    micro = s.get("micro") or {}
    metric = {
        "tp": int(counts.get("tp", 0)),
        "fp": int(counts.get("fp", 0)),
        "fn": int(counts.get("fn", 0)),
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "micro_precision": float(micro.get("precision", 0.0)),
        "micro_recall": float(micro.get("recall", 0.0)),
        "micro_f1": float(micro.get("f1", 0.0)),
        "recall_at_k": float(macro.get(f"recall@{k}", macro.get("recall", 0.0))),
        "duplicates_removed": int(counts.get("duplicates_removed", 0)),
        "predictions_after_dedup": int(counts.get("predictions_after_dedup", 0)),
    }
    return metric, elapsed


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tp","fp","fn","macro_precision","macro_recall","macro_f1",
        "micro_precision","micro_recall","micro_f1","recall_at_k"
    ]
    return {k: after[k] - before[k] for k in keys}


def aggregate_retrieval_logs(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "query_log_count": len(rows),
        "actual_api_calls": sum(int(r.get("actual_api_calls", 0) or 0) for r in rows),
        "cache_hits": sum(int(r.get("cache_hits", 0) or 0) for r in rows),
        "retries": sum(int(r.get("retries", 0) or 0) for r in rows),
        "estimated_cost_usd": sum(float(r.get("estimated_cost_usd", 0.0) or 0.0) for r in rows),
    }


def build_comparison(
    baseline_summary: dict[str, Any],
    qe_metrics_by_k: dict[str, dict[str, Any]],
    evaluation_seconds_by_k: dict[str, float],
    retrieval_efficiency: dict[str, Any],
    target_query_count: int,
) -> dict[str, Any]:
    metrics_by_k = {}
    for k in (20, 50, 100):
        before = baseline_metric(baseline_summary, k)
        after = qe_metrics_by_k[str(k)]
        metrics_by_k[str(k)] = {
            "baseline": before,
            "qe_r9": after,
            "delta": metric_delta(before, after),
            "evaluation_elapsed_seconds": evaluation_seconds_by_k[str(k)],
        }

    d20 = metrics_by_k["20"]["delta"]
    d100 = metrics_by_k["100"]["delta"]
    checks = {
        "top100_tp_gain_positive": d100["tp"] > 0,
        "top100_recall_gain_positive": d100["macro_recall"] > 0,
        "top100_f1_gain_positive": d100["macro_f1"] > 0,
        "top20_precision_not_materially_degraded": d20["macro_precision"] >= -0.002,
    }
    adopt = all(checks.values())
    added_tp = int(d100["tp"])
    calls = int(retrieval_efficiency.get("actual_api_calls", 0) or 0)
    return {
        "schema_version": "day12.qe-r9.final.v1",
        "experiment": "QE-R9 residual retrieval ablation",
        "scope": "residual_retrieval_failures_only",
        "target_query_count": target_query_count,
        "baseline": "frozen_day4_rescue",
        "matching_mode": "strict",
        "metrics_by_k": metrics_by_k,
        "efficiency": {
            **retrieval_efficiency,
            "api_calls_per_added_top100_tp": (calls / added_tp if added_tp > 0 else None),
        },
        "adoption_checks": checks,
        "adopt_candidate": adopt,
        "decision": "candidate_for_e3_integration" if adopt else "reject_qe_r9_keep_frozen_pipeline",
        "conclusion": (
            "QE-R9 produced a measurable strict Top100 retrieval gain."
            if added_tp > 0
            else "QE-R9 produced no additional strict Top100 true positives over the frozen Day4 pipeline."
        ),
        "reporting_boundaries": [
            "QE-R9 targeted only residual retrieval failures.",
            "Gold labels were used only for offline evaluation after retrieval.",
            "The frozen Day8 E3 artifact was not modified.",
            "A rejected QE result is retained as a negative ablation, not a production feature.",
        ],
    }


def render_report(c: dict[str, Any]) -> str:
    lines = [
        "# Day12-3-3 Final QE-R9 Ablation Report","",
        f"- Residual queries targeted: **{c['target_query_count']}**",
        "- Matching mode: **strict**",
        "- Frozen Day8 E3 modified: **No**","",
        "## Quality Comparison","",
        "| K | Baseline TP | QE-R9 TP | ΔTP | Baseline Recall | QE-R9 Recall | ΔRecall | Baseline F1 | QE-R9 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("20","50","100"):
        r = c["metrics_by_k"][k]
        b,a,d = r["baseline"],r["qe_r9"],r["delta"]
        lines.append(
            f"| {k} | {b['tp']} | {a['tp']} | {int(d['tp']):+d} | "
            f"{b['macro_recall']:.6f} | {a['macro_recall']:.6f} | {d['macro_recall']:+.6f} | "
            f"{b['macro_f1']:.6f} | {a['macro_f1']:.6f} |"
        )
    e = c["efficiency"]
    lines += [
        "","## Efficiency","",
        f"- Actual API calls recorded: **{e['actual_api_calls']}**",
        f"- Cache hits recorded: **{e['cache_hits']}**",
        f"- Retries recorded: **{e['retries']}**",
        f"- API calls per added Top100 TP: **{e['api_calls_per_added_top100_tp'] if e['api_calls_per_added_top100_tp'] is not None else 'N/A (no added TP)'}**",
        "","## Decision","",
    ]
    for k,v in c["adoption_checks"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["",f"**Decision: `{c['decision']}`**","",c["conclusion"],""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    p.add_argument("--qe-dir", default=str(DEFAULT_QE_DIR))
    args = p.parse_args()

    gold_path = Path(args.gold)
    baseline_path = Path(args.baseline_summary)
    qe_dir = Path(args.qe_dir)

    required = [gold_path, baseline_path] + [qe_dir / f"predictions_top{k}.jsonl" for k in (20,50,100)]
    missing = [x for x in required if not x.is_file()]
    if missing:
        print("[Day12-3-3] Missing required files:", file=sys.stderr)
        for x in missing:
            print("  -", x, file=sys.stderr)
        return 2

    baseline = read_json(baseline_path)
    metrics, elapsed = {}, {}
    for k in (20,50,100):
        m,t = evaluate_prediction_file(gold_path, qe_dir / f"predictions_top{k}.jsonl", k)
        metrics[str(k)] = m
        elapsed[str(k)] = t
        print(f"[Day12-3-3] Top{k}: TP={m['tp']} MacroRecall={m['macro_recall']:.6f} MacroF1={m['macro_f1']:.6f} elapsed={t:.3f}s")

    logs_path = qe_dir / "query_logs.jsonl"
    efficiency = aggregate_retrieval_logs(read_jsonl(logs_path) if logs_path.is_file() else [])
    plans_path = qe_dir / "query_plans_qe_r9.jsonl"
    target_count = len(read_jsonl(plans_path)) if plans_path.is_file() else len(read_jsonl(logs_path)) if logs_path.is_file() else 0

    comparison = build_comparison(baseline, metrics, elapsed, efficiency, target_count)
    (qe_dir / "qe_vs_frozen_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (qe_dir / "qe_ablation_report.md").write_text(render_report(comparison) + "\n", encoding="utf-8")

    print()
    print("=== Day12-3-3 Final Decision ===")
    print("top100_tp =", comparison["metrics_by_k"]["100"]["baseline"]["tp"], "->", comparison["metrics_by_k"]["100"]["qe_r9"]["tp"])
    print("decision =", comparison["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
