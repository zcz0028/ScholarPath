from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import (
    EvaluationConfig,
    evaluate_records,
    write_evaluation_outputs,
)
from scholarpath.fusion import FusionConfig, fuse_prediction_records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                records.append(value)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse B0 stable ranking with B1.2 high-confidence supplements."
    )
    parser.add_argument("--b0-dir", default="outputs/b0_openalex")
    parser.add_argument("--b1-dir", default="outputs/b1_2_raw_anchor_q3_p40")
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--output-dir", default="outputs/b1_3_fusion_45_5")
    parser.add_argument("--base-keep-top", type=int, default=45)
    parser.add_argument("--supplement-slots", type=int, default=5)
    parser.add_argument("--max-output", type=int, default=100)
    parser.add_argument("--min-occurrence", type=int, default=2)
    parser.add_argument("--max-anchor-rank", type=int, default=50)
    parser.add_argument("--min-raw-anchor-score", type=float, default=0.025)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    return parser


def run_evaluations(
    *,
    gold_path: str | Path,
    prediction_paths: dict[int, Path],
    output_dir: Path,
) -> dict[str, Any]:
    gold_records = load_gold(gold_path)
    summaries: dict[str, Any] = {}

    for k, prediction_path in sorted(prediction_paths.items()):
        predictions = load_predictions(prediction_path)
        for mode in ("strict", "pasa_title"):
            result = evaluate_records(
                gold_records=gold_records,
                prediction_records=predictions,
                config=EvaluationConfig(
                    mode=mode,
                    recall_at=(20, 50, 100),
                    deduplicate=True,
                ),
            )
            eval_dir = output_dir / "evaluation" / f"top{k}" / mode
            write_evaluation_outputs(result, eval_dir)
            summaries[f"top{k}_{mode}"] = result.summary()
    return summaries


def main() -> int:
    args = build_parser().parse_args()

    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2

    config = FusionConfig(
        base_keep_top=args.base_keep_top,
        supplement_slots=args.supplement_slots,
        max_output=args.max_output,
        min_occurrence=args.min_occurrence,
        max_anchor_rank=args.max_anchor_rank,
        min_raw_anchor_score=args.min_raw_anchor_score,
    )
    config_dict = asdict(config)

    b0_path = Path(args.b0_dir) / "predictions_top100.jsonl"
    b1_path = Path(args.b1_dir) / "predictions_top100.jsonl"
    if not b0_path.exists():
        print(f"[ERROR] Missing B0 predictions: {b0_path}")
        return 2
    if not b1_path.exists():
        print(f"[ERROR] Missing B1 predictions: {b1_path}")
        return 2

    b0_records = read_jsonl(b0_path)
    b1_records = read_jsonl(b1_path)
    b0_by_qid = {str(item.get("qid")): item for item in b0_records}
    b1_by_qid = {str(item.get("qid")): item for item in b1_records}

    missing = sorted(set(b0_by_qid) ^ set(b1_by_qid))
    if missing:
        print(f"[ERROR] QID mismatch between B0 and B1 predictions. Missing/different: {missing[:5]}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_records_top100: list[dict[str, Any]] = []
    fusion_logs: list[dict[str, Any]] = []

    aggregate_stats = {
        "base_kept": 0,
        "supplement_inserted": 0,
        "base_filled": 0,
        "supplement_filled": 0,
        "duplicates_skipped": 0,
        "low_confidence_skipped": 0,
    }

    for qid in sorted(b0_by_qid):
        fused, stats = fuse_prediction_records(b0_by_qid[qid], b1_by_qid[qid], config)
        fused_records_top100.append(fused)
        stat_dict = stats.to_dict()
        for key in aggregate_stats:
            aggregate_stats[key] += int(stat_dict.get(key, 0))
        fusion_logs.append(
            {
                "qid": qid,
                "question": fused.get("question"),
                "stats": stat_dict,
                "config": config_dict,
            }
        )

    prediction_paths: dict[int, Path] = {}
    for k in top_ks:
        records_k = [
            {
                "qid": item.get("qid"),
                "question": item.get("question"),
                "papers": (item.get("papers") or [])[:k],
            }
            for item in fused_records_top100
        ]
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records_k)
        prediction_paths[k] = path

    write_jsonl(output_dir / "raw_predictions_top100.jsonl", fused_records_top100)
    write_jsonl(output_dir / "fusion_logs.jsonl", fusion_logs)

    evaluation_summaries = run_evaluations(
        gold_path=args.gold,
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )

    summary = {
        "baseline": "B1.3",
        "strategy": "b0_fallback_b1_high_confidence_supplement",
        "source": {
            "b0_dir": args.b0_dir,
            "b1_dir": args.b1_dir,
        },
        "fusion_config": config_dict,
        "queries": {
            "total": len(fused_records_top100),
            "succeeded": len(fused_records_top100),
            "failed": 0,
        },
        "retrieval": {
            "actual_api_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "raw_results": 0,
            "response_bytes": 0,
            "note": "B1.3 only fuses existing B0/B1 prediction files and does not call OpenAlex.",
        },
        "tokens": {
            "input": 0,
            "output": 0,
            "total": 0,
            "note": "B1.3 uses no LLM.",
        },
        "estimated_cost_usd": 0.0,
        "fusion_stats": aggregate_stats,
        "top_k_outputs": top_ks,
        "prediction_files": {
            str(k): str(path) for k, path in prediction_paths.items()
        },
        "evaluation": evaluation_summaries,
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    focus = evaluation_summaries.get("top50_strict", {})
    counts = focus.get("counts", {})
    macro = focus.get("macro", {})
    print(f"[OK] Queries: {len(fused_records_top100)}")
    print(f"[OK] Fusion stats: {json.dumps(aggregate_stats, ensure_ascii=False)}")
    print(
        "[OK] Top50 strict: "
        f"MacroF1={macro.get('f1')} "
        f"TP={counts.get('tp')} FP={counts.get('fp')} FN={counts.get('fn')}"
    )
    print("[OK] API calls: 0")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
