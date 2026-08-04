from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from scholarpath.selector.selector import (
    annotate_and_select_record,
    make_selector_config,
    raw_meta,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run B5 Selector with relevance reason tags."
    )
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--candidate-dir", default="outputs/b3_semantic_first")
    parser.add_argument("--candidate-file", default="predictions_top100.jsonl")
    parser.add_argument("--output-dir", default="outputs/b5_selector_balanced")
    parser.add_argument(
        "--selector-mode",
        choices=["ranking", "balanced", "precision", "recall_safe"],
        default="balanced",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--sample-size", type=int, default=30)
    return parser


def label_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    per_query_counts: list[int] = []

    for record in records:
        papers = record.get("papers") or []
        if not isinstance(papers, list):
            papers = []
        per_query_counts.append(len(papers))
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            raw = raw_meta(paper)
            label_counter[str(raw.get("b5_relevance_label") or "unknown")] += 1
            for tag in raw.get("b5_reason_tags") or []:
                tag_counter[str(tag)] += 1
            for tag in raw.get("b5_missing_tags") or []:
                missing_counter[str(tag)] += 1

    return {
        "labels": dict(label_counter),
        "reason_tags": dict(tag_counter),
        "missing_tags": dict(missing_counter),
        "per_query_output_count": {
            "min": min(per_query_counts) if per_query_counts else 0,
            "max": max(per_query_counts) if per_query_counts else 0,
            "mean": sum(per_query_counts) / len(per_query_counts) if per_query_counts else 0.0,
        },
    }


def write_reason_samples(path: Path, records: list[dict[str, Any]], *, sample_size: int) -> None:
    lines: list[str] = ["# B5 Selector Reason Samples", ""]
    count = 0
    for record in records:
        qid = record.get("qid")
        question = record.get("question")
        papers = record.get("papers") or []
        if not papers:
            continue
        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"Query: {question}")
        lines.append("")
        for paper in papers[:3]:
            if not isinstance(paper, dict):
                continue
            raw = raw_meta(paper)
            title = paper.get("title") or "<no title>"
            label = raw.get("b5_relevance_label")
            score = raw.get("b5_selector_score")
            reason = raw.get("b5_reason_text")
            lines.append(f"- **{title}**")
            lines.append(f"  - Label: `{label}`")
            lines.append(f"  - Score: `{score}`")
            lines.append(f"  - Reason: {reason}")
            count += 1
            if count >= sample_size:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2

    candidate_path = Path(args.candidate_dir) / args.candidate_file
    if not candidate_path.exists():
        print(f"[ERROR] Missing candidate file: {candidate_path}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_selector_config(args.selector_mode)
    candidate_records = read_jsonl(candidate_path)

    selected_top100: list[dict[str, Any]] = [
        annotate_and_select_record(record, config)
        for record in candidate_records
    ]

    prediction_paths: dict[int, Path] = {}
    for k in top_ks:
        records_k = [
            {
                "qid": record.get("qid"),
                "question": record.get("question"),
                "papers": (record.get("papers") or [])[:k],
            }
            for record in selected_top100
        ]
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records_k)
        prediction_paths[k] = path

    write_jsonl(output_dir / "raw_predictions_top100.jsonl", selected_top100)

    stats = label_stats(selected_top100)
    (output_dir / "selector_label_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_reason_samples(
        output_dir / "selector_reason_samples.md",
        selected_top100,
        sample_size=args.sample_size,
    )

    evaluation_summaries = run_evaluations(
        gold_path=args.gold,
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )

    summary = {
        "baseline": "B5",
        "strategy": "selector_filtering_with_relevance_reason_tags",
        "candidate_dir": args.candidate_dir,
        "candidate_file": args.candidate_file,
        "selector_mode": args.selector_mode,
        "selector_config": {
            "mode": config.mode,
            "semantic_weight": config.semantic_weight,
            "rank_weight": config.rank_weight,
            "b2_weight": config.b2_weight,
            "core_weight": config.core_weight,
            "title_weight": config.title_weight,
            "high_threshold": config.high_threshold,
            "partial_threshold": config.partial_threshold,
            "weak_threshold": config.weak_threshold,
            "min_keep_per_query": config.min_keep_per_query,
            "max_output_per_query": config.max_output_per_query,
        },
        "queries": {
            "total": len(selected_top100),
            "succeeded": len(selected_top100),
            "failed": 0,
        },
        "retrieval": {
            "actual_api_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "raw_results": 0,
            "response_bytes": 0,
            "note": "B5 selector reads existing B3 candidates and does not call OpenAlex.",
        },
        "tokens": {
            "input": 0,
            "output": 0,
            "total": 0,
            "note": "B5 selector uses deterministic scoring and reason tags only.",
        },
        "estimated_cost_usd": 0.0,
        "label_stats": stats,
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

    focus20 = evaluation_summaries.get("top20_strict", {})
    focus50 = evaluation_summaries.get("top50_strict", {})
    print(f"[OK] Queries: {len(selected_top100)}")
    print(f"[OK] Selector mode: {args.selector_mode}")
    print(
        "[OK] Top20 strict: "
        f"MacroF1={focus20.get('macro', {}).get('f1')} "
        f"TP={focus20.get('counts', {}).get('tp')} "
        f"FP={focus20.get('counts', {}).get('fp')} "
        f"FN={focus20.get('counts', {}).get('fn')}"
    )
    print(
        "[OK] Top50 strict: "
        f"MacroF1={focus50.get('macro', {}).get('f1')} "
        f"TP={focus50.get('counts', {}).get('tp')} "
        f"FP={focus50.get('counts', {}).get('fp')} "
        f"FN={focus50.get('counts', {}).get('fn')}"
    )
    print("[OK] API calls: 0")
    print("[OK] Total tokens: 0")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
