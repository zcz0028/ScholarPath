from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from scholarpath.guard.constraint_guard import (
    apply_guard_to_record,
    extract_guard_constraints,
    make_guard_config,
    raw_meta,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
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


def build_guard_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counter: Counter[str] = Counter()
    violation_counter: Counter[str] = Counter()
    evidence_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    constraint_type_counter: Counter[str] = Counter()
    output_counts: list[int] = []

    for record in records:
        papers = record.get("papers") or []
        output_counts.append(len(papers) if isinstance(papers, list) else 0)
        guard = record.get("guard") if isinstance(record.get("guard"), dict) else {}
        for constraint in guard.get("constraints") or []:
            if isinstance(constraint, dict):
                constraint_type_counter[str(constraint.get("constraint_type") or "unknown")] += 1

        if not isinstance(papers, list):
            continue
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            raw = raw_meta(paper)
            decision_counter[str(raw.get("b5_1_guard_decision") or "unknown")] += 1
            for tag in raw.get("b5_1_violation_tags") or []:
                violation_counter[str(tag)] += 1
            for tag in raw.get("b5_1_evidence_tags") or []:
                evidence_counter[str(tag)] += 1
            for item in raw.get("b5_1_missing_constraints") or []:
                missing_counter[str(item)] += 1

    return {
        "decisions": dict(decision_counter),
        "violations": dict(violation_counter),
        "evidence_tags": dict(evidence_counter),
        "missing_constraints": dict(missing_counter),
        "constraint_types": dict(constraint_type_counter),
        "per_query_output_count": {
            "min": min(output_counts) if output_counts else 0,
            "max": max(output_counts) if output_counts else 0,
            "mean": sum(output_counts) / len(output_counts) if output_counts else 0.0,
        },
    }


def write_reason_samples(path: Path, records: list[dict[str, Any]], *, sample_size: int) -> None:
    lines: list[str] = ["# B5.1 Constraint Guard Reason Samples", ""]
    count = 0
    for record in records:
        qid = record.get("qid")
        question = record.get("question")
        guard = record.get("guard") if isinstance(record.get("guard"), dict) else {}
        constraints = guard.get("constraints") or []
        papers = record.get("papers") or []
        if not isinstance(papers, list):
            papers = []
        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"Query: {question}")
        lines.append("")
        if constraints:
            lines.append("Detected constraints:")
            for constraint in constraints:
                if isinstance(constraint, dict):
                    lines.append(
                        f"- `{constraint.get('constraint_type')}` {constraint.get('label')} "
                        f"(severity={constraint.get('severity')})"
                    )
        else:
            lines.append("Detected constraints: none")
        lines.append("")

        for paper in papers[:4]:
            if not isinstance(paper, dict):
                continue
            raw = raw_meta(paper)
            title = paper.get("title") or "<no title>"
            lines.append(f"- **{title}**")
            lines.append(f"  - Decision: `{raw.get('b5_1_guard_decision')}`")
            lines.append(f"  - Guard score: `{raw.get('b5_1_guard_score')}`")
            lines.append(f"  - Final score: `{raw.get('b5_1_final_score')}`")
            lines.append(f"  - Reason: {raw.get('b5_1_guard_reason')}")
            if raw.get("b5_1_violation_tags"):
                lines.append(f"  - Violations: `{raw.get('b5_1_violation_tags')}`")
            count += 1
            if count >= sample_size:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
                return
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run B5.1 hard constraint guard on candidate predictions."
    )
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--candidate-dir", default="outputs/b4_fusion_semantic")
    parser.add_argument("--candidate-file", default="predictions_top100.jsonl")
    parser.add_argument("--output-dir", default="outputs/b5_1_guard_balanced")
    parser.add_argument(
        "--guard-mode",
        choices=["annotate", "balanced", "precision", "recall_safe"],
        default="balanced",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--sample-size", type=int, default=40)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidate_path = Path(args.candidate_dir) / args.candidate_file
    if not candidate_path.exists():
        print(f"[ERROR] Missing candidate file: {candidate_path}")
        return 2

    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_guard_config(args.guard_mode)
    records = read_jsonl(candidate_path)

    guarded_records = [apply_guard_to_record(record, config) for record in records]

    prediction_paths: dict[int, Path] = {}
    for k in top_ks:
        records_k = [
            {
                "qid": record.get("qid"),
                "question": record.get("question"),
                "papers": (record.get("papers") or [])[:k],
            }
            for record in guarded_records
        ]
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records_k)
        prediction_paths[k] = path

    write_jsonl(output_dir / "raw_predictions_top100.jsonl", guarded_records)

    stats = build_guard_stats(guarded_records)
    (output_dir / "guard_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8-sig",
    )
    write_reason_samples(
        output_dir / "guard_reason_samples.md",
        guarded_records,
        sample_size=args.sample_size,
    )

    evaluation_summaries = run_evaluations(
        gold_path=args.gold,
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )

    summary = {
        "baseline": "B5.1",
        "strategy": "hard_constraint_guard_for_academic_search",
        "candidate_dir": args.candidate_dir,
        "candidate_file": args.candidate_file,
        "guard_mode": args.guard_mode,
        "queries": {
            "total": len(guarded_records),
            "succeeded": len(guarded_records),
            "failed": 0,
        },
        "guard": {
            "mode": config.mode,
            "reject_violations": config.reject_violations,
            "filter_rejects": config.filter_rejects,
            "filter_downrank": config.filter_downrank,
            "min_guard_score": config.min_guard_score,
            "max_output_per_query": config.max_output_per_query,
            "min_keep_per_query": config.min_keep_per_query,
            "stats": stats,
        },
        "retrieval": {
            "actual_api_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "raw_results": 0,
            "response_bytes": 0,
            "note": "B5.1 reads existing candidates and applies deterministic hard-constraint rules only.",
        },
        "tokens": {
            "input": 0,
            "output": 0,
            "total": 0,
            "note": "B5.1 does not call any LLM.",
        },
        "estimated_cost_usd": 0.0,
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
    focus100 = evaluation_summaries.get("top100_strict", {})
    print(f"[OK] Queries: {len(guarded_records)}")
    print(f"[OK] Guard mode: {args.guard_mode}")
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
    print(
        "[OK] Top100 strict: "
        f"MacroF1={focus100.get('macro', {}).get('f1')} "
        f"TP={focus100.get('counts', {}).get('tp')} "
        f"FP={focus100.get('counts', {}).get('fp')} "
        f"FN={focus100.get('counts', {}).get('fn')}"
    )
    print("[OK] API calls: 0")
    print("[OK] Total tokens: 0")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
