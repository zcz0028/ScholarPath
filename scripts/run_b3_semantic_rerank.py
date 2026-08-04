from __future__ import annotations

import argparse
import json
import sys
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
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.semantic_rerank import (
    annotate_and_semantic_rerank_record,
    make_semantic_config,
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


def read_benchmark(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_no}: expected JSON object.")
            if value.get("qid") is None:
                raise ValueError(f"Line {line_no}: missing qid.")
            records.append(value)
    return records


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
        description="Run B3 lightweight semantic reranker on existing candidates."
    )
    parser.add_argument("--benchmark", default="data/raw/RealScholarQuery/test.jsonl")
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--candidate-dir", default="outputs/b2_1_constraint_rerank_hybrid")
    parser.add_argument("--candidate-file", default="predictions_top100.jsonl")
    parser.add_argument("--output-dir", default="outputs/b3_semantic_hybrid")
    parser.add_argument(
        "--rerank-mode",
        choices=["conservative", "hybrid", "semantic_first"],
        default="hybrid",
    )
    parser.add_argument("--max-constraints", type=int, default=6)
    parser.add_argument("--max-subqueries", type=int, default=6)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    return parser


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

    benchmark_records = read_benchmark(args.benchmark)
    benchmark_by_qid = {str(item.get("qid")): item for item in benchmark_records}

    candidate_records = read_jsonl(candidate_path)
    config = make_semantic_config(args.rerank_mode)
    decomposer = ConstraintDecomposer(
        max_constraints=args.max_constraints,
        max_subqueries=args.max_subqueries,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reranked_records_top100: list[dict[str, Any]] = []
    decomposition_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []

    for record in candidate_records:
        qid = str(record.get("qid"))
        benchmark = benchmark_by_qid.get(qid, {})
        question = str(benchmark.get("question") or record.get("question") or "")
        decomposition = decomposer.decompose(question)
        reranked = annotate_and_semantic_rerank_record(record, decomposition, config)
        reranked_records_top100.append(reranked)
        decomposition_records.append({"qid": qid, **decomposition.to_dict()})

        papers = reranked.get("papers") or []
        semantic_scores = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            try:
                semantic_scores.append(float(raw_meta(paper).get("b3_semantic_score", 0.0)))
            except (TypeError, ValueError):
                pass
        stats_records.append(
            {
                "qid": qid,
                "question": question,
                "constraint_count": len(decomposition.constraints),
                "candidate_count": len(papers),
                "mean_semantic_score": (
                    sum(semantic_scores) / len(semantic_scores)
                    if semantic_scores
                    else 0.0
                ),
                "max_semantic_score": max(semantic_scores) if semantic_scores else 0.0,
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
            for item in reranked_records_top100
        ]
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records_k)
        prediction_paths[k] = path

    write_jsonl(output_dir / "raw_predictions_top100.jsonl", reranked_records_top100)
    write_jsonl(output_dir / "query_decompositions.jsonl", decomposition_records)
    write_jsonl(output_dir / "semantic_rerank_stats.jsonl", stats_records)

    evaluation_summaries = run_evaluations(
        gold_path=args.gold,
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )

    summary = {
        "baseline": "B3",
        "strategy": "lightweight_semantic_rerank_on_existing_candidates",
        "candidate_dir": args.candidate_dir,
        "candidate_file": args.candidate_file,
        "rerank_mode": args.rerank_mode,
        "rerank_config": {
            "mode": config.mode,
            "rank_weight": config.rank_weight,
            "b2_weight": config.b2_weight,
            "semantic_weight": config.semantic_weight,
            "title_boost": config.title_boost,
            "core_constraint_bonus": config.core_constraint_bonus,
            "low_evidence_penalty": config.low_evidence_penalty,
            "strong_id_bonus": config.strong_id_bonus,
        },
        "queries": {
            "total": len(reranked_records_top100),
            "succeeded": len(reranked_records_top100),
            "failed": 0,
        },
        "retrieval": {
            "actual_api_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "raw_results": 0,
            "response_bytes": 0,
            "note": "B3 reranks existing candidates and does not call OpenAlex.",
        },
        "tokens": {
            "input": 0,
            "output": 0,
            "total": 0,
            "note": "B3.0 uses deterministic lightweight semantic features only.",
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
    print(f"[OK] Queries: {len(reranked_records_top100)}")
    print(f"[OK] Rerank mode: {args.rerank_mode}")
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
