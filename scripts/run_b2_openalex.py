from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from scholarpath.observability.run_log import build_run_summary
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.constraint_aggregation import aggregate_constraint_hits
from scholarpath.retrieval.dates import parse_benchmark_cutoff
from scholarpath.retrieval.openalex import OpenAlexConfig, OpenAlexError, OpenAlexRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run B2 constraint-aware multi-subquery OpenAlex retrieval."
    )
    parser.add_argument("--benchmark", default="data/raw/RealScholarQuery/test.jsonl")
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--output-dir", default="outputs/b2_openalex")
    parser.add_argument("--per-subquery-page", type=int, default=35)
    parser.add_argument("--max-subqueries", type=int, default=6)
    parser.add_argument("--max-constraints", type=int, default=6)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--cutoff-offset-days", type=int, default=7)
    parser.add_argument("--cache-dir", default="data/cache/openalex")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--search-cost-per-1000", type=float, default=1.0)
    return parser


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
            if not str(value.get("question") or "").strip():
                raise ValueError(f"Line {line_no}: empty question.")
            if value.get("qid") is None:
                raise ValueError(f"Line {line_no}: missing qid.")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_stats(stats_list: list[dict[str, Any]], *, elapsed_ms: float) -> dict[str, Any]:
    errors = [item.get("error") for item in stats_list if item.get("error")]
    return {
        "provider": "openalex",
        "actual_api_calls": sum(int(item.get("actual_api_calls", 0)) for item in stats_list),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in stats_list),
        "retries": sum(int(item.get("retries", 0)) for item in stats_list),
        "http_status": None,
        "response_bytes": sum(int(item.get("response_bytes", 0)) for item in stats_list),
        "request_latency_ms": sum(float(item.get("request_latency_ms", 0.0)) for item in stats_list),
        "end_to_end_latency_ms": elapsed_ms,
        "estimated_api_cost_usd": sum(float(item.get("estimated_api_cost_usd", 0.0)) for item in stats_list),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "rate_limit_headers": {},
        "error": "; ".join(str(item) for item in errors) if errors else None,
    }


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
    if not os.getenv("OPENALEX_API_KEY"):
        print("[ERROR] OPENALEX_API_KEY is not set in this terminal.")
        return 2

    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2
    if args.per_subquery_page < 1 or args.per_subquery_page > 100:
        print("[ERROR] --per-subquery-page must be between 1 and 100.")
        return 2

    benchmark_records = read_benchmark(args.benchmark)
    if args.limit_queries is not None:
        benchmark_records = benchmark_records[: max(0, args.limit_queries)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decomposer = ConstraintDecomposer(
        max_constraints=args.max_constraints,
        max_subqueries=args.max_subqueries,
    )
    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            cache_dir=args.cache_dir,
            search_cost_per_1000_calls_usd=args.search_cost_per_1000,
        )
    )

    predictions_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in top_ks}
    raw_records: list[dict[str, Any]] = []
    query_logs: list[dict[str, Any]] = []
    decomposition_records: list[dict[str, Any]] = []

    run_started = time.perf_counter()

    for index, benchmark_record in enumerate(benchmark_records, start=1):
        qid = str(benchmark_record["qid"])
        question = str(benchmark_record["question"]).strip()
        source_meta = benchmark_record.get("source_meta")
        published_time = (
            source_meta.get("published_time")
            if isinstance(source_meta, dict)
            else None
        )
        cutoff = parse_benchmark_cutoff(published_time, args.cutoff_offset_days)
        decomposition = decomposer.decompose(question)
        decomposition_records.append({"qid": qid, **decomposition.to_dict()})

        print(
            f"[{index}/{len(benchmark_records)}] qid={qid} "
            f"constraints={len(decomposition.constraints)} "
            f"subqueries={len(decomposition.subqueries)} cutoff={cutoff}"
        )

        query_started = time.perf_counter()
        subquery_stats: list[dict[str, Any]] = []
        hits = []
        subquery_logs: list[dict[str, Any]] = []

        for subquery_index, subquery in enumerate(decomposition.subqueries, start=1):
            try:
                result = retriever.search(
                    SearchRequest(
                        query=subquery.text,
                        per_page=args.per_subquery_page,
                        to_publication_date=cutoff,
                    ),
                    refresh_cache=args.refresh_cache,
                )
                papers = result.papers
                stats = result.stats.to_dict()
                error = None
            except (OpenAlexError, ValueError, OSError) as exc:
                papers = []
                stats = (
                    exc.stats.to_dict()
                    if isinstance(exc, OpenAlexError) and exc.stats is not None
                    else {
                        "provider": "openalex",
                        "actual_api_calls": 0,
                        "cache_hits": 0,
                        "retries": 0,
                        "http_status": None,
                        "response_bytes": 0,
                        "request_latency_ms": 0.0,
                        "end_to_end_latency_ms": 0.0,
                        "estimated_api_cost_usd": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "rate_limit_headers": {},
                        "error": str(exc),
                    }
                )
                error = str(exc)
                print(f"[ERROR] qid={qid} subquery={subquery_index}: {error}")
                if not args.continue_on_error:
                    return 2

            subquery_stats.append(
                {
                    **stats,
                    "raw_result_count": len(papers),
                    "error": error or stats.get("error"),
                }
            )
            subquery_logs.append(
                {
                    "subquery_index": subquery_index,
                    "subquery_type": subquery.subquery_type,
                    "query": subquery.text,
                    "constraint_ids": subquery.constraint_ids,
                    "reason": subquery.reason,
                    "returned_papers": len(papers),
                    "error": error or stats.get("error"),
                    "stats": stats,
                }
            )

            for rank, paper in enumerate(papers, start=1):
                hits.append(
                    (
                        paper,
                        rank,
                        subquery.subquery_type,
                        subquery.text,
                        subquery.constraint_ids,
                    )
                )

        elapsed_ms = (time.perf_counter() - query_started) * 1000.0
        merged_stats = merge_stats(subquery_stats, elapsed_ms=elapsed_ms)
        merged_papers = aggregate_constraint_hits(
            hits,
            constraints=decomposition.constraints,
        )
        paper_dicts = [paper.to_dict() for paper in merged_papers[: max(top_ks)]]

        raw_records.append(
            {
                "qid": qid,
                "question": question,
                "published_time": published_time,
                "cutoff_date": cutoff,
                "decomposition": decomposition.to_dict(),
                "papers": paper_dicts,
            }
        )

        for k in top_ks:
            predictions_by_k[k].append(
                {
                    "qid": qid,
                    "question": question,
                    "papers": paper_dicts[:k],
                }
            )

        query_logs.append(
            {
                "qid": qid,
                "query_index": index,
                "query": question,
                "published_time": published_time,
                "cutoff_date": cutoff,
                "constraint_count": len(decomposition.constraints),
                "subquery_count": len(decomposition.subqueries),
                "requested_results_per_subquery": args.per_subquery_page,
                "returned_papers": len(merged_papers),
                "raw_result_count": sum(int(item.get("raw_result_count", 0)) for item in subquery_stats),
                "subquery_logs": subquery_logs,
                **merged_stats,
            }
        )

    wall_clock_ms = (time.perf_counter() - run_started) * 1000.0

    write_jsonl(output_dir / "raw_predictions_top100.jsonl", raw_records)
    write_jsonl(output_dir / "query_logs.jsonl", query_logs)
    write_jsonl(output_dir / "query_decompositions.jsonl", decomposition_records)

    prediction_paths: dict[int, Path] = {}
    for k, records in predictions_by_k.items():
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records)
        prediction_paths[k] = path

    summary = build_run_summary(query_logs, wall_clock_ms=wall_clock_ms)
    summary["tokens"]["note"] = "B2 uses deterministic constraint decomposition only, so token usage remains zero."
    summary.update(
        {
            "baseline": "B2",
            "provider": "openalex",
            "query_strategy": "constraint_aware_multi_subquery_recall",
            "cutoff_offset_days": args.cutoff_offset_days,
            "per_subquery_page": args.per_subquery_page,
            "max_subqueries": args.max_subqueries,
            "max_constraints": args.max_constraints,
            "top_k_outputs": top_ks,
            "prediction_files": {
                str(k): str(path) for k, path in prediction_paths.items()
            },
        }
    )

    gold_path = Path(args.gold)
    if gold_path.exists() and args.limit_queries is None:
        summary["evaluation"] = run_evaluations(
            gold_path=gold_path,
            prediction_paths=prediction_paths,
            output_dir=output_dir,
        )
    else:
        summary["evaluation_skipped"] = "Gold file missing or --limit-queries was used."

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Queries: {summary['queries']['total']}")
    print(f"[OK] Actual API calls: {summary['retrieval']['actual_api_calls']}")
    print(f"[OK] Cache hits: {summary['retrieval']['cache_hits']}")
    print(f"[OK] Retries: {summary['retrieval']['retries']}")
    print(f"[OK] Total tokens: {summary['tokens']['total']}")
    print(f"[OK] Estimated API cost: ${summary['estimated_cost_usd']:.6f}")
    print(f"[OK] Wall-clock latency: {summary['latency_ms']['wall_clock']:.2f} ms")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
