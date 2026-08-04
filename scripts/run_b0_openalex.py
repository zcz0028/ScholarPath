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
from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.dates import parse_benchmark_cutoff
from scholarpath.retrieval.openalex import (
    OpenAlexConfig,
    OpenAlexError,
    OpenAlexRetriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B0 raw-query OpenAlex retrieval baseline."
    )
    parser.add_argument(
        "--benchmark",
        default="data/raw/RealScholarQuery/test.jsonl",
    )
    parser.add_argument(
        "--gold",
        default="data/processed/realscholarquery_gold.jsonl",
    )
    parser.add_argument("--output-dir", default="outputs/b0_openalex")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[20, 50, 100],
    )
    parser.add_argument(
        "--cutoff-offset-days",
        type=int,
        default=7,
        help=(
            "Subtract this many days from source_meta.published_time. "
            "Default 7 follows the PaSa public runtime."
        ),
    )
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--cache-dir", default="data/cache/openalex")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--search-cost-per-1000",
        type=float,
        default=1.0,
        help="Estimated OpenAlex search price in USD per 1000 calls.",
    )
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
    max_k = max(top_ks)
    if args.per_page < max_k:
        print("[ERROR] --per-page must be >= the largest --top-k.")
        return 2
    if args.per_page > 100:
        print("[ERROR] OpenAlex --per-page cannot exceed 100 in B0.")
        return 2

    benchmark_records = read_benchmark(args.benchmark)
    if args.limit_queries is not None:
        benchmark_records = benchmark_records[: max(0, args.limit_queries)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            cache_dir=args.cache_dir,
            search_cost_per_1000_calls_usd=args.search_cost_per_1000,
        )
    )

    predictions_by_k: dict[int, list[dict[str, Any]]] = {
        k: [] for k in top_ks
    }
    query_logs: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
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
        print(
            f"[{index}/{len(benchmark_records)}] qid={qid} "
            f"cutoff={cutoff}"
        )

        query_started = time.perf_counter()
        try:
            result = retriever.search(
                SearchRequest(
                    query=question,
                    per_page=args.per_page,
                    to_publication_date=cutoff,
                ),
                refresh_cache=args.refresh_cache,
            )
            papers = result.papers
            stats = result.stats.to_dict()
            error = None
        except (OpenAlexError, ValueError, OSError) as exc:
            papers = []
            error_stats = (
                exc.stats.to_dict()
                if isinstance(exc, OpenAlexError) and exc.stats is not None
                else None
            )
            stats = error_stats or {
                "provider": "openalex",
                "actual_api_calls": 0,
                "cache_hits": 0,
                "retries": 0,
                "http_status": None,
                "response_bytes": 0,
                "request_latency_ms": 0.0,
                "end_to_end_latency_ms": (
                    time.perf_counter() - query_started
                ) * 1000.0,
                "estimated_api_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "rate_limit_headers": {},
                "error": str(exc),
            }
            error = str(exc)
            print(f"[ERROR] qid={qid}: {error}")
            if not args.continue_on_error:
                return 2

        paper_dicts = [paper.to_dict() for paper in papers]
        raw_records.append(
            {
                "qid": qid,
                "question": question,
                "published_time": published_time,
                "cutoff_date": cutoff,
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
                "requested_results": args.per_page,
                "returned_papers": len(papers),
                "raw_result_count": (result.raw_result_count if error is None else 0),
                **stats,
                "error": error or stats.get("error"),
            }
        )

    wall_clock_ms = (time.perf_counter() - run_started) * 1000.0
    raw_path = output_dir / "raw_predictions_top100.jsonl"
    write_jsonl(raw_path, raw_records)

    prediction_paths: dict[int, Path] = {}
    for k, records in predictions_by_k.items():
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records)
        prediction_paths[k] = path

    query_log_path = output_dir / "query_logs.jsonl"
    write_jsonl(query_log_path, query_logs)

    summary = build_run_summary(query_logs, wall_clock_ms=wall_clock_ms)
    summary.update(
        {
            "baseline": "B0",
            "provider": "openalex",
            "query_strategy": "raw_query_direct_search",
            "cutoff_offset_days": args.cutoff_offset_days,
            "per_page": args.per_page,
            "top_k_outputs": top_ks,
            "prediction_files": {
                str(k): str(path) for k, path in prediction_paths.items()
            },
        }
    )

    gold_path = Path(args.gold)
    if gold_path.exists() and args.limit_queries is None:
        evaluation_summaries = run_evaluations(
            gold_path=gold_path,
            prediction_paths=prediction_paths,
            output_dir=output_dir,
        )
        summary["evaluation"] = evaluation_summaries
    else:
        summary["evaluation_skipped"] = (
            "Gold file missing or --limit-queries was used."
        )

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Queries: {summary['queries']['total']}")
    print(f"[OK] Actual API calls: {summary['retrieval']['actual_api_calls']}")
    print(f"[OK] Cache hits: {summary['retrieval']['cache_hits']}")
    print(f"[OK] Total tokens: {summary['tokens']['total']}")
    print(
        "[OK] Estimated API cost: $"
        f"{summary['estimated_cost_usd']:.6f}"
    )
    print(
        "[OK] Wall-clock latency: "
        f"{summary['latency_ms']['wall_clock']:.2f} ms"
    )
    print(f"[OUTPUT] Summary: {summary_path}")
    for k, path in prediction_paths.items():
        print(f"[OUTPUT] Top{k}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
