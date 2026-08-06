from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import (
    EvaluationConfig,
    EvaluationResult,
    evaluate_records,
    write_evaluation_outputs,
)
from scholarpath.fusion.candidate_fusion import (
    CandidateSource,
    FusionConfig,
    fuse_candidate_records,
)
from scholarpath.observability.run_log import build_run_summary
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.semantic_rerank import (
    annotate_and_semantic_rerank_record,
    make_semantic_config,
)
from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.dates import parse_benchmark_cutoff
from scholarpath.retrieval.openalex import OpenAlexConfig, OpenAlexError, OpenAlexRetriever
from scholarpath.retrieval.rescue_retrieval import (
    RescuePlanSpec,
    aggregate_rescue_hits,
    preserve_or_replace_record,
    validate_query_plan_records,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_by_qid(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid") or "").strip()
        if not qid:
            raise ValueError("Record is missing qid.")
        if qid in output:
            raise ValueError(f"Duplicate qid: {qid}")
        output[qid] = record
    return output


def empty_stats(error: str | None = None) -> dict[str, Any]:
    return {
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
        "error": error,
    }


def merge_stats(stats_list: list[Mapping[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    errors = [str(item.get("error")) for item in stats_list if item.get("error")]
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
        "error": "; ".join(errors) if errors else None,
    }


def evaluate_prediction_records(
    *,
    gold_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, Any]],
    output_dir: Path,
    mode: str,
) -> EvaluationResult:
    result = evaluate_records(
        gold_records=gold_records,
        prediction_records=prediction_records,
        config=EvaluationConfig(
            mode=mode,
            recall_at=(20, 50, 100),
            deduplicate=True,
        ),
    )
    write_evaluation_outputs(result, output_dir)
    return result


def zero_recall_qids(result: EvaluationResult) -> set[str]:
    return {item.qid for item in result.per_query if item.tp == 0}


def recovered_query_records(
    baseline: EvaluationResult,
    current: EvaluationResult,
    target_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_qid = {item.qid: item for item in baseline.per_query}
    current_by_qid = {item.qid: item for item in current.per_query}
    recovered: list[dict[str, Any]] = []
    unrecovered: list[dict[str, Any]] = []

    for qid in sorted(target_qids):
        before = baseline_by_qid[qid]
        after = current_by_qid[qid]
        record = {
            "qid": qid,
            "question": after.question,
            "baseline_tp": before.tp,
            "current_tp": after.tp,
            "tp_delta": after.tp - before.tp,
            "matched_pairs": after.matched_pairs,
            "remaining_false_negatives": after.false_negatives,
        }
        if before.tp == 0 and after.tp > 0:
            recovered.append(record)
        elif after.tp == 0:
            unrecovered.append(record)
    return recovered, unrecovered


def write_review(
    *,
    path: Path,
    summary: dict[str, Any],
    recovered: list[dict[str, Any]],
    unrecovered: list[dict[str, Any]],
) -> None:
    comparison = summary["comparison_to_b4"]
    retrieval = summary["retrieval"]
    lines = [
        "# ScholarPath Week 2 Day 4：定向救援召回结果",
        "",
        "## 1. 执行结果",
        "",
        f"- 目标查询数：{summary['target_query_count']}",
        f"- 执行检索计划数：{summary['executed_plan_count']}",
        f"- 实际 API 调用：{retrieval['actual_api_calls']}",
        f"- Cache 命中：{retrieval['cache_hits']}",
        f"- 估算成本：${summary['estimated_cost_usd']:.6f}",
        "",
        "## 2. 与 B4 Top100 对比",
        "",
        f"- B4 TP：{comparison['baseline_top100_tp']}",
        f"- Day 4 TP：{comparison['day4_top100_tp']}",
        f"- TP 增量：{comparison['top100_tp_delta']:+d}",
        f"- B4 zero-recall：{comparison['baseline_zero_recall_count']}",
        f"- Day 4 zero-recall：{comparison['day4_zero_recall_count']}",
        f"- zero-recall 减少：{comparison['zero_recall_reduction']}",
        f"- 恢复查询数：{len(recovered)}",
        "",
        "## 3. 已恢复查询",
        "",
    ]
    if recovered:
        lines.extend(["| QID | TP 增量 | 最佳命中排名 | 命中论文 |", "|---|---:|---:|---|"])
        for item in recovered:
            pairs = item.get("matched_pairs") or []
            best_rank = min(
                (int(pair.get("prediction_rank", 999999)) for pair in pairs),
                default=0,
            )
            titles = "; ".join(
                str((pair.get("gold") or {}).get("title") or "")
                for pair in pairs
            )
            lines.append(
                f"| `{item['qid']}` | {item['tp_delta']} | {best_rank or '-'} | {titles} |"
            )
    else:
        lines.append("本轮没有恢复新的 Gold 论文。")

    lines.extend(["", "## 4. 仍未恢复查询", ""])
    if unrecovered:
        for item in unrecovered:
            lines.append(f"- `{item['qid']}`：{item.get('question') or ''}")
    else:
        lines.append("24 条目标查询均至少恢复 1 篇 Gold 论文。")

    lines.extend(
        [
            "",
            "## 5. 判定规则",
            "",
            "- 达标：Top100 TP 不低于 69，zero-recall 不高于 20，实际 API 调用不超过 48。",
            "- 部分达标：TP 有真实提升，但未达到上述目标；保留有效计划并分析未恢复查询。",
            "- 未达标：TP 没有提升；暂停继续扩大 API 预算，转入标题锚点或引文扩展分析。",
            "",
            "Day 4 召回阶段不读取 Gold；Gold 仅在全部检索完成后用于离线评测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Day-4 anchor-aware rescue retrieval, fusion, reranking, and evaluation."
    )
    parser.add_argument("--benchmark", default="data/raw/RealScholarQuery/test.jsonl")
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument(
        "--query-plans",
        default="outputs/week2_day3_query_plans/query_plans.jsonl",
    )
    parser.add_argument(
        "--base-pool",
        default="outputs/b4_fusion_semantic/fused_candidates_before_rerank.jsonl",
    )
    parser.add_argument(
        "--base-predictions-dir",
        default="outputs/b4_fusion_semantic",
    )
    parser.add_argument("--output-dir", default="outputs/week2_day4_rescue")
    parser.add_argument("--per-plan-page", type=int, default=50)
    parser.add_argument("--rescue-top-n", type=int, default=100)
    parser.add_argument("--max-merged-candidates", type=int, default=300)
    parser.add_argument("--max-plans-per-query", type=int, default=2)
    parser.add_argument("--max-api-calls", type=int, default=48)
    parser.add_argument("--semantic-mode", choices=["conservative", "hybrid", "semantic_first"], default="semantic_first")
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--cutoff-offset-days", type=int, default=7)
    parser.add_argument("--cache-dir", default="data/cache/openalex_day4")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--search-cost-per-1000", type=float, default=1.0)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 1 <= args.per_plan_page <= 100:
        print("[ERROR] --per-plan-page must be between 1 and 100.")
        return 2
    if args.rescue_top_n < 1 or args.max_merged_candidates < 1:
        print("[ERROR] Candidate limits must be positive.")
        return 2

    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2

    benchmark_records = read_jsonl(args.benchmark)
    benchmark_by_qid = records_by_qid(benchmark_records)
    plan_records = read_jsonl(args.query_plans)
    if args.limit_queries is not None:
        plan_records = plan_records[: max(0, args.limit_queries)]

    try:
        plan_validation = validate_query_plan_records(
            plan_records,
            max_plans_per_query=args.max_plans_per_query,
            max_api_calls=args.max_api_calls,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    target_qids = set(plan_validation["qids"])
    missing_benchmark_qids = sorted(target_qids - set(benchmark_by_qid))
    if missing_benchmark_qids:
        print(f"[ERROR] Query plans missing from benchmark: {missing_benchmark_qids}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_manifest = {
        "strategy": "day4_anchor_aware_rescue_retrieval",
        "query_plan_path": str(args.query_plans),
        "target_query_count": len(target_qids),
        "planned_query_count": plan_validation["plan_count"],
        "estimated_api_calls": plan_validation["estimated_api_calls"],
        "max_api_calls": args.max_api_calls,
        "per_plan_page": args.per_plan_page,
        "rescue_top_n": args.rescue_top_n,
        "max_merged_candidates": args.max_merged_candidates,
        "semantic_mode": args.semantic_mode,
        "target_qids": sorted(target_qids),
        "production_retrieval_uses_gold": False,
    }
    (output_dir / "execution_manifest.json").write_text(
        json.dumps(execution_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[DRY RUN] Target queries: {len(target_qids)}")
        print(f"[DRY RUN] Planned API calls: {plan_validation['estimated_api_calls']}")
        print("[DRY RUN] API calls executed: 0")
        print(f"[OUTPUT] Manifest: {output_dir / 'execution_manifest.json'}")
        return 0

    if not os.getenv("OPENALEX_API_KEY"):
        print("[ERROR] OPENALEX_API_KEY is not set in this terminal.")
        return 2

    base_pool_records = records_by_qid(read_jsonl(args.base_pool))
    base_predictions_by_k: dict[int, dict[str, dict[str, Any]]] = {}
    for k in top_ks:
        path = Path(args.base_predictions_dir) / f"predictions_top{k}.jsonl"
        base_predictions_by_k[k] = records_by_qid(read_jsonl(path))

    retriever = OpenAlexRetriever(
        OpenAlexConfig(
            cache_dir=args.cache_dir,
            search_cost_per_1000_calls_usd=args.search_cost_per_1000,
        )
    )
    decomposer = ConstraintDecomposer(max_constraints=6, max_subqueries=6)
    semantic_config = make_semantic_config(args.semantic_mode)
    fusion_config = FusionConfig(max_fused_candidates=args.max_merged_candidates)
    base_source = CandidateSource(name="b4_base", directory="", weight=1.0)
    rescue_source = CandidateSource(name="day4_rescue", directory="", weight=1.15)

    rescue_records_by_qid: dict[str, dict[str, Any]] = {}
    plan_result_records: list[dict[str, Any]] = []
    query_logs: list[dict[str, Any]] = []
    failed_plans: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    actual_call_guard = 0

    for query_index, plan_record in enumerate(plan_records, start=1):
        qid = str(plan_record["qid"])
        benchmark = benchmark_by_qid[qid]
        question = str(benchmark.get("question") or plan_record.get("question") or "").strip()
        source_meta = benchmark.get("source_meta")
        published_time = source_meta.get("published_time") if isinstance(source_meta, dict) else None
        cutoff = parse_benchmark_cutoff(published_time, args.cutoff_offset_days)
        plans = [
            RescuePlanSpec.from_mapping(item)
            for item in plan_record.get("planned_queries", [])[: args.max_plans_per_query]
        ]

        print(f"[{query_index}/{len(plan_records)}] qid={qid} plans={len(plans)} cutoff={cutoff}")
        query_started = time.perf_counter()
        query_stats: list[dict[str, Any]] = []
        hits = []

        for plan_index, plan in enumerate(plans, start=1):
            if actual_call_guard >= args.max_api_calls:
                print("[ERROR] Runtime API call guard reached.")
                return 2
            try:
                result = retriever.search(
                    SearchRequest(
                        query=plan.text,
                        per_page=args.per_plan_page,
                        to_publication_date=cutoff,
                    ),
                    refresh_cache=args.refresh_cache,
                )
                papers = result.papers
                stats = result.stats.to_dict()
                actual_call_guard += int(stats.get("actual_api_calls", 0))
                error = None
            except (OpenAlexError, ValueError, OSError) as exc:
                papers = []
                stats = (
                    exc.stats.to_dict()
                    if isinstance(exc, OpenAlexError) and exc.stats is not None
                    else empty_stats(str(exc))
                )
                actual_call_guard += int(stats.get("actual_api_calls", 0))
                error = str(exc)
                failed_plans.append(
                    {
                        "qid": qid,
                        "plan_index": plan_index,
                        "query": plan.text,
                        "error": error,
                        "stats": stats,
                    }
                )
                print(f"[ERROR] qid={qid} plan={plan_index}: {error}")
                if not args.continue_on_error:
                    return 2

            query_stats.append({**stats, "raw_result_count": len(papers), "error": error or stats.get("error")})
            plan_result_records.append(
                {
                    "qid": qid,
                    "question": question,
                    "plan_index": plan_index,
                    "plan": plan.to_dict(),
                    "cutoff_date": cutoff,
                    "returned_papers": len(papers),
                    "papers": [paper.to_dict(include_raw=True) for paper in papers],
                    "stats": stats,
                    "error": error or stats.get("error"),
                }
            )
            for rank, paper in enumerate(papers, start=1):
                hits.append((paper, rank, plan))

        rescue_papers = aggregate_rescue_hits(hits)[: args.rescue_top_n]
        rescue_records_by_qid[qid] = {
            "qid": qid,
            "question": question,
            "papers": [paper.to_dict(include_raw=True) for paper in rescue_papers],
            "rescue": {
                "strategy": "anchor_aware_multi_plan_openalex",
                "plan_count": len(plans),
                "cutoff_date": cutoff,
            },
        }
        elapsed_ms = (time.perf_counter() - query_started) * 1000.0
        merged = merge_stats(query_stats, elapsed_ms)
        query_logs.append(
            {
                "qid": qid,
                "query_index": query_index,
                "query": question,
                "plan_count": len(plans),
                "returned_papers": len(rescue_papers),
                "raw_result_count": sum(int(item.get("raw_result_count", 0)) for item in query_stats),
                **merged,
            }
        )

    # All 50 qids are represented in the rescue source. Non-target records are empty.
    rescue_records_all: list[dict[str, Any]] = []
    reranked_target_records: dict[str, dict[str, Any]] = {}
    merged_candidates_all: list[dict[str, Any]] = []

    for benchmark in benchmark_records:
        qid = str(benchmark["qid"])
        question = str(benchmark.get("question") or "")
        rescue_record = rescue_records_by_qid.get(
            qid,
            {"qid": qid, "question": question, "papers": []},
        )
        rescue_records_all.append(rescue_record)

        base_pool = base_pool_records[qid]
        if qid not in target_qids:
            merged_candidates_all.append(base_pool)
            continue

        fused_record, fusion_stats = fuse_candidate_records(
            qid=qid,
            question=question,
            source_records=[
                (base_source, base_pool),
                (rescue_source, rescue_record),
            ],
            config=fusion_config,
        )
        fused_record["day4_fusion"] = {
            "base_candidate_count": len(base_pool.get("papers") or []),
            "rescue_candidate_count": len(rescue_record.get("papers") or []),
            "fusion_stats": fusion_stats,
        }
        merged_candidates_all.append(fused_record)
        decomposition = decomposer.decompose(question)
        reranked_target_records[qid] = annotate_and_semantic_rerank_record(
            fused_record,
            decomposition,
            semantic_config,
        )

    write_jsonl(output_dir / "plan_results.jsonl", plan_result_records)
    write_jsonl(output_dir / "rescue_candidates.jsonl", rescue_records_all)
    write_jsonl(output_dir / "merged_candidates_before_rerank.jsonl", merged_candidates_all)
    write_jsonl(output_dir / "query_logs.jsonl", query_logs)
    write_jsonl(output_dir / "failed_plans.jsonl", failed_plans)

    prediction_records_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in top_ks}
    for benchmark in benchmark_records:
        qid = str(benchmark["qid"])
        for k in top_ks:
            prediction_records_by_k[k].append(
                preserve_or_replace_record(
                    qid=qid,
                    target_qids=target_qids,
                    baseline_record=base_predictions_by_k[k][qid],
                    replacement_record=reranked_target_records.get(qid),
                    top_k=k,
                )
            )

    prediction_paths: dict[int, Path] = {}
    for k, records in prediction_records_by_k.items():
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records)
        prediction_paths[k] = path

    gold_records = load_gold(args.gold)
    evaluation_summaries: dict[str, Any] = {}
    evaluation_results: dict[tuple[int, str], EvaluationResult] = {}
    baseline_results: dict[tuple[int, str], EvaluationResult] = {}

    for k in top_ks:
        current_predictions = load_predictions(prediction_paths[k])
        baseline_predictions = load_predictions(
            Path(args.base_predictions_dir) / f"predictions_top{k}.jsonl"
        )

        for mode in ("strict", "strict_v2", "pasa_title"):
            current_result = evaluate_prediction_records(
                gold_records=gold_records,
                prediction_records=current_predictions,
                output_dir=output_dir / "evaluation" / f"top{k}" / mode,
                mode=mode,
            )

            baseline_result = evaluate_records(
                gold_records=gold_records,
                prediction_records=baseline_predictions,
                config=EvaluationConfig(
                    mode=mode,
                    recall_at=(20, 50, 100),
                    deduplicate=True,
                ),
            )

            evaluation_results[(k, mode)] = current_result
            baseline_results[(k, mode)] = baseline_result
            evaluation_summaries[f"top{k}_{mode}"] = current_result.summary()

    max_k = max(top_ks)
    focus_key = (100 if 100 in top_ks else max_k, "strict")
    focus_result = evaluation_results[focus_key]
    baseline_focus = baseline_results[focus_key]
    recovered, unrecovered = recovered_query_records(
        baseline_focus,
        focus_result,
        target_qids,
    )
    write_jsonl(output_dir / "recovered_queries.jsonl", recovered)
    write_jsonl(output_dir / "unrecovered_queries.jsonl", unrecovered)

    wall_clock_ms = (time.perf_counter() - run_started) * 1000.0
    summary = build_run_summary(query_logs, wall_clock_ms=wall_clock_ms)
    baseline_summary = baseline_focus.summary()
    focus_summary = focus_result.summary()
    baseline_zero = len(zero_recall_qids(baseline_focus))
    current_zero = len(zero_recall_qids(focus_result))
    summary.update(
        {
            "baseline": "B4_plus_Day4_rescue",
            "strategy": "anchor_aware_rescue_only_for_retrieval_miss_queries",
            "target_query_count": len(target_qids),
            "executed_plan_count": len(plan_result_records),
            "max_api_calls": args.max_api_calls,
            "per_plan_page": args.per_plan_page,
            "semantic_mode": args.semantic_mode,
            "top_k_outputs": top_ks,
            "production_retrieval_uses_gold": False,
            "evaluation_uses_gold_after_retrieval": True,
            "comparison_to_b4": {
                "baseline_top100_tp": int(baseline_summary["counts"]["tp"]),
                "day4_top100_tp": int(focus_summary["counts"]["tp"]),
                "top100_tp_delta": int(focus_summary["counts"]["tp"] - baseline_summary["counts"]["tp"]),
                "baseline_zero_recall_count": baseline_zero,
                "day4_zero_recall_count": current_zero,
                "zero_recall_reduction": baseline_zero - current_zero,
                "recovered_query_count": len(recovered),
                "unrecovered_target_count": len(unrecovered),
            },
            "evaluation": evaluation_summaries,
            "failed_plan_count": len(failed_plans),
        }
    )
    summary["tokens"]["note"] = "Day 4 uses deterministic plans and OpenAlex; no LLM tokens are consumed."
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(
        path=output_dir / "day4_rescue_review.md",
        summary=summary,
        recovered=recovered,
        unrecovered=unrecovered,
    )

    comparison = summary["comparison_to_b4"]
    print(f"[OK] Target queries: {len(target_qids)}")
    print(f"[OK] Executed plans: {len(plan_result_records)}")
    print(f"[OK] Actual API calls: {summary['retrieval']['actual_api_calls']}")
    print(f"[OK] Cache hits: {summary['retrieval']['cache_hits']}")
    print(f"[OK] Estimated cost: ${summary['estimated_cost_usd']:.6f}")
    print(f"[RESULT] Top100 TP: {comparison['baseline_top100_tp']} -> {comparison['day4_top100_tp']} ({comparison['top100_tp_delta']:+d})")
    print(f"[RESULT] Zero-recall: {comparison['baseline_zero_recall_count']} -> {comparison['day4_zero_recall_count']}")
    print(f"[RESULT] Recovered queries: {comparison['recovered_query_count']}")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
