from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


TOPK_NAMES = ("top20", "top50", "top100")
MODES = ("strict", "pasa_title")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return records
    with source.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_get(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_result_table(b0_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(b0_dir)
    rows: list[dict[str, Any]] = []

    for topk in TOPK_NAMES:
        for mode in MODES:
            summary_path = base / "evaluation" / topk / mode / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            counts = summary.get("counts", {})
            macro = summary.get("macro", {})
            micro = summary.get("micro", {})
            rows.append(
                {
                    "topk": topk,
                    "mode": mode,
                    "tp": counts.get("tp", 0),
                    "fp": counts.get("fp", 0),
                    "fn": counts.get("fn", 0),
                    "gold_papers": counts.get("gold_papers", 0),
                    "predictions_after_dedup": counts.get(
                        "predictions_after_dedup",
                        0,
                    ),
                    "macro_precision": macro.get("precision", 0.0),
                    "macro_recall": macro.get("recall", 0.0),
                    "macro_f1": macro.get("f1", 0.0),
                    "micro_precision": micro.get("precision", 0.0),
                    "micro_recall": micro.get("recall", 0.0),
                    "micro_f1": micro.get("f1", 0.0),
                }
            )
    return rows


def best_row_by_macro_f1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("macro_f1", 0.0)))


def write_result_table_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "topk",
        "mode",
        "tp",
        "fp",
        "fn",
        "gold_papers",
        "predictions_after_dedup",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "0.000000"


def write_result_table_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# B0 OpenAlex基线结果表",
        "",
        "| Top-K | 口径 | TP | FP | FN | Macro P | Macro R | Macro F1 | Micro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topk} | {mode} | {tp} | {fp} | {fn} | {mp} | {mr} | {mf} | {uf} |".format(
                topk=row["topk"],
                mode=row["mode"],
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
                mp=_fmt_float(row["macro_precision"]),
                mr=_fmt_float(row["macro_recall"]),
                mf=_fmt_float(row["macro_f1"]),
                uf=_fmt_float(row["micro_f1"]),
            )
        )

    best = best_row_by_macro_f1(rows)
    if best:
        lines.extend(
            [
                "",
                "## 当前最佳B0设置",
                "",
                (
                    f"- Top-K：`{best['topk']}`，口径：`{best['mode']}`，"
                    f"Macro F1：`{_fmt_float(best['macro_f1'])}`。"
                ),
            ]
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return ">100"


def build_query_diagnostics(
    per_query_records: list[dict[str, Any]],
    query_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    logs_by_qid = {str(item.get("qid")): item for item in query_logs}
    diagnostics: list[dict[str, Any]] = []

    for item in per_query_records:
        qid = str(item.get("qid"))
        counts = item.get("counts", {})
        metrics = item.get("metrics", {})
        matched_pairs = item.get("matched_pairs", [])
        false_positives = item.get("false_positives", [])
        false_negatives = item.get("false_negatives", [])
        log = logs_by_qid.get(qid, {})

        tp = int(counts.get("tp", 0))
        fp = int(counts.get("fp", 0))
        fn = int(counts.get("fn", 0))
        gold_count = int(counts.get("gold", 0))
        pred_count = int(counts.get("predictions_after_dedup", 0))
        recall20 = float(metrics.get("recall@20", 0.0))
        recall50 = float(metrics.get("recall@50", 0.0))
        recall100 = float(metrics.get("recall@100", metrics.get("recall", 0.0)))

        matched_ranks = [
            int(pair.get("prediction_rank", 0))
            for pair in matched_pairs
            if int(pair.get("prediction_rank", 0)) > 0
        ]

        diagnostics.append(
            {
                "qid": qid,
                "question": item.get("question"),
                "gold_count": gold_count,
                "prediction_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "recall@20": recall20,
                "recall@50": recall50,
                "recall@100": recall100,
                "matched_ranks": matched_ranks,
                "best_matched_rank": min(matched_ranks) if matched_ranks else None,
                "worst_matched_rank": max(matched_ranks) if matched_ranks else None,
                "false_positive_count": len(false_positives),
                "false_negative_count": len(false_negatives),
                "api_calls": log.get("api_calls"),
                "cache_hit": log.get("cache_hit"),
                "retries": log.get("retries"),
                "latency_ms": log.get("latency_ms"),
            }
        )

    diagnostics.sort(
        key=lambda row: (
            row["recall"],
            -row["gold_count"],
            row["qid"],
        )
    )
    return diagnostics


def extract_false_negatives(
    per_query_records: list[dict[str, Any]],
    limit_per_query: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in per_query_records:
        qid = item.get("qid")
        question = item.get("question")
        for fn in item.get("false_negatives", [])[:limit_per_query]:
            paper = fn.get("paper", {})
            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "gold_index": fn.get("gold_index"),
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "canonical_id": paper.get("canonical_id"),
                }
            )
    return rows


def extract_false_positives(
    per_query_records: list[dict[str, Any]],
    limit_per_query: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in per_query_records:
        qid = item.get("qid")
        question = item.get("question")
        for fp in item.get("false_positives", [])[:limit_per_query]:
            paper = fp.get("paper", {})
            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "prediction_rank": fp.get("prediction_rank"),
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "doi": paper.get("doi"),
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "canonical_id": paper.get("canonical_id"),
                }
            )
    return rows


def matched_rank_distribution(per_query_records: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counter: Counter[str] = Counter()
    match_type_counter: Counter[str] = Counter()

    for item in per_query_records:
        for pair in item.get("matched_pairs", []):
            rank = int(pair.get("prediction_rank", 0))
            if rank > 0:
                bucket_counter[rank_bucket(rank)] += 1
            match_type = pair.get("match_type") or "unknown"
            match_type_counter[str(match_type)] += 1

    ordered_buckets = {
        bucket: bucket_counter.get(bucket, 0)
        for bucket in ("1-10", "11-20", "21-50", "51-100", ">100")
    }
    return {
        "rank_buckets": ordered_buckets,
        "match_types": dict(match_type_counter),
        "total_matches": sum(bucket_counter.values()),
    }


def build_recommendations(
    output_path: str | Path,
    result_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    zero_recall: list[dict[str, Any]],
    late_recall: list[dict[str, Any]],
    high_fp: list[dict[str, Any]],
    rank_distribution: dict[str, Any],
) -> None:
    best = best_row_by_macro_f1(result_rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    zero_count = len(zero_recall)
    late_count = len(late_recall)
    high_fp_count = len(high_fp)
    total_queries = len(diagnostics)

    lines = [
        "# B1/B2改进建议",
        "",
        "## 1. B0结论",
        "",
    ]
    if best:
        lines.append(
            f"- 当前B0最佳设置为 `{best['topk']} / {best['mode']}`，"
            f"Macro F1=`{_fmt_float(best['macro_f1'])}`。"
        )

    lines.extend(
        [
            f"- Top100仍然零召回查询数：`{zero_count}/{total_queries}`。",
            f"- Top20未召回但Top50/Top100才召回的查询数：`{late_count}`。",
            f"- 高FP查询候选数：`{high_fp_count}`。",
            "",
            "## 2. 对B1查询改写的要求",
            "",
            "B1不要直接增加返回Top-K，而应优先改写零召回查询。建议为每条复杂查询生成：",
            "",
            "1. 核心主题短查询；",
            "2. 方法关键词查询；",
            "3. 数据集/基准关键词查询；",
            "4. 模型或任务别名查询；",
            "5. 去除疑问句和口语表达后的检索式。",
            "",
            "## 3. 对B2多子查询召回的要求",
            "",
            "若查询包含多个约束，不应只用整句搜索。B2应将查询拆成若干子查询，并合并候选池，再用统一ID去重。",
            "",
            "## 4. 对B3排序与精筛的要求",
            "",
            "B0中Top100的FP显著高于Top50，说明盲目扩大候选会伤害F1。B3应使用语义Reranker和元数据过滤优先提升Precision。",
            "",
            "## 5. 排名分布",
            "",
            "```json",
            json.dumps(rank_distribution, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 6. 优先检查的零召回查询",
            "",
        ]
    )

    for item in zero_recall[:10]:
        lines.append(
            f"- `{item['qid']}`，gold={item['gold_count']}：{item.get('question')}"
        )

    lines.extend(["", "## 7. 优先检查的高FP查询", ""])
    for item in high_fp[:10]:
        lines.append(
            f"- `{item['qid']}`，FP={item['fp']}，TP={item['tp']}：{item.get('question')}"
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(slots=True)
class B0AnalysisConfig:
    b0_dir: Path
    output_dir: Path
    mode: str = "strict"
    focus_topk: int = 50
    rank_source_topk: int = 100
    top_n_queries: int = 20


def analyze_b0(config: B0AnalysisConfig) -> dict[str, Any]:
    b0_dir = Path(config.b0_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = b0_dir / "run_summary.json"
    query_logs_path = b0_dir / "query_logs.jsonl"
    per_query_path = (
        b0_dir
        / "evaluation"
        / f"top{config.rank_source_topk}"
        / config.mode
        / "per_query.jsonl"
    )

    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")
    if not per_query_path.exists():
        raise FileNotFoundError(f"Missing per-query file: {per_query_path}")

    run_summary = read_json(run_summary_path)
    query_logs = read_jsonl(query_logs_path)
    per_query_records = read_jsonl(per_query_path)

    result_rows = load_result_table(b0_dir)
    write_result_table_csv(output_dir / "b0_result_table.csv", result_rows)
    write_result_table_markdown(output_dir / "b0_result_table.md", result_rows)

    diagnostics = build_query_diagnostics(per_query_records, query_logs)
    write_jsonl(output_dir / "query_diagnostics.jsonl", diagnostics)

    zero_recall = [
        item for item in diagnostics
        if item["gold_count"] > 0 and item["tp"] == 0
    ]
    zero_recall.sort(key=lambda item: (-item["gold_count"], item["qid"]))

    late_recall = [
        item for item in diagnostics
        if item["recall@20"] == 0.0 and item["recall@100"] > 0.0
    ]
    late_recall.sort(
        key=lambda item: (
            item["best_matched_rank"] if item["best_matched_rank"] is not None else 999,
            -item["gold_count"],
        )
    )

    high_fp = sorted(
        diagnostics,
        key=lambda item: (-item["fp"], item["tp"], item["qid"]),
    )[: config.top_n_queries]

    write_jsonl(output_dir / "zero_recall_queries.jsonl", zero_recall)
    write_jsonl(output_dir / "late_recall_queries.jsonl", late_recall)
    write_jsonl(output_dir / "high_fp_queries.jsonl", high_fp)

    rank_distribution = matched_rank_distribution(per_query_records)
    write_json(output_dir / "matched_rank_distribution.json", rank_distribution)

    false_negatives = extract_false_negatives(per_query_records)
    false_positives = extract_false_positives(per_query_records)
    write_jsonl(output_dir / "false_negatives_sample.jsonl", false_negatives)
    write_jsonl(output_dir / "false_positives_sample.jsonl", false_positives)

    focus_row = None
    for row in result_rows:
        if row.get("topk") == f"top{config.focus_topk}" and row.get("mode") == config.mode:
            focus_row = row
            break

    analysis_summary = {
        "baseline": run_summary.get("baseline", "B0"),
        "provider": run_summary.get("provider"),
        "query_strategy": run_summary.get("query_strategy"),
        "queries": run_summary.get("queries"),
        "retrieval": run_summary.get("retrieval"),
        "tokens": run_summary.get("tokens"),
        "estimated_cost_usd": run_summary.get("estimated_cost_usd"),
        "latency_ms": run_summary.get("latency_ms"),
        "focus": {
            "topk": f"top{config.focus_topk}",
            "mode": config.mode,
            "result": focus_row,
        },
        "best_by_macro_f1": best_row_by_macro_f1(result_rows),
        "error_analysis": {
            "rank_source": f"top{config.rank_source_topk}/{config.mode}",
            "query_count": len(diagnostics),
            "zero_recall_queries": len(zero_recall),
            "late_recall_queries": len(late_recall),
            "top_high_fp_queries": len(high_fp),
            "mean_gold_per_query": (
                fmean(item["gold_count"] for item in diagnostics)
                if diagnostics
                else 0.0
            ),
            "mean_prediction_per_query": (
                fmean(item["prediction_count"] for item in diagnostics)
                if diagnostics
                else 0.0
            ),
            "rank_distribution": rank_distribution,
        },
    }
    write_json(output_dir / "analysis_summary.json", analysis_summary)

    build_recommendations(
        output_dir / "b1_recommendations.md",
        result_rows,
        diagnostics,
        zero_recall,
        late_recall,
        high_fp,
        rank_distribution,
    )

    return analysis_summary
