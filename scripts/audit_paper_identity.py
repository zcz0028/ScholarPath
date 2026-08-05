from __future__ import annotations

import argparse
import csv
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
from scholarpath.evaluation.matching import (
    audit_possible_matches,
    deduplicate_predictions,
    deduplicate_predictions_v2,
    match_papers,
)
from scholarpath.paper.schema import PaperRecord


def _coverage(records: list[PaperRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for paper in records:
        counter["papers"] += 1
        for field_name in (
            "doi",
            "arxiv_id",
            "openalex_id",
            "semantic_scholar_id",
            "url",
            "authors",
            "year",
        ):
            if getattr(paper, field_name):
                counter[f"direct_{field_name}"] += 1
        inferred = paper.inferred_identifiers()
        for field_name in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id"):
            if inferred.get(field_name):
                counter[f"v2_{field_name}"] += 1
        if paper.identity_keys():
            counter["legacy_identified"] += 1
        if paper.identity_keys_v2():
            counter["v2_identified"] += 1
    return dict(counter)


def _pair_key(pair: Any) -> tuple[int, int]:
    return pair.prediction_index, pair.gold_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit ScholarPath paper identifiers, matching, and deduplication."
    )
    parser.add_argument(
        "--gold",
        default="data/processed/realscholarquery_gold.jsonl",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/b4_fusion_semantic/predictions_top100.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/week2_day2_identity_audit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gold_path = (PROJECT_ROOT / args.gold).resolve()
    prediction_path = (PROJECT_ROOT / args.predictions).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_records = load_gold(gold_path)
    prediction_records = load_predictions(prediction_path)
    qids = list(gold_records)

    all_gold = [paper for qid in qids for paper in gold_records[qid]["papers"]]
    all_predictions = [
        paper
        for qid in qids
        for paper in prediction_records.get(qid, {"papers": []})["papers"]
    ]

    per_query: list[dict[str, Any]] = []
    new_matches: list[dict[str, Any]] = []
    possible_matches: list[dict[str, Any]] = []
    dedup_changes: list[dict[str, Any]] = []
    legacy_tp = 0
    v2_tp = 0
    legacy_removed_total = 0
    v2_removed_total = 0

    for qid in qids:
        gold = gold_records[qid]["papers"]
        predictions = prediction_records.get(qid, {"papers": []})["papers"]
        legacy_kept, legacy_removed = deduplicate_predictions(predictions)
        v2_kept, v2_removed = deduplicate_predictions_v2(predictions)
        legacy = match_papers(legacy_kept, gold, mode="strict")
        v2 = match_papers(v2_kept, gold, mode="strict_v2")
        legacy_tp += legacy.tp
        v2_tp += v2.tp
        legacy_removed_total += len(legacy_removed)
        v2_removed_total += len(v2_removed)

        legacy_matched_gold = {pair.gold_index for pair in legacy.pairs}
        for pair in v2.pairs:
            if pair.gold_index in legacy_matched_gold:
                continue
            new_matches.append(
                {
                    "qid": qid,
                    "question": gold_records[qid].get("question"),
                    "prediction_rank_after_v2_dedup": pair.prediction_index + 1,
                    "match_type": pair.match_type,
                    "prediction": v2_kept[pair.prediction_index].to_dict(),
                    "gold": gold[pair.gold_index].to_dict(),
                }
            )

        audits = audit_possible_matches(v2_kept, gold)
        for item in audits:
            possible_matches.append(
                {
                    "qid": qid,
                    "question": gold_records[qid].get("question"),
                    **item.to_dict(),
                    "prediction": v2_kept[item.prediction_index].to_dict(),
                    "gold": gold[item.gold_index].to_dict(),
                }
            )

        if len(legacy_removed) != len(v2_removed):
            dedup_changes.append(
                {
                    "qid": qid,
                    "raw_prediction_count": len(predictions),
                    "legacy_removed_count": len(legacy_removed),
                    "v2_removed_count": len(v2_removed),
                    "newly_removed": v2_removed,
                }
            )

        per_query.append(
            {
                "qid": qid,
                "question": gold_records[qid].get("question"),
                "gold_count": len(gold),
                "raw_prediction_count": len(predictions),
                "legacy": {
                    "tp": legacy.tp,
                    "duplicates_removed": len(legacy_removed),
                },
                "strict_v2": {
                    "tp": v2.tp,
                    "duplicates_removed": len(v2_removed),
                },
                "tp_delta": v2.tp - legacy.tp,
                "dedup_delta": len(v2_removed) - len(legacy_removed),
                "possible_version_matches": len(audits),
            }
        )

    summary = {
        "gold_file": str(gold_path),
        "predictions_file": str(prediction_path),
        "query_count": len(qids),
        "legacy_strict_tp": legacy_tp,
        "strict_v2_tp": v2_tp,
        "tp_delta": v2_tp - legacy_tp,
        "legacy_duplicates_removed": legacy_removed_total,
        "v2_duplicates_removed": v2_removed_total,
        "dedup_delta": v2_removed_total - legacy_removed_total,
        "new_strict_v2_matches": len(new_matches),
        "possible_version_matches": len(possible_matches),
        "identifier_coverage": {
            "gold": _coverage(all_gold),
            "predictions": _coverage(all_predictions),
        },
        "api_calls": 0,
        "legacy_baseline_unchanged": True,
    }

    (output_dir / "identity_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, records in (
        ("per_query_identity_audit.jsonl", per_query),
        ("new_strict_v2_matches.jsonl", new_matches),
        ("possible_version_matches.jsonl", possible_matches),
        ("dedup_changes.jsonl", dedup_changes),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    coverage_rows: list[dict[str, Any]] = []
    for dataset_name, coverage in summary["identifier_coverage"].items():
        total = coverage.get("papers", 0)
        for field_name, count in sorted(coverage.items()):
            if field_name == "papers":
                continue
            coverage_rows.append(
                {
                    "dataset": dataset_name,
                    "field": field_name,
                    "count": count,
                    "total": total,
                    "ratio": count / total if total else 0.0,
                }
            )
    with (output_dir / "identifier_coverage.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "field", "count", "total", "ratio"]
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    review_lines = [
        "# ScholarPath Week 2 Day 2：论文身份与匹配审计",
        "",
        f"- 查询数：{summary['query_count']}",
        f"- Legacy strict TP：{legacy_tp}",
        f"- Strict v2 TP：{v2_tp}",
        f"- TP 变化：{summary['tp_delta']:+d}",
        f"- Legacy 去重数：{legacy_removed_total}",
        f"- V2 去重数：{v2_removed_total}",
        f"- 新增 V2 匹配：{len(new_matches)}",
        f"- 待人工复核版本候选：{len(possible_matches)}",
        "- API 调用：0",
        "",
        "## 判定原则",
        "",
        "- `strict` 保留第一周行为，用于结果可复现；",
        "- `strict_v2` 增加 DOI/URL/arXiv/OpenAlex 身份推断和版本感知去重；",
        "- 近似标题只进入人工审计，不直接计为 TP；",
        "- 只有强标识符或精确标题且元数据无冲突时，才自动匹配。",
        "",
        "## 下一步人工检查",
        "",
        "1. 查看 `new_strict_v2_matches.jsonl`，确认新增 TP 是否真实；",
        "2. 查看 `possible_version_matches.jsonl`，标注预印本/正式版或误匹配；",
        "3. 查看 `dedup_changes.jsonl`，确认新增去重没有误删不同论文；",
        "4. 审核通过后，再决定是否将正式评测模式切换为 `strict_v2`。",
    ]
    (output_dir / "day2_identity_review.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )

    print(f"Identity audit written to: {output_dir}")
    print(f"Query count: {len(qids)}")
    print(f"Legacy strict TP: {legacy_tp}")
    print(f"Strict v2 TP: {v2_tp}")
    print(f"TP delta: {v2_tp - legacy_tp:+d}")
    print(f"Legacy duplicates removed: {legacy_removed_total}")
    print(f"V2 duplicates removed: {v2_removed_total}")
    print(f"New strict-v2 matches: {len(new_matches)}")
    print(f"Possible version matches: {len(possible_matches)}")
    print("API calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
