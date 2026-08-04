from __future__ import annotations

import json
from pathlib import Path

from scholarpath.evaluation.error_analysis import (
    B0AnalysisConfig,
    analyze_b0,
    build_query_diagnostics,
    rank_bucket,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )


def test_rank_bucket() -> None:
    assert rank_bucket(1) == "1-10"
    assert rank_bucket(20) == "11-20"
    assert rank_bucket(21) == "21-50"
    assert rank_bucket(99) == "51-100"
    assert rank_bucket(101) == ">100"


def test_build_query_diagnostics() -> None:
    per_query = [
        {
            "qid": "q1",
            "question": "Query",
            "counts": {
                "gold": 2,
                "predictions_after_dedup": 3,
                "tp": 1,
                "fp": 2,
                "fn": 1,
            },
            "metrics": {
                "precision": 1 / 3,
                "recall": 0.5,
                "f1": 0.4,
                "recall@20": 0.0,
                "recall@50": 0.5,
                "recall@100": 0.5,
            },
            "matched_pairs": [{"prediction_rank": 31, "match_type": "strong_identifier"}],
            "false_positives": [{}, {}],
            "false_negatives": [{}],
        }
    ]
    logs = [{"qid": "q1", "api_calls": 1, "cache_hit": False, "latency_ms": 123.0}]
    diagnostics = build_query_diagnostics(per_query, logs)
    assert diagnostics[0]["qid"] == "q1"
    assert diagnostics[0]["best_matched_rank"] == 31
    assert diagnostics[0]["api_calls"] == 1


def test_analyze_b0(tmp_path: Path) -> None:
    b0_dir = tmp_path / "b0"
    out_dir = tmp_path / "analysis"

    run_summary = {
        "baseline": "B0",
        "provider": "openalex",
        "query_strategy": "raw_query_direct_search",
        "queries": {"total": 2, "succeeded": 2, "failed": 0},
        "retrieval": {"actual_api_calls": 2, "cache_hits": 0, "retries": 0},
        "tokens": {"total": 0},
        "estimated_cost_usd": 0.002,
        "latency_ms": {"wall_clock": 1000},
    }
    write_json(b0_dir / "run_summary.json", run_summary)
    write_jsonl(
        b0_dir / "query_logs.jsonl",
        [
            {"qid": "q1", "api_calls": 1, "cache_hit": False, "latency_ms": 100},
            {"qid": "q2", "api_calls": 1, "cache_hit": False, "latency_ms": 200},
        ],
    )

    summary_template = {
        "counts": {
            "tp": 1,
            "fp": 3,
            "fn": 2,
            "gold_papers": 3,
            "predictions_after_dedup": 4,
        },
        "macro": {
            "precision": 0.25,
            "recall": 0.5,
            "f1": 0.333333,
        },
        "micro": {
            "precision": 0.25,
            "recall": 0.333333,
            "f1": 0.285714,
        },
    }
    for topk in ("top20", "top50", "top100"):
        for mode in ("strict", "pasa_title"):
            write_json(
                b0_dir / "evaluation" / topk / mode / "summary.json",
                summary_template,
            )

    per_query = [
        {
            "qid": "q1",
            "question": "Query one",
            "counts": {
                "gold": 2,
                "predictions_after_dedup": 2,
                "tp": 1,
                "fp": 1,
                "fn": 1,
            },
            "metrics": {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "recall@20": 0.0,
                "recall@50": 0.5,
                "recall@100": 0.5,
            },
            "matched_pairs": [
                {
                    "prediction_rank": 31,
                    "match_type": "strong_identifier",
                    "prediction": {"title": "A"},
                    "gold": {"title": "A"},
                }
            ],
            "false_positives": [{"prediction_rank": 2, "paper": {"title": "FP"}}],
            "false_negatives": [{"gold_index": 1, "paper": {"title": "FN"}}],
        },
        {
            "qid": "q2",
            "question": "Query two",
            "counts": {
                "gold": 1,
                "predictions_after_dedup": 2,
                "tp": 0,
                "fp": 2,
                "fn": 1,
            },
            "metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "recall@20": 0.0,
                "recall@50": 0.0,
                "recall@100": 0.0,
            },
            "matched_pairs": [],
            "false_positives": [
                {"prediction_rank": 1, "paper": {"title": "FP1"}},
                {"prediction_rank": 2, "paper": {"title": "FP2"}},
            ],
            "false_negatives": [{"gold_index": 0, "paper": {"title": "Miss"}}],
        },
    ]
    write_jsonl(b0_dir / "evaluation" / "top100" / "strict" / "per_query.jsonl", per_query)

    result = analyze_b0(B0AnalysisConfig(b0_dir=b0_dir, output_dir=out_dir))

    assert result["error_analysis"]["query_count"] == 2
    assert result["error_analysis"]["zero_recall_queries"] == 1
    assert result["error_analysis"]["late_recall_queries"] == 1
    assert (out_dir / "b0_result_table.md").exists()
    assert (out_dir / "b1_recommendations.md").exists()
