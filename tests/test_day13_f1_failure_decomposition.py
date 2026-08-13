from __future__ import annotations

from scholarpath.evaluation.f1_failure_decomposition import (
    build_failure_row,
    classify_failure_family,
    derive_go_no_go,
    f1_from_counts,
    summarize_failure_rows,
)


def oracle_row(
    *,
    qid: str,
    gold_count: int,
    hit_ranks: list[int],
    best_k: int,
    best_f1: float,
) -> dict:
    return {
        "qid": qid,
        "question": qid,
        "gold_count": gold_count,
        "candidate_count": 100,
        "best_k": best_k,
        "best_f1": best_f1,
        "first_hit_rank": hit_ranks[0] if hit_ranks else None,
        "last_hit_rank": hit_ranks[-1] if hit_ranks else None,
        "hit_ranks": hit_ranks,
    }


def test_failure_taxonomy_is_mutually_exclusive() -> None:
    assert classify_failure_family(
        hit_ranks=[],
        oracle_f1=0.0,
        fixed_f1=0.0,
    )[0] == "retrieval_ceiling"

    assert classify_failure_family(
        hit_ranks=[30],
        oracle_f1=0.1,
        fixed_f1=0.0,
    )[0] == "deep_ranked_hit"

    assert classify_failure_family(
        hit_ranks=[8],
        oracle_f1=0.2,
        fixed_f1=0.0,
    )[0] == "mid_ranked_hit"

    assert classify_failure_family(
        hit_ranks=[2],
        oracle_f1=0.3,
        fixed_f1=0.1,
    )[0] == "cutoff_mismatch"

    assert classify_failure_family(
        hit_ranks=[2],
        oracle_f1=0.105,
        fixed_f1=0.1,
    )[0] == "already_good"


def test_build_failure_row_recomputes_fixed_and_adaptive_f1() -> None:
    row = build_failure_row(
        oracle_row(
            qid="q1",
            gold_count=4,
            hit_ranks=[2, 8],
            best_k=8,
            best_f1=1 / 3,
        ),
        adaptive_row={"qid": "q1", "predicted_k": 10},
        adaptive_query_f1=0.3,
        fixed_k=5,
        top_k=20,
    )

    assert row.tp_at_5 == 1
    assert row.tp_at_20 == 2
    assert row.family == "cutoff_mismatch"
    assert row.adaptive_predicted_k == 10
    assert row.cutoff_error == 2
    assert row.cutoff_direction == "over_return"
    assert row.adaptive_f1 == 0.3
    assert row.fixed_f1 == f1_from_counts(1, 5, 4)


def test_summary_preserves_analysis_boundary() -> None:
    rows = [
        build_failure_row(
            oracle_row(
                qid="q1",
                gold_count=3,
                hit_ranks=[],
                best_k=1,
                best_f1=0.0,
            )
        ),
        build_failure_row(
            oracle_row(
                qid="q2",
                gold_count=3,
                hit_ranks=[2],
                best_k=2,
                best_f1=0.4,
            ),
            adaptive_row={"predicted_k": 8},
        ),
    ]
    summary = summarize_failure_rows(rows)

    assert summary["query_count"] == 2
    assert (
        summary["family_summaries"]["retrieval_ceiling"][
            "query_count"
        ]
        == 1
    )
    assert summary["analysis_boundary"]["gold_used"] is True
    assert summary["analysis_boundary"]["production_inference"] is False
    assert summary["analysis_boundary"]["ranking_modified"] is False


def test_go_no_go_does_not_reward_failed_adaptive_selector() -> None:
    rows = [
        build_failure_row(
            oracle_row(
                qid="q1",
                gold_count=5,
                hit_ranks=[2],
                best_k=2,
                best_f1=0.2857142857,
            ),
            adaptive_row={"predicted_k": 20},
        ),
        build_failure_row(
            oracle_row(
                qid="q2",
                gold_count=5,
                hit_ranks=[3],
                best_k=3,
                best_f1=0.25,
            ),
            adaptive_row={"predicted_k": 20},
        ),
    ]
    summary = summarize_failure_rows(rows)
    decision = derive_go_no_go(summary)

    assert decision["adaptive_beats_fixed"] is False
    assert decision["primary_decision"] == "freeze_selector"
