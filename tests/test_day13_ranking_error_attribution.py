from __future__ import annotations

from scholarpath.evaluation.ranking_error_attribution import (
    build_query_attribution,
    derive_ranking_decision,
    extract_paper_features,
    feature_only_counterfactual,
    oriented_advantage,
    pairwise_feature_summary,
)


def paper(
    *,
    day8: float,
    b4_rank: int,
    b4_score: float = 0.0,
    title_overlap: float = 0.0,
) -> dict:
    return {
        "title": "x",
        "raw": {
            "day8_final_score": day8,
            "b4_best_source_rank": b4_rank,
            "b4_pre_rerank_score": b4_score,
            "b4_source_count": 1,
            "b3_semantic_score": day8,
            "b3_final_score": day8,
            "day8_raw_constraint_coverage": 0.0,
            "day8_canonical_constraint_coverage": 0.0,
            "day8_pure_semantic_features": {
                "query_overlap": day8,
                "expanded_query_overlap": day8,
                "title_overlap": title_overlap,
                "abstract_signal": 0.0,
            },
            "day8_constraint_evidence": [],
        },
    }


def test_extract_features_and_orientation() -> None:
    features = extract_paper_features(
        paper(day8=0.4, b4_rank=3, title_overlap=0.2)
    )
    assert features["day8_final_score"] == 0.4
    assert features["b4_best_source_rank"] == 3.0
    assert features["title_overlap"] == 0.2

    assert oriented_advantage(
        0.5,
        0.4,
        higher_is_better=True,
    ) > 0
    assert oriented_advantage(
        3,
        10,
        higher_is_better=False,
    ) > 0


def test_query_attribution_uses_false_positives_before_first_hit() -> None:
    papers = [
        paper(day8=0.9, b4_rank=50),
        paper(day8=0.8, b4_rank=40),
        paper(day8=0.7, b4_rank=2),
    ]
    row = build_query_attribution(
        qid="q1",
        question="q",
        family="mid_ranked_hit",
        gold_count=1,
        hit_ranks=[3],
        papers=papers,
    )
    assert row.first_hit_rank == 3
    assert row.blocker_count == 2
    assert (
        row.first_tp_advantages["day8_final_score"]
        < 0
    )
    assert (
        row.first_tp_advantages["b4_best_source_rank"]
        > 0
    )


def test_pairwise_summary_detects_orthogonal_b4_signal() -> None:
    records = {
        "q1": {
            "papers": [
                paper(day8=0.9, b4_rank=50),
                paper(day8=0.8, b4_rank=40),
                paper(day8=0.7, b4_rank=2),
            ]
        }
    }
    ranking_rows = [
        {
            "qid": "q1",
            "hit_ranks": [3],
        }
    ]
    summary = pairwise_feature_summary(
        ranking_rows=ranking_rows,
        paper_records_by_qid=records,
    )
    assert (
        summary["day8_final_score"][
            "pairwise_tp_win_rate_vs_blocking_fp"
        ]
        == 0.0
    )
    assert (
        summary["b4_best_source_rank"][
            "pairwise_tp_win_rate_vs_blocking_fp"
        ]
        == 1.0
    )


def test_counterfactual_is_analysis_only() -> None:
    records = {
        "q1": {
            "papers": [
                paper(day8=0.9, b4_rank=50),
                paper(day8=0.8, b4_rank=40),
                paper(day8=0.7, b4_rank=2),
            ]
        }
    }
    rows = [{"qid": "q1", "hit_ranks": [3]}]
    summary = feature_only_counterfactual(
        ranking_rows=rows,
        paper_records_by_qid=records,
    )
    assert summary["b4_best_source_rank"]["analysis_only"] is True
    assert (
        summary["b4_best_source_rank"][
            "feature_only_label_hits_at_5"
        ]
        == 1
    )


def test_decision_requires_separation_and_local_capture_gain() -> None:
    pairwise = {
        "day8_final_score": {
            "pairwise_tp_win_rate_vs_blocking_fp": 0.1
        },
        "b4_best_source_rank": {
            "pairwise_tp_win_rate_vs_blocking_fp": 0.62
        },
    }
    counterfactual = {
        "day8_final_score": {
            "delta_label_hits_at_20": 0,
            "delta_label_hits_at_5": 0,
        },
        "b4_best_source_rank": {
            "delta_label_hits_at_20": 4,
            "delta_label_hits_at_5": 4,
        },
    }
    decision = derive_ranking_decision(
        pairwise_summary=pairwise,
        counterfactual=counterfactual,
    )
    assert (
        decision["primary_decision"]
        == "small_blended_calibration_ablation_warranted"
    )
    assert (
        decision["candidate_signals"][0]["feature"]
        == "b4_best_source_rank"
    )
