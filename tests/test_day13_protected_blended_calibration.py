from __future__ import annotations

from scholarpath.evaluation.protected_blended_calibration import (
    assign_query_folds,
    blended_scores,
    candidate_identity_unchanged,
    rerank_papers,
    select_beta,
    source_rank_prior,
)


def paper(
    title: str,
    *,
    e3: float,
    source_rank: int | None,
) -> dict:
    return {
        "title": title,
        "normalized_title": title.lower(),
        "raw": {
            "day8_final_score": e3,
            "b4_best_source_rank": source_rank,
        },
    }


def test_source_rank_prior_is_bounded_and_monotonic() -> None:
    assert source_rank_prior(None) == 0.0
    assert source_rank_prior(0) == 0.0
    assert source_rank_prior(1) == 1.0
    assert source_rank_prior(2) < source_rank_prior(1)
    assert source_rank_prior(20) < source_rank_prior(2)


def test_beta_zero_preserves_frozen_e3_order() -> None:
    papers = [
        paper("A", e3=0.9, source_rank=50),
        paper("B", e3=0.8, source_rank=1),
    ]
    reranked = rerank_papers(papers, beta=0.0)
    assert [item["title"] for item in reranked] == ["A", "B"]


def test_small_blend_can_use_source_prior_without_changing_candidates() -> None:
    papers = [
        paper("A", e3=0.51, source_rank=100),
        paper("B", e3=0.50, source_rank=1),
    ]
    scores = blended_scores(papers, beta=0.10)
    assert len(scores) == 2
    reranked = rerank_papers(papers, beta=0.10)
    assert {item["title"] for item in reranked} == {"A", "B"}


def test_fold_assignment_is_deterministic_and_balanced() -> None:
    qids = [f"q{i}" for i in range(50)]
    first = assign_query_folds(qids, n_folds=5)
    second = assign_query_folds(reversed(qids), n_folds=5)
    assert first == second
    counts = [list(first.values()).count(fold) for fold in range(5)]
    assert counts == [10, 10, 10, 10, 10]


def test_select_beta_uses_f1_then_ndcg_then_smaller_beta() -> None:
    rows = [
        {"beta": 0.04, "macro_f1_at_5": 0.10, "ndcg_at_10": 0.20},
        {"beta": 0.02, "macro_f1_at_5": 0.10, "ndcg_at_10": 0.20},
        {"beta": 0.06, "macro_f1_at_5": 0.09, "ndcg_at_10": 0.30},
    ]
    assert select_beta(rows)["beta"] == 0.02
