from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.constraint_rerank import (
    annotate_and_rerank_record,
    compute_constraint_coverage,
    make_config,
)


def test_compute_constraint_coverage_matches_title_terms() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about scaling law of video text models"
    )
    paper = {
        "title": "Scaling Laws for Video Text Models",
        "arxiv_id": "2401.00001",
    }
    coverage = compute_constraint_coverage(paper, decomposition.constraints)
    assert coverage["coverage_count"] >= 1
    assert coverage["core_coverage_count"] >= 1


def test_annotate_and_rerank_record_adds_metadata() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about scaling law of video text models"
    )
    record = {
        "qid": "q1",
        "question": "papers about scaling law of video text models",
        "papers": [
            {"title": "Generic Paper", "arxiv_id": "2401.00002"},
            {"title": "Scaling Laws for Video Text Models", "arxiv_id": "2401.00001"},
        ],
    }

    reranked = annotate_and_rerank_record(
        record,
        decomposition,
        make_config("hybrid"),
    )
    assert reranked["papers"]
    assert "b2_1_constraint_score" in reranked["papers"][0]["raw"]
    assert "b2_1_covered_constraints" in reranked["papers"][0]["raw"]


def test_hybrid_can_promote_better_constraint_coverage() -> None:
    decomposition = ConstraintDecomposer().decompose(
        "papers about scaling law of video text models"
    )
    record = {
        "qid": "q1",
        "question": "papers about scaling law of video text models",
        "papers": [
            {"title": "Generic Paper", "arxiv_id": "2401.00002"},
            {"title": "Scaling Laws for Video Text Models", "arxiv_id": "2401.00001"},
        ],
    }

    reranked = annotate_and_rerank_record(
        record,
        decomposition,
        make_config("coverage_first"),
    )
    assert reranked["papers"][0]["title"] == "Scaling Laws for Video Text Models"
