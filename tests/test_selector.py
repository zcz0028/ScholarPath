from scholarpath.selector.selector import (
    build_reason_text,
    compute_selector_score,
    make_selector_config,
    select_papers,
)


def test_selector_scores_relevant_candidate_higher_than_low_evidence() -> None:
    config = make_selector_config("balanced")
    relevant = {
        "title": "Scaling Laws for Video Text Models",
        "arxiv_id": "2401.00001",
        "raw": {
            "b3_original_rank": 2,
            "b3_semantic_score": 0.45,
            "b3_final_score": 0.60,
            "b3_features": {
                "title_overlap": 0.6,
                "expanded_query_overlap": 0.5,
                "abstract_signal": 0.3,
                "constraint_phrase_coverage": 0.5,
                "core_constraint_coverage": 0.5,
                "semantic_score": 0.45,
            },
            "b2_1_constraint_score": 0.50,
            "b2_1_coverage_count": 3,
            "b2_1_coverage_weight": 3.5,
            "b2_1_core_coverage_count": 2,
        },
    }
    weak = {
        "title": "Generic Neural Networks",
        "arxiv_id": "2401.00002",
        "raw": {
            "b3_original_rank": 1,
            "b3_semantic_score": 0.02,
            "b3_final_score": 0.20,
            "b3_features": {
                "title_overlap": 0.0,
                "expanded_query_overlap": 0.0,
                "abstract_signal": 0.0,
                "constraint_phrase_coverage": 0.0,
                "core_constraint_coverage": 0.0,
                "semantic_score": 0.02,
            },
            "b2_1_constraint_score": 0.0,
            "b2_1_coverage_count": 0,
            "b2_1_coverage_weight": 0.0,
            "b2_1_core_coverage_count": 0,
        },
    }

    assert compute_selector_score(relevant, config)["selector_score"] > compute_selector_score(weak, config)["selector_score"]


def test_select_papers_adds_reason_fields() -> None:
    config = make_selector_config("ranking")
    papers = [
        {
            "title": "Scaling Laws for Video Text Models",
            "arxiv_id": "2401.00001",
            "raw": {
                "b3_original_rank": 1,
                "b3_semantic_score": 0.45,
                "b3_features": {
                    "title_overlap": 0.6,
                    "expanded_query_overlap": 0.5,
                    "abstract_signal": 0.3,
                    "constraint_phrase_coverage": 0.5,
                    "core_constraint_coverage": 0.5,
                    "semantic_score": 0.45,
                },
                "b2_1_constraint_score": 0.50,
                "b2_1_coverage_count": 3,
                "b2_1_coverage_weight": 3.5,
                "b2_1_core_coverage_count": 2,
            },
        }
    ]
    selected = select_papers(papers, config)
    raw = selected[0]["raw"]
    assert "b5_selector_score" in raw
    assert "b5_relevance_label" in raw
    assert "b5_reason_text" in raw


def test_build_reason_text_contains_label() -> None:
    reason = build_reason_text(
        "partially_relevant",
        ["semantic_score_medium", "title_matches_query"],
        ["missing_core_constraints"],
    )
    assert "部分相关" in reason
    assert "标题" in reason
