from scholarpath.selector.guard_aware_selector import (
    compute_guard_aware_score,
    make_guard_aware_config,
    select_guard_aware_papers,
)


def make_paper(title: str, selector_signal: float, guard_score: float, guard_decision: str):
    return {
        "title": title,
        "arxiv_id": title.replace(" ", "_"),
        "raw": {
            "b3_original_rank": 1,
            "b3_semantic_score": selector_signal,
            "b3_features": {
                "semantic_score": selector_signal,
                "title_overlap": selector_signal,
                "expanded_query_overlap": selector_signal,
                "abstract_signal": selector_signal,
                "constraint_phrase_coverage": selector_signal,
                "core_constraint_coverage": guard_score,
            },
            "b2_1_constraint_score": selector_signal,
            "b2_1_coverage_count": 2,
            "b2_1_coverage_weight": 2,
            "b2_1_core_coverage_count": 1,
            "b5_1_guard_score": guard_score,
            "b5_1_guard_decision": guard_decision,
            "b5_1_final_score": guard_score,
            "b5_1_missing_constraints": [] if guard_score >= 1 else ["missing"],
            "b5_1_violation_tags": [],
            "b5_1_guard_reason": "test reason",
        },
    }


def test_guard_aware_score_prefers_guard_pass_over_downrank() -> None:
    config = make_guard_aware_config("precision")
    good = make_paper("good paper", 0.4, 1.0, "pass")
    bad = make_paper("bad paper", 0.4, 0.0, "downrank")

    assert compute_guard_aware_score(good, config)["final_score"] > compute_guard_aware_score(bad, config)["final_score"]


def test_precision_filters_downrank_candidate() -> None:
    config = make_guard_aware_config("precision")
    papers = [
        make_paper("good paper", 0.4, 1.0, "pass"),
        make_paper("downrank paper", 0.4, 0.0, "downrank"),
    ]
    selected = select_guard_aware_papers(papers, config)
    titles = {paper["title"] for paper in selected}
    assert "good paper" in titles
    assert "downrank paper" not in titles


def test_ranking_keeps_all_candidates() -> None:
    config = make_guard_aware_config("ranking")
    papers = [
        make_paper("good paper", 0.4, 1.0, "pass"),
        make_paper("downrank paper", 0.1, 0.0, "downrank"),
    ]
    selected = select_guard_aware_papers(papers, config)
    assert len(selected) == 2
