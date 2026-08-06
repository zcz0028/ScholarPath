from __future__ import annotations

import pytest

from scholarpath.paper.schema import PaperRecord
from scholarpath.retrieval.rescue_retrieval import (
    RescuePlanSpec,
    aggregate_rescue_hits,
    preserve_or_replace_record,
    rescue_rank_score,
    validate_query_plan_records,
)


def make_plan(text: str = "test query", priority: int = 90, confidence: float = 0.9) -> RescuePlanSpec:
    return RescuePlanSpec(
        text=text,
        plan_type="task_method",
        priority=priority,
        confidence=confidence,
        reason="test",
        anchor_texts=("anchor",),
        anchor_types=("task",),
    )


def test_validate_plan_records_counts_budget() -> None:
    records = [
        {
            "qid": "q1",
            "planned_queries": [
                {"text": "query one", "estimated_api_calls": 1},
                {"text": "query two", "estimated_api_calls": 1},
            ],
        }
    ]
    result = validate_query_plan_records(records, max_plans_per_query=2, max_api_calls=2)
    assert result["query_count"] == 1
    assert result["plan_count"] == 2
    assert result["estimated_api_calls"] == 2
    assert result["production_retrieval_uses_gold"] is False


def test_validate_plan_records_rejects_gold_leakage() -> None:
    records = [
        {
            "qid": "q1",
            "gold_papers": [{"title": "secret"}],
            "planned_queries": [{"text": "query"}],
        }
    ]
    with pytest.raises(ValueError, match="forbidden gold"):
        validate_query_plan_records(records)


def test_validate_plan_records_enforces_api_budget() -> None:
    records = [
        {
            "qid": "q1",
            "planned_queries": [
                {"text": "query one"},
                {"text": "query two"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="exceed budget"):
        validate_query_plan_records(records, max_api_calls=1)


def test_plan_weight_uses_priority_and_confidence() -> None:
    strong = make_plan(priority=100, confidence=1.0)
    weak = make_plan(priority=20, confidence=0.2)
    assert strong.weight > weak.weight
    assert rescue_rank_score(1, strong) > rescue_rank_score(1, weak)


def test_rescue_rank_score_rewards_better_rank() -> None:
    plan = make_plan()
    assert rescue_rank_score(1, plan) > rescue_rank_score(20, plan)


def test_aggregate_rescue_hits_deduplicates_by_doi() -> None:
    first = PaperRecord(title="First title", doi="10.1000/demo")
    second = PaperRecord(title="Published title", doi="https://doi.org/10.1000/demo")
    papers = aggregate_rescue_hits(
        [
            (first, 5, make_plan("query one")),
            (second, 3, make_plan("query two")),
        ]
    )
    assert len(papers) == 1
    assert papers[0].raw["day4_occurrence_count"] == 2
    assert len(papers[0].raw["day4_plan_hits"]) == 2


def test_aggregate_rescue_hits_adds_multi_plan_reason() -> None:
    paper_a = PaperRecord(title="Same paper", openalex_id="W123")
    paper_b = PaperRecord(title="Same paper", openalex_id="https://openalex.org/W123")
    papers = aggregate_rescue_hits(
        [
            (paper_a, 10, make_plan("query one")),
            (paper_b, 20, make_plan("query two")),
        ]
    )
    assert "multi_plan_supported" in papers[0].raw["day4_reason_tags"]
    assert papers[0].raw["day4_best_rank"] == 10


def test_aggregate_rescue_hits_keeps_stronger_candidate_first() -> None:
    top = PaperRecord(title="Top", openalex_id="W1")
    low = PaperRecord(title="Low", openalex_id="W2")
    papers = aggregate_rescue_hits(
        [
            (low, 40, make_plan()),
            (top, 1, make_plan()),
        ]
    )
    assert papers[0].title == "Top"


def test_preserve_non_target_record_exactly() -> None:
    baseline = {"qid": "q1", "question": "question", "papers": [{"title": "base"}]}
    replacement = {"qid": "q1", "question": "question", "papers": [{"title": "new"}]}
    result = preserve_or_replace_record(
        qid="q1",
        target_qids={"q2"},
        baseline_record=baseline,
        replacement_record=replacement,
        top_k=20,
    )
    assert result == baseline


def test_replace_target_record_and_apply_top_k() -> None:
    baseline = {"qid": "q1", "question": "question", "papers": [{"title": "base"}]}
    replacement = {
        "qid": "q1",
        "question": "question",
        "papers": [{"title": "one"}, {"title": "two"}],
    }
    result = preserve_or_replace_record(
        qid="q1",
        target_qids={"q1"},
        baseline_record=baseline,
        replacement_record=replacement,
        top_k=1,
    )
    assert result["papers"] == [{"title": "one"}]
